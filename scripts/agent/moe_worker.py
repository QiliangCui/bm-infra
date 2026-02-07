import socket
import time
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding 
from jax.sharding import PartitionSpec as P
from functools import partial

from absl import app, flags
from google.cloud import spanner, pubsub_v1
from tpu_inference.kernels.fused_moe.v1.kernel import fused_ep_moe

_FAILED_OOM = -1
_UNKNOWN_ERROR = -100

_HOSTNAME = socket.gethostname()

# --- Flags Definition ---
_PROJECT_ID = flags.DEFINE_string('project_id', "cloud-tpu-inference-test", 'GCP Project ID')
_SUBSCRIPTION_ID = flags.DEFINE_string('subscription_id', "moe-tasks-sub", 'Pub/Sub Subscription ID')
_INSTANCE_ID = flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID')
_DATABASE_ID = flags.DEFINE_string('database_id', 'tune-moe', 'Spanner Database ID')
_WORKER_ID = flags.DEFINE_string('worker_id', _HOSTNAME, 'The worker id')
_DEBUG = flags.DEFINE_bool('debug', False, 'If true, prints results after each case iteration.')

# --- Global JAX Initialization ---
DEVICES = jax.local_devices()
MESH_CACHE = {}

def get_mesh(ep_size):
    """Caches JAX meshes to prevent expensive re-initialization."""
    if ep_size not in MESH_CACHE:
        mesh_devices = np.array(DEVICES)[:ep_size].reshape(1, ep_size)
        MESH_CACHE[ep_size] = jax.sharding.Mesh(mesh_devices, ('data', 'model'))
    return MESH_CACHE[ep_size]

# --- TPU Processing Logic ---

def process_on_tpu(config_row):
    """
    Executes TPU computation. Catches XLA VMEM OOM errors.
    Returns: (avg_latency_us, warmup_time_us, total_case_time_us)
    """
    # Unpack config including the 3 new columns [SQW1, SQW2, RenormalizeTopKLogits]
    if _DEBUG.value:
        print(f"  [DEBUG] Processing Config Row: {config_row}")
    (case_set_id, case_id, ep, tokens_count, h_size, inter_size, 
     experts_count, top_k, dtype_name, bt, btc, bf, bfc, 
     bd1, bd1c, bd2, bd2c, sqw1, sqw2, renorm) = config_row

    dtype = jnp.dtype(dtype_name)
    mesh = get_mesh(ep)
    
    # Static types as requested: scales are float32, tokens/gating are bfloat16
    scale_dtype = jnp.float32
    default_dtype = jnp.bfloat16

    # --- Sharded Generation Functions ---
    def gen_tokens(k):
        return jax.random.normal(k, (tokens_count, h_size), dtype=default_dtype)

    # Define sharded generation functions
    # (experts_count, 2, h_size, inter_size)
    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('model', None, None, None)))
    def gen_w1(k): return jax.random.normal(k, (experts_count, 2, h_size, inter_size), dtype=dtype)

    # (experts_count, inter_size, h_size)
    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('model', None, None)))
    def gen_w2(k): return jax.random.normal(k, (experts_count, inter_size, h_size), dtype=dtype)

    # (tokens_count, experts_count),
    @partial(jax.jit, out_shardings=NamedSharding(mesh, P(None, 'model')))
    def gen_gating(k): return jax.random.normal(k, (tokens_count, experts_count), dtype=default_dtype)


    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('model', None, None, None, None)))
    def gen_scale_w1(k): return jax.random.normal(k, (experts_count, 2, h_size // sqw1, 1, inter_size), dtype=scale_dtype)

    @partial(jax.jit, out_shardings=NamedSharding(mesh, P('model', None, None, None)))
    def gen_scale_w2(k): return jax.random.normal(k, (experts_count, inter_size // sqw2, 1, h_size), dtype=scale_dtype)

    key = jax.random.PRNGKey(int(time.time()))
    
    # 1. Generate core tensors
    tokens = gen_tokens(key)
    w1 = gen_w1(key)
    w2 = gen_w2(key)
    gating_output = gen_gating(key)

    # 2. Handle Quantization Scales
    w1_scale = None
    subc_quant_w1_sz = None
    if sqw1 > 0:
        subc_quant_w1_sz = sqw1
        w1_scale = gen_scale_w1(key)

    w2_scale = None
    subc_quant_w2_sz = None
    if sqw2 > 0:
        subc_quant_w2_sz = sqw2        
        w2_scale = gen_scale_w2(key)

    kwargs = {
        'bt': bt, 'btc': btc if btc != -1 else bt,
        'bf': bf, 'bfc': bfc if bfc != -1 else bf,
        'bd1': bd1, 'bd1c': bd1c if bd1c != -1 else bd1,
        'bd2': bd2, 'bd2c': bd2c if bd2c != -1 else bd2,
        'subc_quant_w1_sz': subc_quant_w1_sz,
        'subc_quant_w2_sz': subc_quant_w2_sz,
        'w1_scale': w1_scale,
        'w2_scale': w2_scale,
        'renormalize_topk_logits': renorm,
    }

    start_case = time.perf_counter()
    try:
        # Warm-up
        warmup_start = time.perf_counter()
        fused_ep_moe(mesh, tokens, w1, w2, gating_output, top_k, **kwargs).block_until_ready()
        warmup_end = time.perf_counter()
        warmup_time_us = int((warmup_end - warmup_start) * 1_000_000)

        # Measured Run
        num_iters = 10
        iter_start = time.perf_counter()
        for _ in range(num_iters):
            fused_ep_moe(mesh, tokens, w1, w2, gating_output, top_k, **kwargs).block_until_ready()
        iter_end = time.perf_counter()
        
        avg_latency_us = int(((iter_end - iter_start) / num_iters) * 1_000_000)
        total_case_time_us = int((time.perf_counter() - start_case) * 1_000_000)
        
        return avg_latency_us, warmup_time_us, total_case_time_us

    except Exception as e:
        error_msg = str(e)
        if "RESOURCE_EXHAUSTED" in error_msg or "vmem" in error_msg:
            # Sentinel for OOM
            return _FAILED_OOM, 0, int((time.perf_counter() - start_case) * 1_000_000)
        
        print(f"!!! Fatal Kernel Error on Case {case_id}: {e}")
        return _UNKNOWN_ERROR, 0, int((time.perf_counter() - start_case) * 1_000_000)

# --- Spanner Management ---

class SpannerManager:
    def __init__(self):
        self.client = spanner.Client(project=_PROJECT_ID.value, disable_builtin_metrics=True)
        self.instance = self.client.instance(_INSTANCE_ID.value)
        self.db = self.instance.database(_DATABASE_ID.value)

    def mark_bucket_in_progress(self, case_set_id, run_id, bucket_id):
        def _do_update(transaction):
            transaction.execute_update(
                "UPDATE WorkBuckets SET Status = 'IN_PROGRESS', WorkerID = @wid, UpdatedAt = PENDING_COMMIT_TIMESTAMP() "
                "WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
                params={'id': case_set_id, 'rid': run_id, 'bid': bucket_id, 'wid': _WORKER_ID.value},
                param_types={'id': spanner.param_types.STRING, 'rid': spanner.param_types.STRING, 
                             'bid': spanner.param_types.INT64, 'wid': spanner.param_types.STRING}
            )
        self.db.run_in_transaction(_do_update)

    def mark_bucket_completed(self, case_set_id, run_id, bucket_id, bucket_total_time_us):
        def _do_update(transaction):
            transaction.execute_update(
                "UPDATE WorkBuckets SET Status = 'COMPLETED', TotalTime = @tt, UpdatedAt = PENDING_COMMIT_TIMESTAMP() "
                "WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
                params={'id': case_set_id, 'rid': run_id, 'bid': bucket_id, 'tt': bucket_total_time_us},
                param_types={'id': spanner.param_types.STRING, 'rid': spanner.param_types.STRING, 
                             'bid': spanner.param_types.INT64, 'tt': spanner.param_types.INT64}
            )
        self.db.run_in_transaction(_do_update)

    def get_already_processed_ids(self, case_set_id, run_id, start, end):
        query = ("SELECT CaseId FROM CaseResults "
                 "WHERE ID = @id AND RunId = @rid AND CaseId BETWEEN @s AND @e")
        with self.db.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={'id': case_set_id, 'rid': run_id, 's': start, 'e': end},
                param_types={'id': spanner.param_types.STRING, 'rid': spanner.param_types.STRING, 
                             's': spanner.param_types.INT64, 'e': spanner.param_types.INT64}
            )
            return {row[0] for row in results}

    def get_bucket_configs(self, case_set_id, start, end):
        # Updated query to include SQW1, SQW2, RenormalizeTopKLogits
        query = (
            "SELECT ID, CaseId, EP, NumTokens, HiddenSize, IntermediateSize, "
            "NumExpertes, TopK, DType, BT, BTC, BF, BFC, BD1, BD1C, BD2, BD2C, "
            "SQW1, SQW2, RenormalizeTopKLogits "
            "FROM Cases WHERE ID = @id AND CaseId BETWEEN @s AND @e "
            "ORDER BY CaseId ASC"
        )
        with self.db.snapshot() as snapshot:
            results = snapshot.execute_sql(
                query,
                params={'id': case_set_id, 's': start, 'e': end},
                param_types={'id': spanner.param_types.STRING, 's': spanner.param_types.INT64, 'e': spanner.param_types.INT64}
            )
            return {row[1]: row for row in results}

    def save_results_batch(self, results_list):
        if not results_list: return
        with self.db.batch() as b:
            b.insert_or_update(
                table='CaseResults',
                columns=('ID', 'RunId', 'CaseId', 'ProcessedStatus', 'WorkerID', 'Latency', 'WarmupTime', 'TotalTime', 'ProcessedAt'),
                values=results_list
            )

# --- Pub/Sub Callback Interface ---

def get_callback(spanner_mgr):
    def callback(message):
        try:
            bucket_start_perf = time.perf_counter()
            data = message.data.decode("utf-8").split("|")
            case_set_id, run_id, bucket_id, start, end = data[0], data[1], int(data[2]), int(data[3]), int(data[4])
            
            print(f"[{_WORKER_ID.value}] Claimed CaseSetId: {case_set_id}, RunId: {run_id}, Bucket {bucket_id} ({start}-{end})")
            spanner_mgr.mark_bucket_in_progress(case_set_id, run_id, bucket_id)

            processed_ids = spanner_mgr.get_already_processed_ids(case_set_id, run_id, start, end)
            all_configs = spanner_mgr.get_bucket_configs(case_set_id, start, end)

            results_buffer = []
            for cid in range(start, end + 1):
                if cid in processed_ids: continue
                config = all_configs.get(cid)
                if not config: continue

                latency, warmup, total_case = process_on_tpu(config)
                
                if latency == _FAILED_OOM:
                    status = "FAILED_OOM"
                elif latency == _UNKNOWN_ERROR:
                    status = "UNKNOWN_ERROR"
                else:
                    status = "SUCCESS" 
                
                results_buffer.append(
                    (case_set_id, run_id, cid, status, _WORKER_ID.value, latency, warmup, total_case, spanner.COMMIT_TIMESTAMP)
                )

                if _DEBUG.value:
                    if latency == sys.maxsize:
                        print(f"  [DEBUG] Case {cid}: AvgLat=OOM, Warmup={warmup:,}us, Total={total_case:,}us")
                    else:
                        # Apply comma formatting to the integer latency before printing
                        print(f"  [DEBUG] Case {cid}: AvgLat={latency:,}us, Warmup={warmup:,}us, Total={total_case:,}us")

                if len(results_buffer) >= 10:
                    spanner_mgr.save_results_batch(results_buffer)
                    results_buffer = []

            spanner_mgr.save_results_batch(results_buffer)
            
            bucket_total_time_us = int((time.perf_counter() - bucket_start_perf) * 1_000_000)
            spanner_mgr.mark_bucket_completed(case_set_id, run_id, bucket_id, bucket_total_time_us)
            
            message.ack()
            print(f"[{_WORKER_ID.value}] Bucket {bucket_id} COMPLETED in {bucket_total_time_us/1e6:.2f}s.")

        except Exception as e:
            print(f"!!! Fatal Worker Error: {e}")
            message.nack()

    return callback

# --- Main Entry Point ---

def main(argv):
    spanner_mgr = SpannerManager()
    subscriber = pubsub_v1.SubscriberClient()
    sub_path = subscriber.subscription_path(_PROJECT_ID.value, _SUBSCRIPTION_ID.value)

    # flow_control: Pull 1 bucket at a time, lease heartbeat for 6 hours
    flow_control = pubsub_v1.types.FlowControl(max_messages=1, max_lease_duration=21600)

    streaming_pull_future = subscriber.subscribe(
        sub_path, callback=get_callback(spanner_mgr), flow_control=flow_control
    )

    print(f"TPU Worker {_WORKER_ID.value} ready (DEBUG={_DEBUG.value})")
    with subscriber:
        try:
            streaming_pull_future.result()
        except KeyboardInterrupt:
            streaming_pull_future.cancel()

if __name__ == '__main__':
    app.run(main)