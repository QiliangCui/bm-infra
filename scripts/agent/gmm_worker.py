import socket
import time
import sys
import jax
import jax.numpy as jnp
from absl import app, flags
from google.cloud import spanner, pubsub_v1
from tpu_inference.kernels.megablox.gmm import gmm

_HOSTNAME = socket.gethostname()
_PROJECT_ID = flags.DEFINE_string('project_id', "cloud-tpu-inference-test", 'GCP Project ID')
_SUBSCRIPTION_ID = flags.DEFINE_string('subscription_id', "gmm-tasks-sub", 'Pub/Sub Subscription ID')
_INSTANCE_ID = flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID')
_DATABASE_ID = flags.DEFINE_string('database_id', 'tune-gmm', 'Spanner Database ID')
_WORKER_ID = flags.DEFINE_string('worker_id', _HOSTNAME, 'The worker id')
_DEBUG = flags.DEFINE_bool('debug', False, 'If true, prints results after each case iteration.')

DEVICES = jax.local_devices()
BENCHMARK_DEVICE = DEVICES[0]
_CACHED_TENSORS = {}
_LAST_STRUCTURAL_CONFIG = None

def clear_tensor_cache():
    global _CACHED_TENSORS
    for key in list(_CACHED_TENSORS.keys()): del _CACHED_TENSORS[key]
    _CACHED_TENSORS = {}

def process_on_tpu(config_row):
    global _LAST_STRUCTURAL_CONFIG
    (cs_id, case_id, m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn) = config_row
    
    if _DEBUG.value:
        print(f"  [DEBUG] Processing Config Row: {config_row}")

    structural_config = (m, k, n, cg, ldt, rdt, q_block)
    
    if structural_config != _LAST_STRUCTURAL_CONFIG:
        if _DEBUG.value:
            print(f"  [DEBUG] Structural config changed: {_LAST_STRUCTURAL_CONFIG} -> {structural_config}. Re-generating tensors.")
        clear_tensor_cache()
        _LAST_STRUCTURAL_CONFIG = structural_config
        key = jax.random.PRNGKey(int(time.time()))
        with jax.default_device(BENCHMARK_DEVICE):
            _CACHED_TENSORS['lhs'] = jax.random.normal(key, (m, k), dtype=jnp.dtype(ldt))
            _CACHED_TENSORS['rhs'] = jax.random.normal(key, (cg, k, n), dtype=jnp.dtype(rdt))
            _CACHED_TENSORS['rhs_scale'] = jax.random.uniform(key, (cg, k // q_block, 1, n), dtype=jnp.float32)
            
            # m: the total tokens
            # cg: current group size, say 8 core to host 160 experts model, the cg is 20.
            # avg_size = m // cg so that the m tokens can be used up.
            avg_size = m // cg
            gs = jnp.full((cg,), avg_size, dtype=jnp.int32)
            if m % cg: gs = gs.at[-1].add(m - int(gs.sum()))
            _CACHED_TENSORS['group_sizes'] = gs
            _CACHED_TENSORS['group_offset'] = jnp.array(0, dtype=jnp.int32)

    lhs, rhs, rhs_scale = _CACHED_TENSORS['lhs'], _CACHED_TENSORS['rhs'], _CACHED_TENSORS['rhs_scale']
    group_sizes, group_offset = _CACHED_TENSORS['group_sizes'], _CACHED_TENSORS['group_offset']
    tiling = (tm, tk, tn)
    start_case = time.perf_counter()
    try:
        # Warm-up
        warmup_start = time.perf_counter()
        gmm(lhs, rhs, group_sizes, preferred_element_type=lhs.dtype, 
            rhs_scale=rhs_scale, tiling=tiling, group_offset=group_offset).block_until_ready()
        warmup_end = time.perf_counter()
        warmup_time_us = int((warmup_end - warmup_start) * 1_000_000)

        # Measured Run
        num_iters = 10
        iter_start = time.perf_counter()
        for _ in range(num_iters):
            gmm(lhs, rhs, group_sizes, preferred_element_type=lhs.dtype, 
                rhs_scale=rhs_scale, tiling=tiling, group_offset=group_offset).block_until_ready()
        iter_end = time.perf_counter()
        
        avg_latency = int(((iter_end - iter_start) / num_iters) * 1_000_000)
        total_case_time_us = int((time.perf_counter() - start_case) * 1_000_000)
        return avg_latency, warmup_time_us, total_case_time_us

    except Exception as e:
        error_msg = str(e).lower()
        if "resource_exhausted" in error_msg or "vmem" in error_msg:
            if _DEBUG.value:
                print(f"  [DEBUG] Case {case_id} FAILED with VMEM OOM.")
            return sys.maxsize, 0, int((time.perf_counter() - start_case) * 1_000_000)
        raise e

class SpannerManager:
    def __init__(self):
        self.client = spanner.Client(project=_PROJECT_ID.value, disable_builtin_metrics=True)
        self.db = self.client.instance(_INSTANCE_ID.value).database(_DATABASE_ID.value)

    def mark_bucket_in_progress(self, cs_id, r_id, b_id):
        self.db.run_in_transaction(lambda tx: tx.execute_update(
            "UPDATE WorkBuckets SET Status = 'IN_PROGRESS', WorkerID = @wid, UpdatedAt = PENDING_COMMIT_TIMESTAMP() WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
            params={'id': cs_id, 'rid': r_id, 'bid': b_id, 'wid': _WORKER_ID.value}))

    def mark_bucket_completed(self, cs_id, r_id, b_id, tt_us):
        self.db.run_in_transaction(lambda tx: tx.execute_update(
            "UPDATE WorkBuckets SET Status = 'COMPLETED', TotalTime = @tt, UpdatedAt = PENDING_COMMIT_TIMESTAMP() WHERE ID = @id AND RunId = @rid AND BucketId = @bid",
            params={'id': cs_id, 'rid': r_id, 'bid': b_id, 'tt': tt_us}))

    def get_already_processed_ids(self, cs_id, r_id, start, end):
        query = "SELECT CaseId FROM CaseResults WHERE ID = @id AND RunId = @rid AND CaseId BETWEEN @s AND @e"
        with self.db.snapshot() as snp:
            return {row[0] for row in snp.execute_sql(query, params={'id': cs_id, 'rid': r_id, 's': start, 'e': end})}

    def get_bucket_configs(self, cs_id, start, end):
        query = "SELECT ID, CaseId, M, K, N, NumTotalGroups, NumCurrentGroups, LhsDType, RhsDType, QuantBlockSize, TM, TK, TN FROM Cases WHERE ID = @id AND CaseId BETWEEN @s AND @e ORDER BY CaseId ASC"
        with self.db.snapshot() as snp:
            return {row[1]: row for row in snp.execute_sql(query, params={'id': cs_id, 's': start, 'e': end})}

    def save_results_batch(self, results):
        if not results: return
        with self.db.batch() as b:
            b.insert_or_update(table='CaseResults', columns=('ID', 'RunId', 'CaseId', 'ProcessedStatus', 'WorkerID', 'Latency', 'WarmupTime', 'TotalTime', 'ProcessedAt'), values=results)

def get_callback(spanner_mgr):
    def callback(message):
        try:
            bucket_start_perf = time.perf_counter()
            cs_id, r_id, b_id, start, end = message.data.decode("utf-8").split("|")
            b_id, start, end = int(b_id), int(start), int(end)
            
            if _DEBUG.value:
                print(f"[{_WORKER_ID.value}] Claimed Bucket {b_id} ({start}-{end}) for CaseSet: {cs_id}")
            
            spanner_mgr.mark_bucket_in_progress(cs_id, r_id, b_id)
            processed_ids = spanner_mgr.get_already_processed_ids(cs_id, r_id, start, end)
            all_configs = spanner_mgr.get_bucket_configs(cs_id, start, end)
            results = []
            
            for cid in range(start, end + 1):
                if cid in processed_ids: continue
                config = all_configs.get(cid)
                if not config: continue
                
                latency, warmup, total = process_on_tpu(config)
                status = "SUCCESS" if latency != sys.maxsize else "FAILED_OOM"
                
                results.append((cs_id, r_id, cid, status, _WORKER_ID.value, latency, warmup, total, spanner.COMMIT_TIMESTAMP))
                
                if _DEBUG.value:
                    lat_str = "OOM" if latency == sys.maxsize else f"{latency:,}us"
                    print(f"  [DEBUG] Case {cid}: AvgLat={lat_str}, Warmup={warmup:,}us, Total={total:,}us")

                if len(results) >= 10:
                    spanner_mgr.save_results_batch(results)
                    results = []
            
            spanner_mgr.save_results_batch(results)
            bucket_tt_us = int((time.perf_counter() - bucket_start_perf) * 1_000_000)
            spanner_mgr.mark_bucket_completed(cs_id, r_id, b_id, bucket_tt_us)
            
            if _DEBUG.value:
                print(f"[{_WORKER_ID.value}] Bucket {b_id} COMPLETED in {bucket_tt_us/1e6:.2f}s.")
            
            message.ack()
        except Exception as e:
            print(f"!!! Error in callback: {e}"); message.nack()
    return callback

def main(argv):
    spanner_mgr = SpannerManager()
    sub = pubsub_v1.SubscriberClient()
    streaming_pull_future = sub.subscribe(sub.subscription_path(_PROJECT_ID.value, _SUBSCRIPTION_ID.value), callback=get_callback(spanner_mgr), flow_control=pubsub_v1.types.FlowControl(max_messages=1, max_lease_duration=21600))
    print(f"GMM Worker {_WORKER_ID.value} ready (DB={_DATABASE_ID.value}, DEBUG={_DEBUG.value})")
    with sub:
        try: streaming_pull_future.result()
        except KeyboardInterrupt: streaming_pull_future.cancel()

if __name__ == '__main__': app.run(main)