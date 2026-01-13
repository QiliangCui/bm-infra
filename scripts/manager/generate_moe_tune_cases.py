import itertools
import math
import sys
import time
from absl import app, flags
from google.cloud import spanner
from google.api_core import retry, exceptions
from tqdm import tqdm
import jax.numpy as jnp
from jax._src import dtypes

# --- Experiment Metadata Flags ---
_CASE_SET_ID = flags.DEFINE_string('case_set_id', 'test1', 'Unique ID for this experiment run.')
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '', 'Manual description (if empty, auto-generates from flags).')
_DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'If True, skip all Spanner operations.')

# --- Model Configuration Flags ---
_EP_SIZES = flags.DEFINE_list('ep_sizes', ['4'], 'EP sizes')
_NUM_TOKENS_LIST = flags.DEFINE_list('num_tokens_list', ['128'], 'Tokens')
_HIDDEN_SIZE_LIST = flags.DEFINE_list('hidden_size_list', ['3072'], 'Hidden sizes')
_INTERMEDIATE_SIZE_LIST = flags.DEFINE_list('intermediate_size_list', ['3072'], 'Intermediate')
_NUM_EXPERTS_LIST = flags.DEFINE_list('num_experts_list', ['128'], 'Experts')
_TOP_K_LIST = flags.DEFINE_list('top_k_list', ['4'], 'Top K')
_DTYPE_LIST = flags.DEFINE_list('dtype_list', ['bfloat16'], 'Dtypes')

# --- Tiling Search Space Flags ---
_BT_LST = flags.DEFINE_list('bt_lst', ['16', '32', '64', '128', '256'], 'bt')
_BF_LST = flags.DEFINE_list('bf_lst', ['1280', '1536'], 'bf')
_BD1_LST = flags.DEFINE_list('bd1_lst', ['1280', '1536'], 'bd1')
_BD2_LST = flags.DEFINE_list('bd2_lst', ['1280', '1536'], 'bd2')
_BTC_LST = flags.DEFINE_list('btc_lst', ['16', '32', '64', '128', '256'], 'btc')
_BFC_LST = flags.DEFINE_list('bfc_lst', ['-1'], 'Chunk sizes for features (-1 to copy bf).')
_BD1C_LST = flags.DEFINE_list('bd1c_lst', ['-1'], 'Chunk sizes for hidden dim 1 (-1 to copy bd1).')
_BD2C_LST = flags.DEFINE_list('bd2c_lst', ['-1'], 'Chunk sizes for hidden dim 2 (-1 to copy bd2).')

# --- Spanner Connection Config ---
SPANNER_INSTANCE = 'vllm-bm-inst'
SPANNER_DATABASE = 'tune-moe'
BATCH_SIZE = 1000  # Number of rows per batch mutation

# --- Spanner Management ---

class SpannerBatchInserter:
    def __init__(self, instance_id, database_id):
        # We only initialize the client/database if not a dry run
        if not _DRY_RUN.value:
            self.client = spanner.Client(disable_builtin_metrics=True)
            self.instance = self.client.instance(instance_id)
            self.database = self.instance.database(database_id)
        else:
            self.database = None
        
        self.current_case_id = 0
        self.invalid_count = 0
        self.buffer = []

    @retry.Retry(predicate=retry.if_transient_error)
    def flush(self):
        """Writes current buffer to Spanner Cases table."""
        if not self.buffer:
            return
        
        # SKIP network call if dry run
        if _DRY_RUN.value:
            self.buffer = []
            return

        with self.database.batch() as b:
            b.insert(
                table='Cases',
                columns=(
                    'ID', 'CaseId', 'EP', 'NumTokens', 'HiddenSize', 'IntermediateSize', 
                    'NumExpertes', 'TopK', 'DType', 'BT', 'BTC', 'BF', 'BFC', 
                    'BD1', 'BD1C', 'BD2', 'BD2C'
                ),
                values=self.buffer
            )
        self.buffer = []

    def add_row(self, row):
        self.buffer.append(row)
        self.current_case_id += 1
        if len(self.buffer) >= BATCH_SIZE:
            self.flush()

def generate_summary_desc():
    """Generates a summary string of all flags defined in this script."""
    summary_parts = []
    current_module_flags = flags.FLAGS.get_key_flags_for_module(__name__)
    for flag in sorted(current_module_flags, key=lambda f: f.name):
        if flag.name in ['case_set_id', 'case_set_desc']:
            continue
        val = flag.value
        val_str = ",".join(map(str, val)) if isinstance(val, list) else str(val)
        summary_parts.append(f"{flag.name}={val_str}")
    return " | ".join(summary_parts)

def init_parent_creating(database, case_set_id, scan_space):
    """Initializes the CaseSet record as 'CREATING'. Fails if ID exists."""
    if _DRY_RUN.value: return # SKIP for dry run

    final_desc = _CASE_SET_DESC.value or generate_summary_desc()
    def _do_insert(transaction):
        row = transaction.execute_sql(
            "SELECT ID FROM CaseSet WHERE ID = @id",
            params={'id': case_set_id},
            param_types={'id': spanner.param_types.STRING}
        ).one_or_none()
        
        if row:
            raise exceptions.AlreadyExists(f"CaseSet ID '{case_set_id}' already exists.")
        
        transaction.execute_update(
            "INSERT INTO CaseSet (ID, Description, Status, ScanSpace) VALUES (@id, @desc, 'CREATING', @ss)",
            params={'id': case_set_id, 'desc': final_desc, 'ss': scan_space},
            param_types={'id': spanner.param_types.STRING, 'desc': spanner.param_types.STRING, 'ss': spanner.param_types.INT64}
        )
    database.run_in_transaction(_do_insert)

def finish_parent_completed(database, case_set_id, valid, invalid, duration):
    """Finalizes CaseSet record with stats and 'COMPLETED' status."""
    if _DRY_RUN.value: return # SKIP for dry run

    def _do_update(transaction):
        transaction.execute_update(
            "UPDATE CaseSet SET Status = 'COMPLETED', Valid = @v, Invalid = @i, DurationSeconds = @d WHERE ID = @id",
            params={'id': case_set_id, 'v': valid, 'i': invalid, 'd': duration},
            param_types={'id': spanner.param_types.STRING, 'v': spanner.param_types.INT64, 'i': spanner.param_types.INT64, 'd': spanner.param_types.FLOAT64}
        )
    database.run_in_transaction(_do_update)

# --- Logic & Constraints ---

def get_dtype_packing(dtype):
    bits = (dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))
    return 32 // bits

def is_valid_config(c, num_tokens, ep_size, t_packing, hidden_size, intermediate_size):
    local_num_tokens = num_tokens // ep_size
    if local_num_tokens < t_packing * 8 or local_num_tokens < c['bt']: return False
    if c['btc'] > c['bt'] or c['bt'] % c['btc'] != 0: return False
    if c['bfc'] > c['bf'] or c['bf'] % c['bfc'] != 0 or c['bfc'] % 128 != 0: return False
    if c['bd1c'] > c['bd1'] or c['bd1c'] % (t_packing * 128) != 0 or c['bd1'] % c['bd1c'] != 0: return False
    if c['bd2c'] > c['bd2'] or c['bd2c'] % (t_packing * 128) != 0 or c['bd2'] % c['bd2c'] != 0: return False
    if hidden_size % c['bd1'] != 0 or hidden_size % c['bd2'] != 0 or intermediate_size % c['bf'] != 0: return False
    return True

# --- Main ---

def main(argv):
    if _DRY_RUN.value:
        print(">>> DRY RUN MODE ENABLED: No data will be written to Spanner. <<<")

    inserter = SpannerBatchInserter(SPANNER_INSTANCE, SPANNER_DATABASE)

    # 1. Setup flat parameter space
    input_params = [
        [int(x) for x in _EP_SIZES.value], [int(x) for x in _NUM_TOKENS_LIST.value],
        [int(x) for x in _HIDDEN_SIZE_LIST.value], [int(x) for x in _INTERMEDIATE_SIZE_LIST.value],
        [int(x) for x in _NUM_EXPERTS_LIST.value], [int(x) for x in _TOP_K_LIST.value],
        [jnp.dtype(s) for s in _DTYPE_LIST.value]
    ]
    tiling_params = [
        [int(x) for x in _BT_LST.value], [int(x) for x in _BF_LST.value],
        [int(x) for x in _BD1_LST.value], [int(x) for x in _BD2_LST.value],
        [int(x) for x in _BTC_LST.value], [int(x) for x in _BFC_LST.value],
        [int(x) for x in _BD1C_LST.value], [int(x) for x in _BD2C_LST.value]
    ]
    all_param_lists = input_params + tiling_params
    total_combinations = math.prod(len(lst) for lst in all_param_lists)

    # 2. Check existence and set status to CREATING
    try:
        init_parent_creating(inserter.database, _CASE_SET_ID.value, total_combinations)
        if not _DRY_RUN.value:
            print(f"Initialized CaseSet: {_CASE_SET_ID.value} (Status: CREATING)")
    except exceptions.AlreadyExists as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)

    start_time = time.time()

    # 3. Stream through the flat product
    with tqdm(total=total_combinations, desc="Processing Cases", unit="cfg") as pbar:
        for vals in itertools.product(*all_param_lists):
            pbar.update(1)
            
            # Unpack values
            ep, tokens, h, inter, experts, top_k, dtype = vals[0:7]
            bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = vals[7:15]

            # Logic for -1 (copy base block size)
            if bfc == -1: bfc = bf
            if bd1c == -1: bd1c = bd1
            if bd2c == -1: bd2c = bd2

            kwargs = {'bt':bt, 'bf':bf, 'bd1':bd1, 'bd2':bd2, 'btc':btc, 'bfc':bfc, 'bd1c':bd1c, 'bd2c':bd2c}
            t_packing = get_dtype_packing(dtype)

            if is_valid_config(kwargs, tokens, ep, t_packing, h, inter):
                row = (
                    _CASE_SET_ID.value, inserter.current_case_id,
                    ep, tokens, h, inter, experts, top_k, str(dtype.name),
                    bt, btc, bf, bfc, bd1, bd1c, bd2, bd2c
                )
                inserter.add_row(row)
            else:
                inserter.invalid_count += 1

    # 4. Final flush and update status
    inserter.flush()
    duration = time.time() - start_time
    finish_parent_completed(inserter.database, _CASE_SET_ID.value, inserter.current_case_id, inserter.invalid_count, duration)

    if _DRY_RUN.value:
        print(f"\n[DRY RUN COMPLETE]")
    else:
        print(f"\nDone! CaseSet '{_CASE_SET_ID.value}' Status updated to COMPLETED.")
    
    print(f"Stats: Valid={inserter.current_case_id:,} | Invalid={inserter.invalid_count:,} | Duration={duration:.2f}s")

if __name__ == '__main__':
    app.run(main)