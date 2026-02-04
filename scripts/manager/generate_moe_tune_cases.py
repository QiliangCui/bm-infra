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
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '', 'Manual description.')
_DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'Skip Spanner operations.')

# --- Model Configuration Flags ---
_EP_SIZES = flags.DEFINE_list('ep_sizes', ['4'], 'EP sizes')
_NUM_TOKENS_LIST = flags.DEFINE_list('num_tokens_list', ['128'], 'Tokens')
_HIDDEN_SIZE_LIST = flags.DEFINE_list('hidden_size_list', ['3072'], 'Hidden sizes')
_INTERMEDIATE_SIZE_LIST = flags.DEFINE_list('intermediate_size_list', ['3072'], 'Intermediate')
_NUM_EXPERTS_LIST = flags.DEFINE_list('num_experts_list', ['128'], 'Experts')
_TOP_K_LIST = flags.DEFINE_list('top_k_list', ['4'], 'Top K')
_DTYPE_LIST = flags.DEFINE_list('dtype_list', ['bfloat16'], 'Dtypes')
_SQW1_LIST = flags.DEFINE_list('sqw1_list', ['0'], 'SQW1')
_SQW2_LIST = flags.DEFINE_list('sqw2_list', ['0'], 'SQW2')
_RENORMALIZE_TOPK_LOGITS_LIST = flags.DEFINE_list('renormalize_topk_logits_list', ['False'], 'Renormalize')

# --- Tiling Search Space Flags ---
_BT_LST = flags.DEFINE_list('bt_lst', ['16', '32', '64', '128', '256'], 'bt')
_BF_LST = flags.DEFINE_list('bf_lst', ['1280', '1536'], 'bf')
_BD1_LST = flags.DEFINE_list('bd1_lst', ['1280', '1536'], 'bd1')
_BD2_LST = flags.DEFINE_list('bd2_lst', ['1280', '1536'], 'bd2')
_BTC_LST = flags.DEFINE_list('btc_lst', ['16', '32', '64', '128', '256'], 'btc')

# Logic flag for chunk generation
_SIMPLE_TUNE = flags.DEFINE_boolean('simple_tune', False, 'If True, chunks = tiles. If False, chunks = n * 256.')

# --- Spanner Config ---
SPANNER_INSTANCE = 'vllm-bm-inst'
SPANNER_DATABASE = 'tune-moe'
BATCH_SIZE = 1000  

# --- Logic & Constraints ---

def get_dtype_packing(dtype):
    bits = (dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))
    return 32 // bits

def filter_out_invalid_bd1(arr, hidden_size):
    return [x for x in arr if x % 128 == 0 and hidden_size % x == 0]

def filter_out_invalid_bd2(arr, hidden_size):
    return [x for x in arr if x % 128 == 0 and hidden_size % x == 0]

def filter_out_invalid_bf(arr, intermediate_size):
    return [x for x in arr if x % 128 == 0 and intermediate_size % x == 0]

def get_chunk_candidates(full_size):
    """Generates chunk candidates based on simple_tune flag."""
    if _SIMPLE_TUNE.value:
        return [full_size]
    # Candidates are multiples of 256 that divide the tile size evenly
    candidates = [val for val in range(256, full_size + 1, 256) if full_size % val == 0]
    return candidates if candidates else [full_size]

def is_valid_config(c, t_packing):
    """Validates tiling and chunking logic."""
    if c['btc'] > c['bt'] or c['bt'] % c['btc'] != 0: return False
    # Additional constraints for chunk alignment
    if c['bd1c'] % (t_packing * 128) != 0: return False
    if c['bd2c'] % (t_packing * 128) != 0: return False
    return True

# --- Spanner Management ---

class SpannerManager:
    def __init__(self, instance_id, database_id):
        self.current_case_id = 0
        self.invalid_count = 0
        self.buffer = []
        if not _DRY_RUN.value:
            self.client = spanner.Client(disable_builtin_metrics=True)
            self.instance = self.client.instance(instance_id)
            self.database = self.instance.database(database_id)
        else:
            self.database = None

    def init_case_set(self, case_set_id):
        if _DRY_RUN.value: return
        desc = _CASE_SET_DESC.value or "Auto-generated MoE Tune Set"
        def _do_insert(tx):
            row = tx.execute_sql("SELECT ID FROM CaseSet WHERE ID = @id", 
                                 params={'id': case_set_id}, 
                                 param_types={'id': spanner.param_types.STRING}).one_or_none()
            if row: raise exceptions.AlreadyExists(f"ID '{case_set_id}' exists.")
            tx.execute_update(
                "INSERT INTO CaseSet (ID, Description, Status) VALUES (@id, @desc, 'CREATING')",
                params={'id': case_set_id, 'desc': desc},
                param_types={'id': spanner.param_types.STRING, 'desc': spanner.param_types.STRING}
            )
        self.database.run_in_transaction(_do_insert)

    def finish_case_set(self, case_set_id, valid, invalid, duration):
        if _DRY_RUN.value: return
        def _do_update(tx):
            tx.execute_update(
                "UPDATE CaseSet SET Status = 'COMPLETED', Valid = @v, Invalid = @i, DurationSeconds = @d WHERE ID = @id",
                params={'id': case_set_id, 'v': valid, 'i': invalid, 'd': duration},
                param_types={'id': spanner.param_types.STRING, 'v': spanner.param_types.INT64, 
                             'i': spanner.param_types.INT64, 'd': spanner.param_types.FLOAT64}
            )
        self.database.run_in_transaction(_do_update)

    @retry.Retry(predicate=retry.if_transient_error)
    def flush(self):
        if not self.buffer or _DRY_RUN.value:
            self.buffer = []
            return
        with self.database.batch() as b:
            b.insert(table='Cases', columns=(
                'ID', 'CaseId', 'EP', 'NumTokens', 'HiddenSize', 'IntermediateSize', 
                'NumExpertes', 'TopK', 'DType', 'BT', 'BTC', 'BF', 'BFC', 
                'BD1', 'BD1C', 'BD2', 'BD2C', 'SQW1', 'SQW2', 'RenormalizeTopKLogits'
            ), values=self.buffer)
        self.buffer = []

    def add_row(self, row):
        self.buffer.append(row)
        self.current_case_id += 1
        if len(self.buffer) >= BATCH_SIZE: self.flush()

# --- Main ---

def main(argv):
    mgr = SpannerManager(SPANNER_INSTANCE, SPANNER_DATABASE)
    mgr.init_case_set(_CASE_SET_ID.value)
    
    # Pre-parse lists
    model_params = list(itertools.product(
        [int(x) for x in _EP_SIZES.value], [int(x) for x in _NUM_TOKENS_LIST.value],
        [int(x) for x in _HIDDEN_SIZE_LIST.value], [int(x) for x in _INTERMEDIATE_SIZE_LIST.value],
        [int(x) for x in _NUM_EXPERTS_LIST.value], [int(x) for x in _TOP_K_LIST.value],
        [jnp.dtype(s) for s in _DTYPE_LIST.value], [int(x) for x in _SQW1_LIST.value],
        [int(x) for x in _SQW2_LIST.value], [(x.lower() == 'true') for x in _RENORMALIZE_TOPK_LOGITS_LIST.value]
    ))

    bt_raw = [int(x) for x in _BT_LST.value]
    bf_raw = [int(x) for x in _BF_LST.value]
    bd1_raw = [int(x) for x in _BD1_LST.value]
    bd2_raw = [int(x) for x in _BD2_LST.value]
    btc_raw = [int(x) for x in _BTC_LST.value]

    start_time = time.time()

    for m in tqdm(model_params, desc="Generating Cases"):
        ep, tokens, h, inter, expert, tk, dt, s1, s2, rn = m
        t_packing = get_dtype_packing(dt)       
        
        # Apply top-level filtering based on model dims
        valid_bf = filter_out_invalid_bf(bf_raw, inter)
        valid_bd1 = filter_out_invalid_bd1(bd1_raw, h)
        valid_bd2 = filter_out_invalid_bd2(bd2_raw, h)

        for bt, bf, bd1, bd2, btc in itertools.product(bt_raw, valid_bf, valid_bd1, valid_bd2, btc_raw):            
            
            bfc_list = get_chunk_candidates(bf)            
            bd1c_list = get_chunk_candidates(bd1)
            bd2c_list = get_chunk_candidates(bd2)            
            for bfc, bd1c, bd2c in itertools.product(bfc_list, bd1c_list, bd2c_list):
                cfg = {'bt':bt, 'btc':btc, 'bfc':bfc, 'bd1c':bd1c, 'bd2c':bd2c}
                
                if is_valid_config(cfg, t_packing):
                    mgr.add_row((
                        _CASE_SET_ID.value, mgr.current_case_id, ep, tokens, h, inter, 
                        expert, tk, str(dt.name), bt, btc, bf, bfc, bd1, bd1c, bd2, bd2c, s1, s2, rn
                    ))
                else:
                    mgr.invalid_count += 1

    mgr.flush()
    duration = time.time() - start_time
    mgr.finish_case_set(_CASE_SET_ID.value, mgr.current_case_id, mgr.invalid_count, duration)
    print(f"\nDone. Valid: {mgr.current_case_id} | Invalid: {mgr.invalid_count} | {duration:.2f}s")

if __name__ == '__main__':
    app.run(main)