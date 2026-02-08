import itertools
import time
from absl import app, flags
from google.cloud import spanner
from google.api_core import retry, exceptions
from tqdm import tqdm
import jax.numpy as jnp

# --- Experiment Metadata Flags ---
_CASE_SET_ID = flags.DEFINE_string('case_set_id', 'gmm_qwen3_v1', 'Unique ID for this experiment run.')
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '', 'Manual description.')
_DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'Skip Spanner operations.')

# --- Memory Limit Flag ---
_PER_CHIP_MEM_LIMIT_GB = flags.DEFINE_integer('mem_limit_gb', 16, 'Memory limit per TPU chip in GB.')

# --- Model Configuration Flags (Extracted from logs) ---
_M_LIST = flags.DEFINE_list('m_list', ['128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768', '65536'], 'M tokens')
_K_LIST = flags.DEFINE_list('k_list', ['6144', '2560'], 'K reduction dim')
_N_LIST = flags.DEFINE_list('n_list', ['5120', '6144'], 'N output dim')
_TOTAL_GROUPS = flags.DEFINE_list('total_groups', ['160'], 'Total experts')
_CURRENT_GROUPS = flags.DEFINE_list('current_groups', ['20'], 'Experts per shard')
_LHS_DTYPE_LIST = flags.DEFINE_list('lhs_dtype_list', ['bfloat16'], 'LHS Dtypes')
_RHS_DTYPE_LIST = flags.DEFINE_list('rhs_dtype_list', ['float8_e4m3fn'], 'RHS Dtypes')

# --- Tiling Search Space Flags ---
_TM_LST = flags.DEFINE_list('tm_lst', ['128', '256', '512', '1024'], 'tm')
_TK_LST = flags.DEFINE_list('tk_lst', ['128', '256', '512', '1024', '2048'], 'tk')
_TN_LST = flags.DEFINE_list('tn_lst', ['128', '256', '512', '1024', '1280', '2048'], 'tn')

# --- Spanner Config ---
SPANNER_INSTANCE = 'vllm-bm-inst'
SPANNER_DATABASE = 'tune-gmm'
BATCH_SIZE = 1000  

def estimate_chip_memory_bytes(m, k, n, cg, ldt, rdt, q_block):
    """Estimates HBM usage per chip to avoid OOM during data generation."""
    dtype_map = {'bfloat16': 2, 'float32': 4, 'float8_e4m3fn': 1}
    lsz = dtype_map.get(ldt, 2)
    rsz = dtype_map.get(rdt, 1)
    
    lhs_mem = m * k * lsz
    rhs_mem = cg * k * n * rsz
    num_blocks = k // q_block
    scale_mem = cg * num_blocks * n * 4  # Scales are float32
    out_mem = m * n * 2 # bfloat16 output
    
    return lhs_mem + rhs_mem + scale_mem + out_mem

def is_valid_gmm_config(m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn, mem_limit_gb):
    """Validates configuration based on gmm.py constraints."""
    # 1. M must be divisible by TM
    if m % tm != 0: return False
    
    # 2. TK must be aligned with quantization block size
    if tk % q_block != 0 and q_block % tk != 0: return False
    
    # 3. Memory limit check
    mem_bytes = estimate_chip_memory_bytes(m, k, n, cg, ldt, rdt, q_block)
    if mem_bytes > (mem_limit_gb * 1024**3): return False
    
    return True

class SpannerManager:
    def __init__(self, instance_id, database_id):
        self.current_case_id = 0
        self.invalid_count = 0
        self.buffer = []
        if not _DRY_RUN.value:
            self.client = spanner.Client()
            self.instance = self.client.instance(instance_id)
            self.database = self.instance.database(database_id)
        else:
            self.database = None

    def init_case_set(self, case_set_id, scan_space):
        """Initializes the CaseSet row."""
        if _DRY_RUN.value: return
        desc = _CASE_SET_DESC.value or "GMM Kernel Tune Set"
        def _do_insert(tx):
            tx.execute_update(
                "INSERT INTO CaseSet (ID, Description, Status, ScanSpace) VALUES (@id, @desc, 'CREATING', @scan)",
                params={'id': case_set_id, 'desc': desc, 'scan': scan_space},
                param_types={'id': spanner.param_types.STRING, 'desc': spanner.param_types.STRING, 'scan': spanner.param_types.INT64}
            )
        self.database.run_in_transaction(_do_insert)

    def finish_case_set(self, case_set_id, valid, invalid, duration):
        """Updates tracking columns upon completion."""
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
        if not self.buffer or _DRY_RUN.value: return
        with self.database.batch() as b:
            b.insert(table='Cases', columns=(
                'ID', 'CaseId', 'M', 'K', 'N', 'NumTotalGroups', 'NumCurrentGroups',
                'LhsDType', 'RhsDType', 'QuantBlockSize', 'TM', 'TK', 'TN'
            ), values=self.buffer)
        self.buffer = []

    def add_row(self, row):
        self.buffer.append(row)
        self.current_case_id += 1
        if len(self.buffer) >= BATCH_SIZE: self.flush()

def main(argv):
    mgr = SpannerManager(SPANNER_INSTANCE, SPANNER_DATABASE)
    
    # 1. Define Search Space
    problem_space = list(itertools.product(
        [int(x) for x in _M_LIST.value], [int(x) for x in _K_LIST.value],
        [int(x) for x in _N_LIST.value], [int(x) for x in _TOTAL_GROUPS.value],
        [int(x) for x in _CURRENT_GROUPS.value], _LHS_DTYPE_LIST.value, _RHS_DTYPE_LIST.value
    ))
    tiling_space = list(itertools.product(
        [int(x) for x in _TM_LST.value], [int(x) for x in _TK_LST.value], [int(x) for x in _TN_LST.value]
    ))
    
    total_combinations = len(problem_space) * len(tiling_space)
    mgr.init_case_set(_CASE_SET_ID.value, total_combinations)
    
    # 2. Iterate and Validate
    start_time = time.time()
    for p in tqdm(problem_space, desc="Generating GMM Cases"):
        m, k, n, tg, cg, ldt, rdt = p
        q_block = k # Per-channel quantization block size
        
        for tm, tk, tn in tiling_space:
            if is_valid_gmm_config(m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn, _PER_CHIP_MEM_LIMIT_GB.value):
                mgr.add_row((_CASE_SET_ID.value, mgr.current_case_id, m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn))
            else:
                mgr.invalid_count += 1

    # 3. Finalize
    mgr.flush()
    duration = time.time() - start_time
    mgr.finish_case_set(_CASE_SET_ID.value, mgr.current_case_id, mgr.invalid_count, duration)
    
    print(f"\nDone. Valid: {mgr.current_case_id} | Invalid: {mgr.invalid_count} | Duration: {duration:.2f}s")

if __name__ == '__main__':
    app.run(main)