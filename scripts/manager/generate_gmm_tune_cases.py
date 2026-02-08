import itertools
import time
from absl import app, flags
from google.cloud import spanner
from google.api_core import retry, exceptions
from tqdm import tqdm
import jax.numpy as jnp

# --- Experiment Metadata Flags ---
_CASE_SET_ID = flags.DEFINE_string('case_set_id', 'gmm_v1', 'Unique ID for experiment.')
_CASE_SET_DESC = flags.DEFINE_string('case_set_desc', '', 'Manual description.')
_DRY_RUN = flags.DEFINE_boolean('dry_run', False, 'Skip Spanner.')

# --- Memory Limit Flag ---
_PER_CHIP_MEM_LIMIT_GB = flags.DEFINE_integer('mem_limit_gb', 16, 'Memory limit per TPU chip in GB.')

# --- Model Configuration Flags ---
_M_LIST = flags.DEFINE_list('m_list', ['128', '512', '2048', '8192', '16384', '32768', '65536'], 'M tokens')
_K_LIST = flags.DEFINE_list('k_list', ['6144', '2560'], 'K reduction dim')
_N_LIST = flags.DEFINE_list('n_list', ['5120', '6144'], 'N output dim')
_TOTAL_GROUPS = flags.DEFINE_list('total_groups', ['160'], 'Total experts')
_CURRENT_GROUPS = flags.DEFINE_list('current_groups', ['20', '40'], 'Experts per shard')
_LHS_DTYPE_LIST = flags.DEFINE_list('lhs_dtype_list', ['bfloat16'], 'LHS Dtypes')
_RHS_DTYPE_LIST = flags.DEFINE_list('rhs_dtype_list', ['float8_e4m3fn'], 'RHS Dtypes')

# --- Tiling Search Space Flags ---
_TM_LST = flags.DEFINE_list('tm_lst', ['128', '256', '512'], 'tm')
_TK_LST = flags.DEFINE_list('tk_lst', ['128', '256', '512', '1024', '2048'], 'tk')
_TN_LST = flags.DEFINE_list('tn_lst', ['128', '256', '512', '1024', '1280', '2048'], 'tn')

# --- Spanner Config ---
SPANNER_INSTANCE = 'vllm-bm-inst'
SPANNER_DATABASE = 'tune-gmm'
BATCH_SIZE = 1000  

def estimate_chip_memory_bytes(m, k, n, cg, ldt, rdt, q_block):
    """
    Estimates HBM usage per chip based on sharding logic in gmm_worker.py.
    """
    dtype_map = {'bfloat16': 2, 'float32': 4, 'float8_e4m3fn': 1}
    lsz = dtype_map.get(ldt, 2)
    rsz = dtype_map.get(rdt, 1)
    
    # 1. LHS: Replicated (m, k)
    lhs_mem = m * k * lsz
    # 2. RHS Experts: Sharded (cg, k, n)
    rhs_mem = cg * k * n * rsz
    # 3. RHS Scales: Sharded (cg, k//q_block, 1, n)
    num_blocks = k // q_block
    scale_mem = cg * num_blocks * n * 4  # Scales are float32
    # 4. Output Buffer: Replicated (m, n) 
    out_mem = m * n * 2 # bfloat16
    
    return lhs_mem + rhs_mem + scale_mem + out_mem

def is_valid_gmm_config(m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn, mem_limit_gb):
    """Refined GMM kernel constraints and memory check."""
    # 1. M-dimension strictly divisible by TM
    if m % tm != 0: return False
    
    # 2. K-tile alignment with quantization block
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

    def init_case_set(self, case_set_id, scan_space):
        if _DRY_RUN.value: return
        def _do_insert(tx):
            tx.execute_update(
                "INSERT INTO CaseSet (ID, Description, Status, ScanSpace) VALUES (@id, @desc, 'CREATING', @scan)",
                params={'id': case_set_id, 'desc': _CASE_SET_DESC.value or "GMM Tune", 'scan': scan_space})
        self.database.run_in_transaction(_do_insert)

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
    
    # Define problem space
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
    
    start_time = time.time()
    for p in tqdm(problem_space, desc="Generating Valid GMM Cases"):
        m, k, n, tg, cg, ldt, rdt = p
        q_block = k # Per-channel quantization scale block size
        
        for tm, tk, tn in tiling_space:
            if is_valid_gmm_config(m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn, _PER_CHIP_MEM_LIMIT_GB.value):
                mgr.add_row((_CASE_SET_ID.value, mgr.current_case_id, m, k, n, tg, cg, ldt, rdt, q_block, tm, tk, tn))
            else:
                mgr.invalid_count += 1

    mgr.flush()
    print(f"Done. Valid: {mgr.current_case_id} | Invalid: {mgr.invalid_count}")

if __name__ == '__main__':
    app.run(main)

