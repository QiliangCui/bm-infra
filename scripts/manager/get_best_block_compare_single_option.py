import sys
import jax.numpy as jnp
from jax._src import dtypes
from absl import app, flags
from google.cloud import spanner

# --- Flags Definition ---
_CASE_SET_ID = flags.DEFINE_string('case_set_id', None, 'Unique ID for the experiment run.')
_RUN_ID = flags.DEFINE_string('run_id', None, 'Specific Run ID.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', 'tuned_block_sizes.py', 'The file to update/create.')
_INSTANCE_ID = flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID.')
_DATABASE_ID = flags.DEFINE_string('database_id', 'tune-moe', 'Spanner Database ID.')

flags.mark_flag_as_required('case_set_id')
flags.mark_flag_as_required('run_id')

# --- Fixed Baseline Configuration ---
FIXED_BLOCK = (64, 1536, 1536, 1536, 64, 1536, 1536, 1536)

def get_w_packing(dtype_name):
    dtype = jnp.dtype(dtype_name)
    bits = (dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))
    return 32 // bits

def get_baseline_latency(database, model_key):
    """Queries Spanner for latency of the FIXED_BLOCK baseline."""
    h, inter, exp, topk, tp, wp, tokens, ep = model_key
    # Destructure the fixed block for the query
    bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = FIXED_BLOCK
    
    query = """
        SELECT cr.Latency 
        FROM CaseResults cr
        JOIN Cases c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
        WHERE c.HiddenSize = @h AND c.IntermediateSize = @inter AND c.NumTokens = @tokens 
          AND c.EP = @ep AND c.BT = @bt AND c.BF = @bf AND c.BD1 = @bd1 AND c.BD2 = @bd2
          AND cr.ProcessedStatus = 'SUCCESS'
        ORDER BY cr.ProcessedAt DESC LIMIT 1
    """
    params = {'h': h, 'inter': inter, 'tokens': tokens, 'ep': ep, 
              'bt': bt, 'bf': bf, 'bd1': bd1, 'bd2': bd2}
    
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(query, params=params, 
                                       param_types={k: spanner.param_types.INT64 for k in params.keys()})
        row = results.one_or_none()
        return row[0] if row else "N/A"

def fetch_best_configs(database):
    query = """
    WITH BestLatencies AS (
      SELECT 
        c.HiddenSize, c.IntermediateSize, c.NumExpertes, c.TopK, 
        c.DType, c.NumTokens, c.EP, c.SQW1, c.SQW2, c.RenormalizeTopKLogits,
        MIN(cr.Latency) as MinLat
      FROM CaseResults AS cr
      JOIN Cases AS c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
      WHERE cr.ID = @csid AND cr.RunId = @rid AND cr.ProcessedStatus = 'SUCCESS'
      GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    ),
    BestCases AS (
      SELECT bl.*, MIN(c.CaseId) as BestCaseId
      FROM CaseResults AS cr
      JOIN Cases AS c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
      JOIN BestLatencies bl ON 
        c.HiddenSize = bl.HiddenSize AND c.IntermediateSize = bl.IntermediateSize AND
        c.NumExpertes = bl.NumExpertes AND c.TopK = bl.TopK AND
        c.DType = bl.DType AND c.NumTokens = bl.NumTokens AND
        c.EP = bl.EP AND c.SQW1 = bl.SQW1 AND c.SQW2 = bl.SQW2 AND
        c.RenormalizeTopKLogits = bl.RenormalizeTopKLogits AND
        cr.Latency = bl.MinLat
      WHERE cr.ID = @csid AND cr.RunId = @rid
      GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    )
    SELECT 
        bc.HiddenSize, bc.IntermediateSize, bc.NumExpertes, bc.TopK, bc.DType, 
        bc.NumTokens, bc.EP, c.BT, c.BF, c.BD1, c.BD2, c.BTC, c.BFC, c.BD1C, c.BD2C,
        bc.BestCaseId, bc.MinLat
    FROM BestCases bc
    JOIN Cases c ON c.ID = @csid AND c.CaseId = bc.BestCaseId
    """
    results = []
    with database.snapshot() as snapshot:
        rows = snapshot.execute_sql(query, params={'csid': _CASE_SET_ID.value, 'rid': _RUN_ID.value},
                                    param_types={'csid': spanner.param_types.STRING, 'rid': spanner.param_types.STRING})
        for row in rows:
            results.append(row)
    return results

def main(argv):
    # Load existing dictionary
    local_dict = {}
    try:
        from scripts.manager.tuned_moe_block_sizes import TUNED_BLOCK_SIZES
        local_dict = TUNED_BLOCK_SIZES.copy()
    except ImportError:
        pass

    client = spanner.Client()
    db = client.instance(_INSTANCE_ID.value).database(_DATABASE_ID.value)
    
    rows = fetch_best_configs(db)
    if not rows:
        print("No benchmark results found. Exiting.")
        return

    print(f"\n--- Benchmark Winner vs. Fixed Baseline ---")
    # Added 'Baseline Tiling' column as requested
    header = (f"{'Key (H, I, E, K, TP, WP, Tok, EP)':<50} | "
              f"{'Best Tiling Found':<45} | "
              f"{'Baseline Tiling (Fixed)':<45} | "
              f"{'Best Lat':<10} | "
              f"{'Base Lat':<10} | "
              f"{'Improv (%)':<10}")
    print(header)
    print("-" * len(header))

    for r in rows:
        h, inter, exp, topk, dtype, tokens, ep, bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c, cid, lat = r
        t_p, w_p = 2, get_w_packing(dtype)
        
        model_key = (h, inter, exp, topk, t_p, w_p, tokens, ep)
        new_val = (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)

        # 1. Compare best with fixed block latency
        baseline_lat = get_baseline_latency(db, model_key)
        
        improv_str = "N/A"
        if isinstance(baseline_lat, int) and baseline_lat > 0:
            improvement = ((baseline_lat - lat) / baseline_lat) * 100
            improv_str = f"{improvement:+.2f}%"
        
        base_lat_str = f"{baseline_lat:,}" if isinstance(baseline_lat, int) else "N/A"

        # Explicitly printing FIXED_BLOCK as the "Old Tiling"
        print(f"{str(model_key):<50} | {str(new_val):<45} | "
              f"{str(FIXED_BLOCK):<45} | "
              f"{lat:<10,} | {base_lat_str:<10} | {improv_str:<10}")

        local_dict[model_key] = new_val

    # 2. Write merged dictionary to file (remains the same)
    with open(_OUTPUT_PATH.value, 'w') as f:
        f.write("# Auto-generated Tuned Block Sizes\n")
        f.write("TUNED_BLOCK_SIZES = {\n")
        for k in sorted(local_dict.keys()):
            v = local_dict[k]
            f.write(f"    {k}: (\n")
            for item in v:
                f.write(f"        256 * {item // 256},\n" if item > 0 and item % 256 == 0 else f"        {item},\n")
            f.write("    ),\n")
        f.write("}\n")

    print(f"\nUpdate complete. Total entries in { _OUTPUT_PATH.value}: {len(local_dict)}")
if __name__ == '__main__':
    app.run(main)