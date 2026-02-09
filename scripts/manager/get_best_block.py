import sys
import jax.numpy as jnp
from jax._src import dtypes
from absl import app, flags, logging
from google.cloud import spanner

# --- Flags Definition ---
FLAGS = flags.FLAGS

_CASE_SET_ID = flags.DEFINE_string('case_set_id', None, 'Unique ID for the experiment run.')
_RUN_ID = flags.DEFINE_string('run_id', None, 'Specific Run ID.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'The file to update/create.')
_INSTANCE_ID = flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID.')
_DATABASE_ID = flags.DEFINE_string('database_id', None, 'Spanner Database ID.')

# Defined using enum style as requested
_TUNE = flags.DEFINE_enum('tune', 'moe', ['moe', 'gmm'], 'The tuning target: "moe" or "gmm".')

flags.mark_flag_as_required('case_set_id')
flags.mark_flag_as_required('run_id')

def get_w_packing(dtype_name):
    dtype = jnp.dtype(dtype_name)
    bits = (dtypes.bit_width(dtype) if hasattr(dtypes, "bit_width") else dtypes.itemsize_bits(dtype))
    return 32 // bits

def get_old_latency(database, model_key, old_tiling):
    """Queries Spanner for historical latency of the exact tiling in your current dict."""
    if FLAGS.tune == 'moe':
        h, inter, exp, topk, tp, wp, tokens, ep = model_key
        bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c = old_tiling
        query = """
            SELECT cr.Latency 
            FROM CaseResults cr
            JOIN Cases c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
            WHERE c.HiddenSize = @h AND c.IntermediateSize = @inter AND c.NumTokens = @tokens 
              AND c.EP = @ep AND c.BT = @bt AND c.BF = @bf AND c.BD1 = @bd1 AND c.BD2 = @bd2
              AND cr.ProcessedStatus = 'SUCCESS'
            ORDER BY cr.ProcessedAt DESC LIMIT 1
        """
        params = {'h': h, 'inter': inter, 'tokens': tokens, 'ep': ep, 'bt': bt, 'bf': bf, 'bd1': bd1, 'bd2': bd2}
        p_types = {k: spanner.param_types.INT64 for k in params.keys()}
    else:
        # GMM Logic
        m, k, n, tg, cg, ldt, rdt, qb = model_key
        tm, tk, tn = old_tiling
        query = """
            SELECT cr.Latency 
            FROM CaseResults cr
            JOIN Cases c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
            WHERE c.M = @m AND c.K = @k AND c.N = @n AND c.NumTotalGroups = @tg 
              AND c.NumCurrentGroups = @cg AND c.LhsDType = @ldt AND c.RhsDType = @rdt 
              AND c.QuantBlockSize = @qb AND c.TM = @tm AND c.TK = @tk AND c.TN = @tn
              AND cr.ProcessedStatus = 'SUCCESS'
            ORDER BY cr.ProcessedAt DESC LIMIT 1
        """
        params = {'m': m, 'k': k, 'n': n, 'tg': tg, 'cg': cg, 'ldt': ldt, 'rdt': rdt, 'qb': qb, 'tm': tm, 'tk': tk, 'tn': tn}
        p_types = {k: spanner.param_types.INT64 for k in params.keys() if k not in ['ldt', 'rdt']}
        p_types.update({'ldt': spanner.param_types.STRING, 'rdt': spanner.param_types.STRING})

    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(query, params=params, param_types=p_types)
        row = results.one_or_none()
        return row[0] if row else ""

def fetch_best_configs(database):
    if FLAGS.tune == 'moe':
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
    else:
        # GMM Logic
        query = """
        WITH BestLatencies AS (
          SELECT 
            c.M, c.K, c.N, c.NumTotalGroups, c.NumCurrentGroups, 
            c.LhsDType, c.RhsDType, c.QuantBlockSize,
            MIN(cr.Latency) as MinLat
          FROM CaseResults AS cr
          JOIN Cases AS c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
          WHERE cr.ID = @csid AND cr.RunId = @rid AND cr.ProcessedStatus = 'SUCCESS'
          GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
        ),
        BestCases AS (
          SELECT bl.*, MIN(c.CaseId) as BestCaseId
          FROM CaseResults AS cr
          JOIN Cases AS c ON cr.ID = c.ID AND cr.CaseId = c.CaseId
          JOIN BestLatencies bl ON 
            c.M = bl.M AND c.K = bl.K AND c.N = bl.N AND
            c.NumTotalGroups = bl.NumTotalGroups AND c.NumCurrentGroups = bl.NumCurrentGroups AND
            c.LhsDType = bl.LhsDType AND c.RhsDType = bl.RhsDType AND
            c.QuantBlockSize = bl.QuantBlockSize AND
            cr.Latency = bl.MinLat
          WHERE cr.ID = @csid AND cr.RunId = @rid
          GROUP BY 1, 2, 3, 4, 5, 6, 7, 8, 9
        )
        SELECT 
            bc.M, bc.K, bc.N, bc.NumTotalGroups, bc.NumCurrentGroups, bc.LhsDType, 
            bc.RhsDType, bc.QuantBlockSize, c.TM, c.TK, c.TN,
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
    # Determine Database and Output path based on tune flag
    db_id = _DATABASE_ID.value or ('tune-gmm' if FLAGS.tune == 'gmm' else 'tune-moe')
    out_path = _OUTPUT_PATH.value or (f'tuned_{FLAGS.tune}_block_sizes.py')

    # 1. Load existing data
    local_dict = {}
    try:
        if FLAGS.tune == 'moe':
            from tuned_moe_block_sizes import TUNED_BLOCK_SIZES
        else:
            from tuned_gmm_block_sizes import TUNED_BLOCK_SIZES
        local_dict = TUNED_BLOCK_SIZES.copy()
        print(f"Loaded {len(local_dict)} existing entries from {out_path}")
    except ImportError:
        print(f"Starting with a fresh dictionary for {FLAGS.tune}.")
        raise

    client = spanner.Client(disable_builtin_metrics=True)
    db = client.instance(_INSTANCE_ID.value).database(db_id)
    
    # 2. Fetch new benchmark winners
    rows = fetch_best_configs(db)
    if not rows:
        print("No benchmark results found. Exiting.")
        return

    print(f"\n--- {FLAGS.tune.upper()} Merged Benchmark Report ---")
    
    # FIX: Pre-calculate tiling width variable to prevent f-string format errors
    tiling_width = 45 if FLAGS.tune == 'moe' else 30

    if FLAGS.tune == 'moe':
        header = (f"{'Action':<8} | "
                  f"{'Key (Hidden, Inter, Exp, TopK, TPack, WPack, Toks, EP)':<60} | "
                  f"{'New Tiling':<{tiling_width}} | "
                  f"{'Old Tiling':<{tiling_width}} | "
                  f"{'Lat':<8} | "
                  f"{'OldLat':<8} | "
                  f"{'Improv':<8}")
    else:
        header = (f"{'Action':<8} | "
                  f"{'Key (M, K, N, TG, CG, LDT, RDT, QB)':<60} | "
                  f"{'New Tiling (TM, TK, TN)':<{tiling_width}} | "
                  f"{'Old Tiling':<{tiling_width}} | "
                  f"{'Lat':<8} | "
                  f"{'OldLat':<8} | "
                  f"{'Improv':<8}")
    print(header)
    print("-" * len(header))

    # 3. Merge logic
    for r in rows:
        if FLAGS.tune == 'moe':
            h, inter, exp, topk, dtype, tokens, ep, bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c, cid, lat = r
            t_p, w_p = 2, get_w_packing(dtype)
            model_key = (h, inter, exp, topk, t_p, w_p, tokens, ep)
            new_val = (bt, bf, bd1, bd2, btc, bfc, bd1c, bd2c)
        else:
            m, k, n, tg, cg, ldt, rdt, qb, tm, tk, tn, cid, lat = r
            model_key = (m, k, n, tg, cg, ldt, rdt, qb)
            new_val = (tm, tk, tn)

        old_lat_str = ""
        old_tiling_str = "None"
        improv_str = ""
        
        if model_key in local_dict:
            old_val = local_dict[model_key]
            if old_val != new_val:
                action = "update"
                old_lat = get_old_latency(db, model_key, old_val)
                if isinstance(old_lat, (int, float)) and old_lat > 0:
                    improvement = ((old_lat - lat) / old_lat) * 100
                    improv_str = f"{improvement:.2f}%"
                    old_lat_str = f"{old_lat:,}"
                else:
                    old_lat_str = str(old_lat) if old_lat else ""
                old_tiling_str = str(old_val)
            else:
                action = "keep"
        else:
            action = "add"

        if action != "keep":
            # Using pre-calculated tiling_width for clean column alignment
            print(f"{action:<8} | {str(model_key):<60} | {str(new_val):<{tiling_width}} | "
                  f"{old_tiling_str:<{tiling_width}} | {lat:<8,} | {old_lat_str:<8} | {improv_str:<8}")
        local_dict[model_key] = new_val

    # 4. Write full merged dictionary to file
    with open(out_path, 'w') as f:
        f.write(f"# Auto-generated Tuned {FLAGS.tune.upper()} Block Sizes\n")
        f.write("TUNED_BLOCK_SIZES = {\n")
        for k in sorted(local_dict.keys()):
            v = local_dict[k]
            f.write(f"    {k}: (\n")
            for item in v:
                if isinstance(item, int) and item > 0 and item % 256 == 0:
                    f.write(f"        256 * {item // 256},\n")
                else:
                    f.write(f"        {repr(item)},\n")
            f.write("    ),\n")
        f.write("}\n")

    print(f"\nMerge complete. Updated file: {out_path} (Total entries: {len(local_dict)})")

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)