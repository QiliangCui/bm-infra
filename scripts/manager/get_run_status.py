import sys
from absl import app, flags, logging
from google.cloud import spanner

# --- Flags Definition ---
FLAGS = flags.FLAGS

flags.DEFINE_string('project_id', "cloud-tpu-inference-test", 'GCP Project ID')
flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID')
flags.DEFINE_string('database_id', None, 'Spanner Database ID (Overrides --tune choice)')

# New Flag for Tune Type
flags.DEFINE_enum('tune', 'moe', ['moe', 'gmm'], 'The tuning target: "moe" or "gmm".')

flags.DEFINE_string('id', None, 'The CaseSet ID (Parent table ID).', required=True)
flags.DEFINE_string('run_id', None, 'The specific RunId to filter results.', required=True)
flags.DEFINE_string('status', None, 'Filter by ProcessedStatus (e.g., SUCCESS, FAILED_OOM)')
flags.DEFINE_integer('limit', None, 'Maximum number of rows to return.')

def get_run_status(_):
    """Queries CaseResults joined with Cases for detailed architectural info."""
    
    # Determine Database ID based on logic
    db_id = FLAGS.database_id
    if db_id is None:
        db_id = 'tune-gmm' if FLAGS.tune == 'gmm' else 'tune-moe'

    client = spanner.Client(project=FLAGS.project_id)
    instance = client.instance(FLAGS.instance_id)
    database = instance.database(db_id)

    # Base query components common to both
    query_cols = ["r.CaseId, r.ProcessedStatus, r.Latency, r.WarmupTime, r.WorkerID"]
    
    if FLAGS.tune == 'moe':
        # MoE Specific Columns
        query_cols += [
            "c.HiddenSize, c.IntermediateSize, c.NumExpertes, c.TopK, c.SQW1, c.SQW2, c.NumTokens, c.EP",
            "c.BT, c.BTC, c.BF, c.BFC, c.BD1, c.BD1C, c.BD2, c.BD2C"
        ]
        header = (f"{'CaseId':<7} | {'Status':<10} | {'Latency':<10} | {'Warmup':<10} | "
                  f"{'Key (H, I, E, K, SQ1, SQ2, Tok, EP)':<40} | "
                  f"{'Tiling (BT, BTC, BF, BFC, BD1, BD1C, BD2, BD2C)':<45}")
    else:
        # GMM Specific Columns
        query_cols += [
            "c.M, c.K, c.N, c.NumTotalGroups, c.NumCurrentGroups, c.LhsDType, c.RhsDType, c.QuantBlockSize",
            "c.TM, c.TK, c.TN"
        ]
        header = (f"{'CaseId':<7} | {'Status':<10} | {'Latency':<10} | {'Warmup':<10} | "
                  f"{'Key (M, K, N, TG, CG, LDT, RDT, QB)':<50} | "
                  f"{'Tiling (TM, TK, TN)':<25}")

    query_parts = [
        f"SELECT {', '.join(query_cols)}",
        "FROM CaseResults r",
        "JOIN Cases c ON r.ID = c.ID AND r.CaseId = c.CaseId",
        "WHERE r.ID = @id AND r.RunId = @rid"
    ]
    
    params = {'id': FLAGS.id, 'rid': FLAGS.run_id}
    param_types = {'id': spanner.param_types.STRING, 'rid': spanner.param_types.STRING}

    if FLAGS.status:
        query_parts.append("AND r.ProcessedStatus = @status")
        params['status'] = FLAGS.status
        param_types['status'] = spanner.param_types.STRING

    query_parts.append("ORDER BY r.CaseId ASC")
    if FLAGS.limit:
        query_parts.append("LIMIT @limit")
        params['limit'] = FLAGS.limit
        param_types['limit'] = spanner.param_types.INT64

    final_query = " ".join(query_parts)

    try:
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(final_query, params=params, param_types=param_types)

            print("\n" + "=" * 175)
            print(header)
            print("-" * 175)

            row_count = 0
            for row in results:
                row_count += 1
                # Shared prefix
                case_id, status, lat, warm, worker = row[0:5]
                lat_str = f"{lat:,}" if lat != sys.maxsize else "OOM"
                warm_str = f"{warm:,}" if warm else "0"

                if FLAGS.tune == 'moe':
                    (h, i, e, k, sq1, sq2, tok, ep, bt, btc, bf, bfc, bd1, bd1c, bd2, bd2c) = row[5:]
                    key_tuple = f"({h}, {i}, {e}, {k}, {sq1}, {sq2}, {tok}, {ep})"
                    tiling_tuple = f"({bt}, {btc if btc != -1 else bt}, {bf}, {bfc if bfc != -1 else bf}, {bd1}, {bd1c}, {bd2}, {bd2c})"
                    print(f"{case_id:<7} | {status:<10} | {lat_str:<10} | {warm_str:<10} | "
                          f"{key_tuple:<40} | {tiling_tuple:<45}")
                else:
                    (m, k, n, tg, cg, ldt, rdt, qb, tm, tk, tn) = row[5:]
                    key_tuple = f"({m}, {k}, {n}, {tg}, {cg}, {ldt}, {rdt}, {qb})"
                    tiling_tuple = f"({tm}, {tk}, {tn})"
                    print(f"{case_id:<7} | {status:<10} | {lat_str:<10} | {warm_str:<10} | "
                          f"{key_tuple:<50} | {tiling_tuple:<25}")

            print("=" * 175)
            logging.info(f"Displayed {row_count} rows from {db_id}.")

    except Exception as e:
        logging.error(f"Failed to query Spanner: {e}")
        sys.exit(1)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(get_run_status)