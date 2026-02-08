import sys
from absl import app, flags, logging
from google.cloud import spanner

# --- Flags Definition ---
FLAGS = flags.FLAGS

flags.DEFINE_string('project_id', "cloud-tpu-inference-test", 'GCP Project ID')
flags.DEFINE_string('instance_id', 'vllm-bm-inst', 'Spanner Instance ID')
# Default database_id is now handled dynamically based on the --tune flag
flags.DEFINE_string('database_id', None, 'Spanner Database ID (Overrides --tune choice)')

# New Flag for Tune Type
flags.DEFINE_enum('tune', 'moe', ['moe', 'gmm'], 'The tuning target: "moe" or "gmm".')

flags.DEFINE_string('id', None, 'The CaseSet ID (Parent table ID).', required=True)
flags.DEFINE_string('run_id', None, 'The specific RunId to aggregate.', required=True)

def get_run_statistic(_):
    """Groups and counts ProcessedStatus with timing statistics for MoE or GMM."""
    
    # Determine Database ID based on logic
    db_id = FLAGS.database_id
    if db_id is None:
        db_id = 'tune-gmm' if FLAGS.tune == 'gmm' else 'tune-moe'

    client = spanner.Client(project=FLAGS.project_id)
    instance = client.instance(FLAGS.instance_id)
    database = instance.database(db_id)

    # Both MoE and GMM use the same tracking table structure for results
    total_cases_query = "SELECT COUNT(*) FROM Cases WHERE ID = @id"
    
    status_query = (
        "SELECT ProcessedStatus, COUNT(*) as StatusCount, SUM(TotalTime) as SumTime "
        "FROM CaseResults "
        "WHERE ID = @id AND RunId = @rid "
        "GROUP BY ProcessedStatus "
        "ORDER BY StatusCount DESC"
    )
    
    params = {'id': FLAGS.id, 'rid': FLAGS.run_id}
    param_types = {'id': spanner.param_types.STRING, 'rid': spanner.param_types.STRING}

    try:
        with database.snapshot(multi_use=True) as snapshot:
            # 1. Get Total Expected from the Cases table
            total_cases_result = snapshot.execute_sql(
                total_cases_query, 
                params={'id': FLAGS.id}, 
                param_types={'id': spanner.param_types.STRING}
            )
            total_expected = list(total_cases_result)[0][0]

            # 2. Get Status Statistics from CaseResults
            results = list(snapshot.execute_sql(status_query, params=params, param_types=param_types))

            print(f"\n{FLAGS.tune.upper()} Run Statistics Summary")
            print(f"Database: {db_id}")
            print(f"ID:       {FLAGS.id}")
            print(f"RunId:    {FLAGS.run_id}")
            
            header = f"{'ProcessedStatus':<20} | {'Count':<10} | {'% of Tot':<10} | {'Total (Hr)':<12} | {'Avg (Sec)':<10}"
            print("=" * len(header))
            print(header)
            print("-" * len(header))

            total_processed = 0
            accumulated_time_us = 0
            
            for row in results:
                status, count, sum_time_us = row
                status_str = status if status else "UNKNOWN"
                sum_time_us = sum_time_us if sum_time_us else 0
                
                pct = (count / total_expected * 100) if total_expected > 0 else 0
                total_hr = sum_time_us / 1_000_000 / 3_600
                avg_sec = (sum_time_us / count / 1_000_000) if count > 0 else 0
                
                print(f"{status_str:<20} | {count:<10,} | {pct:>8.2f}% | {total_hr:>12.2f} | {avg_sec:>10.2f}")
                
                total_processed += count
                accumulated_time_us += sum_time_us

            pending = max(0, total_expected - total_processed)
            processed_pct = (total_processed / total_expected * 100) if total_expected > 0 else 0
            pending_pct = (pending / total_expected * 100) if total_expected > 0 else 0
            
            total_processed_hr = accumulated_time_us / 1_000_000 / 3_600
            overall_avg_sec = (accumulated_time_us / total_processed / 1_000_000) if total_processed > 0 else 0

            print("-" * len(header))
            print(f"{'TOTAL PROCESSED':<20} | {total_processed:<10,} | {processed_pct:>8.2f}% | {total_processed_hr:>12.2f} | {overall_avg_sec:>10.2f}")
            print(f"{'PENDING/NOT RUN':<20} | {pending:<10,} | {pending_pct:>8.2f}% | {'-':>12} | {'-':>10}")
            print("-" * len(header))
            print(f"{'TOTAL CASES IN SET':<20} | {total_expected:<10,} | 100.00% | {'-':>12} | {'-':>10}")
            print("=" * len(header))

    except Exception as e:
        logging.error(f"Failed to query Spanner: {e}")
        sys.exit(1)

if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(get_run_statistic)