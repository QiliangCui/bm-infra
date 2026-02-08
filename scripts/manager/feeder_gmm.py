import math
import sys
from absl import app, flags
from google.cloud import spanner, pubsub_v1

# --- Flags ---
_CASE_SET_ID = flags.DEFINE_string('case_set_id', 'gmm_qwen3_v1', 'Target CaseSet ID for GMM tuning.')
_RUN_ID = flags.DEFINE_string('run_id', 'run_001', 'Unique Run ID for this benchmark pass.')
_BUCKET_SIZE = flags.DEFINE_integer('bucket_size', 100, 'Number of CaseIds per Pub/Sub message.')

# --- Config ---
PROJECT_ID = "cloud-tpu-inference-test"
SPANNER_INSTANCE = 'vllm-bm-inst'
SPANNER_DATABASE = 'tune-gmm'
TOPIC_ID = "vllm-tune-queue-tpu7x-2"

def get_total_valid_cases(database, case_set_id):
    """Fetches the Valid case count from the CaseSet table."""
    with database.snapshot() as snapshot:
        results = snapshot.execute_sql(
            "SELECT Valid FROM CaseSet WHERE ID = @id",
            params={'id': case_set_id},
            param_types={'id': spanner.param_types.STRING}
        )
        row = results.one_or_none()
        if not row:
            raise ValueError(f"CaseSet ID '{case_set_id}' not found in Spanner.")
        return row[0]

def create_buckets_in_spanner(database, case_set_id, run_id, total_cases, bucket_size):
    """Generates bucket metadata and inserts into WorkBuckets table."""
    buckets = []
    for i in range(0, total_cases, bucket_size):
        bucket_id = i // bucket_size
        start = i
        end = min(i + bucket_size - 1, total_cases - 1)
        # Format: (ID, RunId, BucketId, StartCaseId, EndCaseId, Status, WorkerID, TotalTime, UpdatedAt)
        # Matches the GMM SDL: ID, RunId, BucketId, StartCaseId, EndCaseId, Status, WorkerID, TotalTime, UpdatedAt
        buckets.append((case_set_id, run_id, bucket_id, start, end, "PENDING", None, None, spanner.COMMIT_TIMESTAMP))

    def write_batches(transaction, bucket_slice):
        transaction.insert(
            table='WorkBuckets',
            columns=('ID', 'RunId', 'BucketId', 'StartCaseId', 'EndCaseId', 'Status', 'WorkerID', 'TotalTime', 'UpdatedAt'),
            values=bucket_slice
        )

    print(f"Initializing {len(buckets):,} buckets in Spanner...")
    sub_batch_size = 5000 
    for i in range(0, len(buckets), sub_batch_size):
        database.run_in_transaction(write_batches, buckets[i:i + sub_batch_size])
        print(f"  Progress: {min(i + sub_batch_size, len(buckets)):,} / {len(buckets):,}")

def publish_to_pubsub(case_set_id, run_id, total_cases, bucket_size):
    """Publishes bucket tasks to Pub/Sub topic."""
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    
    total_buckets = math.ceil(total_cases / bucket_size)
    print(f"Publishing {total_buckets:,} messages to Pub/Sub...")

    for i in range(0, total_cases, bucket_size):
        bucket_id = i // bucket_size
        start = i
        end = min(i + bucket_size - 1, total_cases - 1)
        
        # Message format: ID|RunId|BucketId|Start|End
        message_str = f"{case_set_id}|{run_id}|{bucket_id}|{start}|{end}"
        publisher.publish(topic_path, message_str.encode("utf-8"))

    print("All tasks published to Pub/Sub.")

def main(argv):
    spanner_client = spanner.Client()
    instance = spanner_client.instance(SPANNER_INSTANCE)
    database = instance.database(SPANNER_DATABASE)

    try:
        # 1. Get total valid case count from CaseSet
        total_valid = get_total_valid_cases(database, _CASE_SET_ID.value)
        print(f"Found {total_valid:,} valid cases for '{_CASE_SET_ID.value}'.")

        # 2. Register Work Buckets in Spanner for tracking
        create_buckets_in_spanner(database, _CASE_SET_ID.value, _RUN_ID.value, total_valid, _BUCKET_SIZE.value)

        # 3. Queue the work in Pub/Sub for workers to consume
        publish_to_pubsub(_CASE_SET_ID.value, _RUN_ID.value, total_valid, _BUCKET_SIZE.value)

        print("\nGMM Feeder setup complete. TPU Workers can now begin processing.")

    except Exception as e:
        print(f"Feeder failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    app.run(main)