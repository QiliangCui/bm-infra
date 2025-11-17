#!/bin/bash

# === Usage ===
if [ "$#" -ne 5 ]; then
  echo "Usage: $0 <input.csv|gs://path/to/input.csv> CODEHASH JOB_REFERENCE RUN_TYPE EXTRA_ENVS"
  exit 1
fi

CSV_FILE_ARG="$1"
CODEHASH="$2"
JOB_REFERENCE="$3"
RUN_TYPE="$4"
EXTRA_ENVS="$5"

if [[ "$CSV_FILE_ARG" == gs://* ]]; then
  echo "GCS path detected. Downloading from $CSV_FILE_ARG"
  CSV_FILE=$(mktemp)
  if ! gsutil cp "$CSV_FILE_ARG" "$CSV_FILE"; then
    echo "Failed to download from GCS: $CSV_FILE_ARG"
    rm "$CSV_FILE"
    exit 1
  fi
  # Schedule cleanup of the temporary file on exit
  trap 'rm -f "$CSV_FILE"' EXIT
else
  CSV_FILE="$CSV_FILE_ARG"
fi

if [ ! -f "$CSV_FILE" ]; then
  echo "CSV file not found: $CSV_FILE"
  exit 1
fi

# milliseconds: one hour
VERY_LARGE_EXPECTED_ETEL=3600000

# === Config ===
# Make sure these environment variables are set or export here
# export GCP_PROJECT_ID="your-project"
# export GCP_INSTANCE_ID="your-instance"
# export GCP_DATABASE_ID="your-database"

# === Read CSV and skip header ===
tail -n +2 "$CSV_FILE" | while read -r line || [ -n "${line}" ]; do

  line=$(echo "$line" | tr -d '\r')
    # Safely split CSV line into variables
  IFS=',' read -r \
    DEVICE \
    MODEL \
    MAX_NUM_SEQS \
    MAX_NUM_BATCHED_TOKENS \
    TENSOR_PARALLEL_SIZE \
    MAX_MODEL_LEN \
    DATASET \
    INPUT_LEN \
    OUTPUT_LEN \
    EXPECTED_ETEL \
    NUM_PROMPTS \
    MODELTAG \
    PREFIX_LEN \
    REQUEST_RATE <<< "$line"

  RECORD_ID=$(uuidgen | tr 'A-Z' 'a-z')

  # calculate the queue name from the device
  QUEUE_TOPIC="vllm-bm-queue-$DEVICE"

  # Check if the topic exists
  if ! gcloud pubsub topics describe "$QUEUE_TOPIC" --project="$GCP_PROJECT_ID" &>/dev/null; then
    echo "Topic '$QUEUE_TOPIC' does not exist in $GCP_PROJECT_ID."
    echo "Skip creating record in RunRecord table."
    continue
  fi
  
  echo "Inserting Run: $RECORD_ID"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen, LastUpdate, CreatedBy,JobReference, ExpectedETEL, NumPrompts, ModelTag, PrefixLen, RequestRate, ExtraEnvs,
    ) VALUES (
      '$RECORD_ID', 'CREATED', PENDING_COMMIT_TIMESTAMP(), '$DEVICE', '$MODEL', '$RUN_TYPE', '$CODEHASH',
      $MAX_NUM_SEQS,
      $MAX_NUM_BATCHED_TOKENS,
      $TENSOR_PARALLEL_SIZE,
      $MAX_MODEL_LEN,
      '$DATASET',
      $INPUT_LEN,
      $OUTPUT_LEN,
      PENDING_COMMIT_TIMESTAMP(),
      '$USER',
      '$JOB_REFERENCE',
      ${EXPECTED_ETEL:-$VERY_LARGE_EXPECTED_ETEL},
      ${NUM_PROMPTS:-1000},
      '${MODELTAG:-PROD}',
      ${PREFIX_LEN:-0},
      ${REQUEST_RATE:-inf},
      '$EXTRA_ENVS'
    );"
  
  # If insert failed, just continue without publishing
  if [ $? -ne 0 ]; then
    echo "Insert failed for $RECORD_ID — skipping publish." >&2
    continue
  fi

  echo "Publishing to Pub/Sub queue: $GCP_QUEUE"
  # Construct key-value string
  MESSAGE_BODY="RecordId=$RECORD_ID"
  # Publish the message
  gcloud pubsub topics publish $QUEUE_TOPIC \
    --project="$GCP_PROJECT_ID" \
    --message="$MESSAGE_BODY" > /dev/null

  echo "$RECORD_ID scheduled."
done