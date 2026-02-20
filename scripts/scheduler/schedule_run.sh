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
# Using tail -n +2 to skip header.
tail -n +2 "$CSV_FILE" | while read -r line || [ -n "${line}" ]; do
  # Remove carriage returns from Windows-formatted CSVs
  line=$(echo "$line" | tr -d '\r')

  # Skip empty lines
  [ -z "$line" ] && continue

  # Use Python to safely parse the CSV line.
  # This handles internal commas inside quoted fields (like JSON in AdditionalConfig).
  # mapfile captures the output into an array called 'fields'.
  mapfile -t fields < <(python3 -c "import csv, sys; line = sys.stdin.read(); reader = csv.reader([line], quotechar=\"'\"); print('\n'.join(next(reader)))" <<< "$line")

  # Map array indices to descriptive variables
  DEVICE="${fields[0]}"
  MODEL="${fields[1]}"
  MAX_NUM_SEQS="${fields[2]}"
  MAX_NUM_BATCHED_TOKENS="${fields[3]}"
  TENSOR_PARALLEL_SIZE="${fields[4]}"
  MAX_MODEL_LEN="${fields[5]}"
  DATASET="${fields[6]}"
  INPUT_LEN="${fields[7]}"
  OUTPUT_LEN="${fields[8]}"
  EXPECTED_ETEL="${fields[9]}"
  NUM_PROMPTS="${fields[10]}"
  MODELTAG="${fields[11]}"
  PREFIX_LEN="${fields[12]}"
  ADDITIONAL_CONFIG="${fields[13]}"
  EXTRA_ARGS="${fields[14]}"

  RECORD_ID=$(uuidgen | tr 'A-Z' 'a-z')

  # calculate the queue name from the device
  QUEUE_TOPIC="vllm-bm-queue-$DEVICE"

  # Check if the topic exists
  if ! gcloud pubsub topics describe "$QUEUE_TOPIC" --project="$GCP_PROJECT_ID" &>/dev/null; then
    echo "Topic '$QUEUE_TOPIC' does not exist in $GCP_PROJECT_ID."
    echo "Skip creating record in RunRecord table."
    continue
  fi

  # Helper to handle SQL quoting for JSON/String fields.
  # This now properly escapes internal single quotes by doubling them (' -> '').
  prepare_sql_val() {
    local val="$1"
    local default="$2"

    # if empty, return default
    if [ -z "$val" ]; then
      echo "$default"
      return
    fi

    # Remove leading/trailing single quotes if the CSV parser preserved them
    val="${val#\'}"
    val="${val%\'}"

    # Escape internal single quotes for Spanner SQL (replace ' with '')
    local escaped_val="${val//\'/\'\'}"

    # Wrap the escaped value in single quotes for the SQL statement
    echo "'$escaped_val'"
  }

  SQL_ADDITIONAL_CONFIG=$(prepare_sql_val "$ADDITIONAL_CONFIG" "'{}'")
  SQL_EXTRA_ARGS=$(prepare_sql_val "$EXTRA_ARGS" "''")

  echo "Inserting Run: $RECORD_ID"
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen,
      Dataset, InputLen, OutputLen, LastUpdate, CreatedBy, JobReference,
      ExpectedETEL, NumPrompts, ModelTag, PrefixLen, ExtraEnvs,
      AdditionalConfig, ExtraArgs
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
      '$EXTRA_ENVS',
      $SQL_ADDITIONAL_CONFIG,
      $SQL_EXTRA_ARGS
    );"
  
  # If insert failed, just continue without publishing
  if [ $? -ne 0 ]; then
    echo "Insert failed for $RECORD_ID — skipping publish." >&2
    continue
  fi

  echo "Publishing to Pub/Sub queue: $QUEUE_TOPIC"
  # Construct key-value string
  MESSAGE_BODY="RecordId=$RECORD_ID"

  # Publish the message
  gcloud pubsub topics publish "$QUEUE_TOPIC" \
    --project="$GCP_PROJECT_ID" \
    --message="$MESSAGE_BODY" > /dev/null

  echo "$RECORD_ID scheduled."
done
