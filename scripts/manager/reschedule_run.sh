#!/bin/bash

# === Usage ===
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 JOB_REFERENCE [STATUS]"
  exit 1
fi

JOB_REFERENCE="$1"
STATUS="${2:-FAILED}"

# === Config ===
# Ensure these are exported:
# export GCP_PROJECT_ID
# export GCP_INSTANCE_ID
# export GCP_DATABASE_ID

echo "Querying records with JobReference='$JOB_REFERENCE' and Status='$STATUS'..."

RECORDS_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --project="$GCP_PROJECT_ID" \
  --sql="SELECT RecordId, Device FROM RunRecord WHERE JobReference='$JOB_REFERENCE' AND Status='$STATUS';" \
  --format=json)

RECORD_COUNT=$(echo "$RECORDS_JSON" | jq '.rows | length')

if [ "$RECORD_COUNT" -eq 0 ]; then
  echo "No matching records found."
  exit 0
fi

echo "Found $RECORD_COUNT matching records."

echo "$RECORDS_JSON" | jq -c '.rows[]' | while read -r row; do
  RECORD_ID=$(echo "$row" | jq -r '.[0]')
  DEVICE=$(echo "$row" | jq -r '.[1]')
  if [[ $DEVICE == tpu7x* ]]; then
    QUEUE_TOPIC="vllm-tt-queue-$DEVICE"
  else
    QUEUE_TOPIC="vllm-bm-queue-$DEVICE"
  fi

  # Check if Pub/Sub topic exists
  if ! gcloud pubsub topics describe "$QUEUE_TOPIC" --project="$GCP_PROJECT_ID" &>/dev/null; then
    echo "Topic '$QUEUE_TOPIC' does not exist. Skipping RecordId=$RECORD_ID."
    continue
  fi

  echo "Updating RecordId=$RECORD_ID to RECREATED..."

  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="UPDATE RunRecord SET Status='RECREATED', LastUpdate=PENDING_COMMIT_TIMESTAMP() WHERE RecordId='$RECORD_ID';"

  echo "Publishing RecordId=$RECORD_ID to $QUEUE_TOPIC..."

  gcloud pubsub topics publish "$QUEUE_TOPIC" \
    --project="$GCP_PROJECT_ID" \
    --message="RecordId=$RECORD_ID"
done
