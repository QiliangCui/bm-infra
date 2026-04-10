#!/bin/bash

# === Usage ===
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 JOB_REFERENCE"
  exit 1
fi

JOB_REFERENCE="$1"

# === Config ===
# Ensure these are exported:
# export GCP_PROJECT_ID
# export GCP_INSTANCE_ID
# export GCP_DATABASE_ID

echo "Querying records with JobReference='$JOB_REFERENCE' and Status in (CREATED, RUNNING)..."

RECORDS_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --project="$GCP_PROJECT_ID" \
  --sql="SELECT RecordId FROM RunRecord WHERE JobReference='$JOB_REFERENCE' AND Status IN ('CREATED', 'RUNNING');" \
  --format=json)

RECORD_COUNT=$(echo "$RECORDS_JSON" | jq '.rows | length')

if [ "$RECORD_COUNT" -eq 0 ]; then
  echo "No cancellable records found for JobReference='$JOB_REFERENCE'."
  exit 0
fi

echo "Found $RECORD_COUNT record(s). Marking as CANCELLED..."

echo "$RECORDS_JSON" | jq -c '.rows[]' | while read -r row; do
  RECORD_ID=$(echo "$row" | jq -r '.[0]')

  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --project="$GCP_PROJECT_ID" \
    --sql="UPDATE RunRecord SET Status='CANCELLED', LastUpdate=PENDING_COMMIT_TIMESTAMP() WHERE RecordId='$RECORD_ID';"

  echo "Cancelled RecordId=$RECORD_ID"
done
