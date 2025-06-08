#!/bin/bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
RESULT_FILE="artifacts/${RECORD_ID}.result"

# Ensure GCP_INSTANCE_NAME is set
: "${GCP_INSTANCE_NAME:?Need to set GCP_INSTANCE_NAME}"

# Case 1: result file does not exist → mark as FAILED
if [ ! -f "$RESULT_FILE" ]; then
  echo "Result file not found: $RESULT_FILE. Marking status as FAILED."

  SQL="UPDATE RunRecord SET Status='FAILED', RunBy='${GCP_INSTANCE_NAME}', LastUpdate=CURRENT_TIMESTAMP() WHERE RecordId = '${RECORD_ID}';"

  echo "Executing SQL:"
  echo "$SQL"

  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="$SQL"

  exit 0
fi

# Case 2: result file exists → parse and mark as COMPLETED
assignments=""
while IFS='=' read -r key value; do
  if [[ -n "$key" && -n "$value" ]]; then
    if [[ "$value" =~ ^[0-9.]+$ ]]; then
      assignments+="${key}=${value}, "
    else
      assignments+="${key}='${value}', "
    fi
  fi
done < "$RESULT_FILE"

# Clean up trailing comma+space
assignments="${assignments%, }"

# Add status, RunBy, and timestamp
assignments="${assignments}, Status='COMPLETED', RunBy='${GCP_INSTANCE_NAME}', LastUpdate=CURRENT_TIMESTAMP()"

# Build SQL
SQL="UPDATE RunRecord SET ${assignments} WHERE RecordId = '${RECORD_ID}';"

echo "Executing SQL:"
echo "$SQL"

gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
  --project="$GCP_PROJECT_ID" \
  --instance="$GCP_INSTANCE_ID" \
  --sql="$SQL"
