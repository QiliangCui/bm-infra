#!/bin/bash

# 10 seconds

set -a
source /etc/environment
set +a

QUEUE_WAITING=5

SUBSCRIPTION_NAME="$GCP_QUEUE-agent"

# Trap SIGTERM and set flag
trap 'echo "Received SIGTERM."; should_exit=1' SIGTERM

while true; do
  if [[ "$should_exit" -eq 1 ]]; then
    echo "Exiting on SIGTERM."
    break
  fi

  echo "Polling for message..."

  MESSAGE=$(gcloud pubsub subscriptions pull "$SUBSCRIPTION_NAME" \
    --project="$GCP_PROJECT_ID" \
    --limit=1 \
    --format="json")

  if [ -z "$MESSAGE" ] || [[ "$MESSAGE" == "[]" ]]; then
    echo "No messages found."
    sleep $QUEUE_WAITING
    continue
  fi

  # Extract ackId and base64-encoded message body
  ACK_ID=$(echo "$MESSAGE" | jq -r '.[0].ackId')
  BASE64_DATA=$(echo "$MESSAGE" | jq -r '.[0].message.data')

  # Decode base64 to string
  MESSAGE_BODY=$(echo "$BASE64_DATA" | base64 --decode)

  echo "Received message: $MESSAGE_BODY"

  # Parse key-value pairs
  declare -A kv
  IFS=';' read -ra PAIRS <<< "$MESSAGE_BODY"
  for pair in "${PAIRS[@]}"; do
    KEY="${pair%%=*}"
    VALUE="${pair#*=}"
    kv[$KEY]="$VALUE"
  done

  RECORD_ID="${kv[RecordId]}"  

  echo "Parsed RecordId: $RECORD_ID"  

  if [ -z "$RECORD_ID" ] || [ "$RECORD_ID" == "null" ]; then
    echo "Invalid or missing record_id. Skipping message without ack."
    continue
  fi
  
  # Query TryCount as JSON
  echo "Query $RECORD_ID..."
  TRY_COUNT_JSON=$(gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --format=json \
    --sql="SELECT TryCount FROM RunRecord WHERE RecordId='$RECORD_ID'")

  # Parse TryCount using jq, convert to number, default to 0 if missing
  TRY_COUNT=$(echo "$TRY_COUNT_JSON" | jq -r '.rows[0][0] | tonumber // 0')

  # Increment TryCount
  NEW_TRY_COUNT=$((TRY_COUNT + 1))

  # Update record
  echo "Updating record $RECORD_ID: Status=RUNNING, TryCount=$NEW_TRY_COUNT, RunBy=$GCP_INSTANCE_NAME..."
  gcloud spanner databases execute-sql "$GCP_DATABASE_ID" \
    --project="$GCP_PROJECT_ID" \
    --instance="$GCP_INSTANCE_ID" \
    --sql="UPDATE RunRecord SET Status='RUNNING', LastUpdate=CURRENT_TIMESTAMP(), TryCount=$NEW_TRY_COUNT, RunBy='${GCP_INSTANCE_NAME}' WHERE RecordId='$RECORD_ID'"

  gcloud pubsub subscriptions ack "$SUBSCRIPTION_NAME" \
    --project="$GCP_PROJECT_ID" \
    --ack-ids="$ACK_ID"
  echo "Message acknowledged."  
  
  #
  # do the work
  # 
  echo "./scripts/agent/run_job.sh $RECORD_ID"
  ./scripts/agent/run_job.sh "$RECORD_ID"
  
  #
  # Report result
  #
  echo "Reporting result..."
  ./scripts/agent/report_result.sh "$RECORD_ID"

  echo "./scripts/cleanup_docker.sh"
  ./scripts/cleanup_docker.sh

  echo "Cleaning up old vllm envs..."
  ./scripts/agent/clean_old_vllm_envs.sh || true
  ./scripts/agent/clean_old_vllm_envs_v2.sh || true
done