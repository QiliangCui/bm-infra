#!/bin/bash
set -euo pipefail

# Record the exact start time in a format journalctl understands.
SCRIPT_START_TIME=$(date +"%Y-%m-%d %H:%M:%S")

#
# Check input argument
#
if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
echo "Record ID: $RECORD_ID"

#
# This function will be called by the trap on any script error.
#
upload_failure_logs() {
  echo "--- SCRIPT FAILED: Capturing journalctl logs ---"

  # systemd automatically provides this env var with the service name.
  # This is much more robust than hardcoding the service name!
  SERVICE_NAME=${_SYSTEMD_UNIT:-"bm-agent.service"}

  # Create a temporary file for the logs.
  LOG_FILE=$(mktemp "/tmp/${SERVICE_NAME}-failure-log-${RECORD_ID}.txt")

  echo "Capturing logs for service '$SERVICE_NAME' since '$SCRIPT_START_TIME'..."

  # Grab all journal logs for this specific service unit since the script started.
  journalctl -u "$SERVICE_NAME" --since "$SCRIPT_START_TIME" > "$LOG_FILE"

  # Define where to upload the log in GCS.
  # Assumes GCS_BUCKET and RECORD_ID are available from your env file.
  if [[ -n "${GCS_BUCKET:-}" && -n "${RECORD_ID:-}" ]]; then
    GCS_PATH="gs://$GCS_BUCKET/job_logs/$RECORD_ID/failure_journal_run_job.txt"
    echo "Uploading failure log to $GCS_PATH"
    gsutil cp "$LOG_FILE" "$GCS_PATH"
  else
    echo "GCS_BUCKET or RECORD_ID not set. Cannot upload failure log."
  fi

  # Clean up the temporary file.
  rm "$LOG_FILE"

  # The script will exit automatically after the trap finishes.
}

# Set the trap. This 'arms' the function to run on any command failure.
trap 'upload_failure_logs' ERR

echo "deleting artifacts".
rm -rf artifacts

#
# Create running config
#
echo "Creating running config..."
./scripts/agent/create_config.sh "$RECORD_ID"
if [ $? -ne 0 ]; then
  echo "Error creating running config."
  exit 1
fi

#
# This makes GCS_BUCKET and other vars available to the whole script.
#
ENV_FILE="artifacts/${RECORD_ID}.env"
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
else
  echo "Error: Config file $ENV_FILE not found after create_config.sh"
  exit 1
fi

#
# Run job in docker
#
if [[ "${LOCAL_RUN_BM:-}" == "1" ]]; then
  echo "Running locally..."
  ./scripts/agent/local_run_bm.sh "artifacts/${RECORD_ID}.env"
elif [[ "${LOCAL_RUN_BM:-}" == "2" ]]; then
  echo "Running locally with V2..."
  ./scripts/agent/local_run_bm_v2.sh "artifacts/${RECORD_ID}.env"
else
  echo "Running job in docker..."
  ./scripts/agent/docker_run_bm.sh "artifacts/${RECORD_ID}.env"
fi
if [ $? -ne 0 ]; then
  echo "Error running job in docker."
  exit 1
fi

echo "Benchmark script completed successfully."

#
# Disarm the trap on successful completion.
#
trap - ERR

exit 0
