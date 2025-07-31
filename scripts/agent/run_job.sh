#!/bin/bash
set -euo pipefail

#
# Check input argument
#
if [ $# -ne 1 ]; then
  echo "Usage: $0 <RECORD_ID>"
  exit 1
fi

RECORD_ID="$1"
echo "Record ID: $RECORD_ID"

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
# Run job in docker
#
if [[ "${LOCAL_RUN_BM:-}" == "1" ]]; then
  echo "Running locally..."
  # ./scripts/agent/local_run_bm.sh "artifacts/${RECORD_ID}.env"
  ./scripts/agent/no_install_run_bm.sh "artifacts/${RECORD_ID}.env"
else
  echo "Running job in docker..."
  ./scripts/agent/docker_run_bm.sh "artifacts/${RECORD_ID}.env"
fi
if [ $? -ne 0 ]; then
  echo "Error running job in docker."
  exit 1
fi
