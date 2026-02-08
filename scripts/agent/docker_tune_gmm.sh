#!/bin/bash

# 1. Configuration from Environment (Set by Machine Definition)
# This file typically exists on GCP TPU VMs to provide instance metadata.
source /etc/environment

# These vars come from /etc/environment or use defaults
PROJECT_ID=${GCP_PROJECT_ID:-"cloud-tpu-inference-test"}
INSTANCE_ID=${GCP_INSTANCE_ID:-"vllm-bm-inst"}
DATABASE_ID=${GCP_DATABASE_ID:-"tune-gmm"} # Updated for GMM

# The subscription name follows the pattern from the MoE infra
# Based on your previous setup, the queue was 'vllm-tune-queue-tpu7x-2'
SUBSCRIPTION_ID="${GCP_QUEUE}-agent"

# Nightly TPU image used for benchmarking
# around 2/7/2026
IMAGE_NAME="vllm/vllm-tpu:nightly-ironwood-20260207-c515c6a-48312e5"

# Locate the GMM worker script relative to the infra path
LOCAL_INFRA_PATH="/home/$USER/bm-infra"
LOCAL_SCRIPT_PATH="$LOCAL_INFRA_PATH/scripts/agent/gmm_worker.py"
CONTAINER_DEST_PATH="/workspace/gmm_worker.py"

# Ensure the local script exists before attempting to run
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo "Error: Local script not found at $LOCAL_SCRIPT_PATH"
    echo "Ensure you have created gmm_worker.py in the scripts/agent/ directory."
    exit 1
fi

# 2. Pull the specific nightly image
echo "Pulling Ironwood nightly TPU image..."
docker pull "$IMAGE_NAME"

# 3. Launch GMM TPU Worker
# Consistent with MoE pattern: privileged mode, host network, and volume mounts
echo "Launching GMM TPU Worker Container..."
docker run --rm \
    --name "tpu-tune-gmm" \
    -e PYTHONUNBUFFERED=1 \
    --privileged \
    --network host \
    --shm-size=16gb \
    -v "$LOCAL_INFRA_PATH":"/workspace/bm-infra" \
    -v "$LOCAL_SCRIPT_PATH":"$CONTAINER_DEST_PATH" \
    "$IMAGE_NAME" \
    /bin/bash -c "
        echo 'Updating Python Environment...' && \
        pip install --no-cache-dir \
            google-cloud-spanner \
            google-api-core \
            google-cloud-pubsub \
            absl-py \
            tqdm && \
        
        echo 'Exporting PYTHONPATH...' && \
        export PYTHONPATH=\$PYTHONPATH:/workspace/bm-infra && \
        
        echo 'Starting GMM worker process...' && \
        python3 $CONTAINER_DEST_PATH \
            --project_id='$PROJECT_ID' \
            --subscription_id='$SUBSCRIPTION_ID' \
            --instance_id='$INSTANCE_ID' \
            --database_id='$DATABASE_ID' \
            --worker_id='$GCP_INSTANCE_NAME' \
            --debug
    "