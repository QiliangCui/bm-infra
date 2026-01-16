#!/bin/bash

# 1. Configuration from Environment (Set by Machine Definition)

source /etc/environment

# These vars come from /etc/environment
PROJECT_ID=${GCP_PROJECT_ID:-"cloud-tpu-inference-test"}
INSTANCE_ID=${GCP_INSTANCE_ID:-"vllm-bm-inst"}
DATABASE_ID=${GCP_DATABASE_ID:-"tune-moe"}
# The subscription name is usually the queue name + "-agent" based on your Terraform
SUBSCRIPTION_ID="${GCP_QUEUE}-agent"

IMAGE_NAME="vllm/vllm-tpu:nightly-ironwood-20260115-8b93316-4c1c501"

# Since the machine definition clones the repo to /home/bm-agent/bm-infra,
# we locate the script relative to that path.
# Assuming the script is at: scripts/agent/moe_worker.py
LOCAL_INFRA_PATH="/home/$USER/bm-infra"
LOCAL_SCRIPT_PATH="$LOCAL_INFRA_PATH/scripts/agent/moe_worker.py"
CONTAINER_DEST_PATH="/workspace/moe_worker.py"

# Ensure the local script exists
if [ ! -f "$LOCAL_SCRIPT_PATH" ]; then
    echo "Error: Local script not found at $LOCAL_SCRIPT_PATH"
    echo "Ensure git clone was successful in the machine definition."
    exit 1
fi

# 2. Pull the specific nightly image
echo "Pulling Ironwood nightly TPU image..."
docker pull "$IMAGE_NAME"

# 3. Launch TPU Worker
# Note: We pass the host's /etc/environment or specific env vars so the 
# containerized python script can see them if needed.
echo "Launching TPU Worker Container for Run..."
docker run --rm \
    --name "tpu-tune" \
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
        
        echo 'Starting worker process...' && \
        python3 $CONTAINER_DEST_PATH \
            --project_id='$PROJECT_ID' \
            --subscription_id='$SUBSCRIPTION_ID' \
            --instance_id='$INSTANCE_ID' \
            --database_id='$DATABASE_ID' \
            --worker_id='$GCP_INSTANCE_NAME' \
            --debug
    "