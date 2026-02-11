#!/bin/bash
#
# This script creates a Google Cloud TPU VM and monitors its status.
# If the VM is not 'READY', it waits. If the VM does not exist, it recreates it.

set -euo pipefail

# --- 0. PARSE ACCELERATOR TYPES, COUNTS, AND DISK SIZES ---
if [[ $# -eq 0 || $(($# % 3)) -ne 0 ]]; then
  echo "Usage: $0 <accelerator_type_1> <count_1> <disk_size_1> [<accelerator_type_2> <count_2> <disk_size_2> ...]" >&2
  echo "Example: $0 tpu7x-2 4 500GB tpu-v5p-16 2 1TB" >&2
  exit 1
fi

ACCELERATOR_TYPES=()
TPU_COUNTS=()
DISK_SIZES=()

while [[ $# -gt 0 ]]; do
  ACCELERATOR_TYPE_ARG="$1"
  TPU_COUNT_ARG="$2"
  DISK_SIZE_ARG="$3"

  if ! [[ "$TPU_COUNT_ARG" =~ ^[0-9]+$ && "$TPU_COUNT_ARG" -gt 0 ]]; then
    echo "Error: Count for '$ACCELERATOR_TYPE_ARG' must be a positive integer, but got '$TPU_COUNT_ARG'." >&2
    exit 1
  fi

  if ! [[ "$DISK_SIZE_ARG" =~ ^[0-9]+(GB|TB)$ ]]; then
    echo "Error: Disk size for '$ACCELERATOR_TYPE_ARG' must be in a valid format (e.g., '500GB', '1TB'), but got '$DISK_SIZE_ARG'." >&2
    exit 1
  fi

  ACCELERATOR_TYPES+=("$ACCELERATOR_TYPE_ARG")
  TPU_COUNTS+=("$TPU_COUNT_ARG")
  DISK_SIZES+=("$DISK_SIZE_ARG")
  shift 3
done

echo "Configuration:"
for i in "${!ACCELERATOR_TYPES[@]}"; do
  echo "  - Accelerator Type: ${ACCELERATOR_TYPES[$i]}, Count: ${TPU_COUNTS[$i]}, Disk Size: ${DISK_SIZES[$i]}"
done

# --- Script Parameters ---
# export GCP_PROJECT_ID_FOR_TPU_VM="cloud-tpu-multipod-dev"
export GCP_PROJECT_ID_FOR_TPU_VM="cloud-tpu-inference-test"
export GCP_PROJECT_ID_FOR_OTHER_RESOURCES="cloud-tpu-inference-test"
# export TPU_NAME="amangu-v7x-2"
# export ZONE="us-central2-newvmos"
export ZONE="us-central1-c"
# export RUNTIME_VERSION="v2-test-tpu7-ubuntu2404"
export RUNTIME_VERSION="v2-alpha-tpu7-ubuntu2404"
export DISK_TYPE="hyperdisk-balanced"
export WAIT_INTERVAL=30 # Seconds to wait between status checks

# --- TPU VM Startup Script Placeholder params ---
export project_id="${GCP_PROJECT_ID_FOR_OTHER_RESOURCES}"
export spanner_instance="vllm-bm-inst"
export spanner_db="vllm-bm-runs"
export region="us-central2"
export purpose="bm"
# export instance_name - set in loop
export gcs_bucket="vllm-cb-storage2"
# export branch_hash="a256cdb8a20211c7a24c06bd644a0d8184be5714"  # amangu/iw_support
export branch_name="amangu/iw_support"
# export persistent_device_name - set in loop

# --- Prepare startup script template ---
# echo "Cleaning up old bm-infra..."
rm -rf bm-infra

# sudo apt-get update && sudo apt-get install git

# echo "Cloning branch '${branch_name}' from bm-infra..."
git clone --branch "${branch_name}" https://github.com/QiliangCui/bm-infra.git

INPUT_TEMPLATE="./startup_tpu7x.sh.tpl"
ALLOWED_VARS='${project_id} ${spanner_instance} ${spanner_db} ${region} ${purpose} ${instance_name} ${gcs_bucket} ${accelerator_type} ${branch_name} ${persistent_device_name}'

echo "Startup script template: $INPUT_TEMPLATE"
echo "-------------------------------------------------"


# --- Function to ensure Pub/Sub topic and subscription exist ---
#
# Arguments:
#   $1: The name of the Pub/Sub topic.
#   $2: The name of the Pub/Sub subscription.
#
# TODO: Change the project ID from $GCP_PROJECT_ID_FOR_TPU_VM to $GCP_PROJECT_ID_FOR_OTHER_RESOURCES
ensure_pubsub() {
  local TOPIC_NAME="$1"
  local SUBSCRIPTION_NAME="$2"

  echo "Verifying Pub/Sub setup for Topic: '$TOPIC_NAME' and Subscription: '$SUBSCRIPTION_NAME'..."

  # Check and Create the Topic
  if ! gcloud pubsub topics describe "$TOPIC_NAME" --project="$GCP_PROJECT_ID_FOR_OTHER_RESOURCES" --quiet &>/dev/null; then
    echo "Topic not found. Creating topic '$TOPIC_NAME'..."
    gcloud pubsub topics create "$TOPIC_NAME" --project="$GCP_PROJECT_ID_FOR_OTHER_RESOURCES"
    echo "✅ Topic created successfully."
  else
    echo "Topic '$TOPIC_NAME' already exists."
  fi

  # Check and Create the Subscription
  if ! gcloud pubsub subscriptions describe "$SUBSCRIPTION_NAME" --project="$GCP_PROJECT_ID_FOR_OTHER_RESOURCES" --quiet &>/dev/null; then
    echo "Subscription not found. Creating subscription '$SUBSCRIPTION_NAME' for topic '$TOPIC_NAME'..."
    gcloud pubsub subscriptions create "$SUBSCRIPTION_NAME" --topic="$TOPIC_NAME" --project="$GCP_PROJECT_ID_FOR_OTHER_RESOURCES" --ack-deadline=600
    echo "✅ Subscription created successfully."
  else
    echo "Subscription '$SUBSCRIPTION_NAME' already exists."
  fi
}

# --- Ensure Pub/Sub resources exist for all accelerator types ---
for ACCELERATOR_TYPE in "${ACCELERATOR_TYPES[@]}"; do
  TOPIC_NAME="vllm-bm-queue-${ACCELERATOR_TYPE}"
  SUBSCRIPTION_NAME="${TOPIC_NAME}-agent"
  ensure_pubsub "$TOPIC_NAME" "$SUBSCRIPTION_NAME"
done

# --- Function to set gcloud alias based on the ENV variable ---

ENV="prod"  # "prod" or "staging" (Hardcoded to prod for now)

# Set gcloud alias based on the ENV variable (test or staging)
set_gcloud_alias_based_on_test_environment() {
  shopt -s expand_aliases
  if [[ ${ENV} == "staging" ]]; then
    alias gcloud="CLOUDSDK_API_ENDPOINT_OVERRIDES_TPU=https://staging-tpu.sandbox.googleapis.com/ CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPUTE=https://www.googleapis.com/compute/staging_v1/ CLOUDSDK_API_CLIENT_OVERRIDES_COMPUTE=staging_v1 gcloud"
  elif [[ ${ENV} == "test" ]]; then
    alias gcloud="CLOUDSDK_API_ENDPOINT_OVERRIDES_TPU=https://test-tpu.sandbox.googleapis.com/  CLOUDSDK_API_ENDPOINT_OVERRIDES_COMPUTE=https://www.googleapis.com/compute/staging_v1/ CLOUDSDK_API_CLIENT_OVERRIDES_COMPUTE=staging_v1 gcloud"
  fi
}

set_gcloud_alias_based_on_test_environment

# --- Create and Monitor TPU VMs ---
while true; do
  echo "--- Starting new monitoring cycle ---"
  for idx in "${!ACCELERATOR_TYPES[@]}"; do
    ACCELERATOR_TYPE="${ACCELERATOR_TYPES[$idx]}"
    TPU_COUNT="${TPU_COUNTS[$idx]}"
    DISK_SIZE_FOR_TYPE="${DISK_SIZES[$idx]}"

    # Define names based on the current accelerator type
    TPU_NAME_BASE="vllm-tpu-${ACCELERATOR_TYPE}-bm"
    DISK_NAME_BASE="vllm-hyperdisk-${ACCELERATOR_TYPE}"
    export accelerator_type="${ACCELERATOR_TYPE}" # For envsubst

    echo "Checking status for $TPU_COUNT VMs of type $ACCELERATOR_TYPE..."

    for i in $(seq 1 "$TPU_COUNT"); do
      TPU_NAME="${TPU_NAME_BASE}-${i}"
      DISK_NAME="${DISK_NAME_BASE}-${i}"

      # Check if the TPU VM exists and get its status.
      # The `|| true` prevents the script from exiting if the describe command fails (e.g., TPU not found).
      TPU_STATUS=$(gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
        --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
        --zone="$ZONE" \
        --format="value(state)" 2>/dev/null || true)

      # Case 1: The TPU does not exist. Create it.
      if [[ -z "$TPU_STATUS" ]]; then
        echo "TPU '$TPU_NAME' not found. Attempting to create it..."

        # Ensure disk exists, create if not.
        if gcloud compute disks describe "$DISK_NAME" --zone="$ZONE" --project="$GCP_PROJECT_ID_FOR_TPU_VM" &>/dev/null; then
          echo "Disk '$DISK_NAME' already exists."
        else
          echo "Disk '$DISK_NAME' not found. Creating disk..."
          if gcloud compute disks create "$DISK_NAME" \
            --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
            --zone="$ZONE" \
            --size="$DISK_SIZE_FOR_TYPE" \
            --type="$DISK_TYPE"; then
            echo "✅ Disk '$DISK_NAME' created successfully."
          else
            echo "❌ Failed to create disk '$DISK_NAME'. Skipping TPU creation for this VM."
            continue # next VM in for loop
          fi
        fi

        # Prepare startup script for this specific VM
        export instance_name="${TPU_NAME}"
        export persistent_device_name="disk/by-id/google-persistent-disk-1"
        OUTPUT_SCRIPT="rendered_startup_${ACCELERATOR_TYPE}_${i}.sh"
        envsubst "$ALLOWED_VARS" < "$INPUT_TEMPLATE" > "$OUTPUT_SCRIPT"
        echo "Rendered startup script for $TPU_NAME: $OUTPUT_SCRIPT"

        # Lauch creation in background
        (
          echo "Starting background creation for '$TPU_NAME'..."
          if gcloud alpha compute tpus tpu-vm create "$TPU_NAME" \
            --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
            --zone="$ZONE" \
            --accelerator-type="$ACCELERATOR_TYPE" \
            --version="$RUNTIME_VERSION" \
            --metadata-from-file=startup-script="$OUTPUT_SCRIPT" \
            --reserved; then
            echo "✅ Background creation process for $TPU_NAME finished."
          else
            echo "❌ Background creation process for $TPU_NAME failed."
          fi
        ) &
        echo "Creation for $TPU_NAME launched in background."

      # Case 2: The TPU is in the READY state.
      elif [[ "$TPU_STATUS" == "READY" ]]; then
        echo "✅ TPU '$TPU_NAME' is in READY state."

        # Check if the disk is already attached to the TPU VM. If not, attach it.
        if gcloud alpha compute tpus tpu-vm describe "$TPU_NAME" \
          --zone="$ZONE" \
          --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
          --format="value(dataDisks.sourceDisk)" 2>/dev/null | grep -Fq "/$DISK_NAME"; then
          echo "✅ Disk '$DISK_NAME' is already attached to $TPU_NAME. No action needed."
        else
          echo "TPU '$TPU_NAME' is READY but disk '$DISK_NAME' is not attached. Attempting to attach disk..."
          # Try to attach the disk. Wrap in 'if' to catch failures.
          if gcloud alpha compute tpus tpu-vm attach-disk "$TPU_NAME" \
            --disk="$DISK_NAME" \
            --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
            --zone="$ZONE"; then
            echo "✅ Disk '$DISK_NAME' attached successfully to $TPU_NAME."
          else
            echo "❌ Failed to attach disk '$DISK_NAME' to TPU VM '$TPU_NAME'. Will retry in next loop."
          fi
        fi

      # Case 3: The TPU is in a failed or terminated state.
      elif [[ "$TPU_STATUS" == "FAILED" || "$TPU_STATUS" == "TERMINATED" || "$TPU_STATUS" == "STOPPED" ]]; then
         echo "❗️ TPU '$TPU_NAME' is in a failed/stopped state ($TPU_STATUS). It will be deleted and recreated on the next check."
         # The next loop iteration will not find the TPU and will trigger the creation logic.
         gcloud alpha compute tpus tpu-vm delete "$TPU_NAME" \
            --project="$GCP_PROJECT_ID_FOR_TPU_VM" \
            --zone="$ZONE" \
            --quiet

      # Case 4: The TPU is in a transitional state (e.g., CREATING, RESTARTING).
      else
        echo "TPU '$TPU_NAME' is currently in state: $TPU_STATUS. Waiting..."
      fi
    done # end for loop over TPU VMs for one accelerator type
  done # end for loop over accelerator types

  # Wait before the next check.
  echo "[$(date)] Sleeping for $WAIT_INTERVAL seconds before checking all VMs again..."
  sleep "$WAIT_INTERVAL"
done
