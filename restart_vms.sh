#!/bin/bash

# This script dynamically fetches and restarts a series of TPU VMs.
# It checks the initial state of each VM and only issues a 'stop'
# command if the VM is currently RUNNING.

# --- Default Configuration ---
DRY_RUN=false
VM_NAME_FILTER=""
ZONE_FILTER=""
PROJECT_ID="cloud-tpu-inference-test" # Set your project ID here

# --- Argument Parsing ---
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -z|--zone)
            if [[ -n "$2" && "$2" != -* ]]; then
                ZONE_FILTER="$2"
                shift 2
            else
                echo "Error: The --zone option requires a value." >&2
                exit 1
            fi
            ;;
        -*)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
        *)
            if [ -n "$VM_NAME_FILTER" ]; then
                echo "Error: Multiple name filters provided ('$VM_NAME_FILTER' and '$1')." >&2
                exit 1
            fi
            VM_NAME_FILTER="$1"
            shift
            ;;
    esac
done

# --- Usage Validation ---
if [ -z "$VM_NAME_FILTER" ] || [ -z "$ZONE_FILTER" ]; then
  echo "Error: Both a VM name filter and a zone are required."
  echo ""
  echo "Usage: $0 -z <ZONE> <VM_NAME_FILTER> [-d|--dry-run]"
  exit 1
fi

# --- Configuration ---
MAX_START_RETRIES=4
START_RETRY_DELAY=30
STATUS_CHECK_DELAY=15
MAX_STATUS_CHECKS=20

# --- Helper Functions (retry_command, wait_for_status) ---
function retry_command() {
  local attempt=1
  local exit_code
  while [ $attempt -le $MAX_START_RETRIES ]; do
    echo "--> Attempt $attempt of $MAX_START_RETRIES to issue command: ${@}"
    "$@"
    exit_code=$?
    if [ $exit_code -eq 0 ]; then echo "--> Command submitted successfully."; return 0; fi
    echo "--> Command failed with exit code $exit_code."
    if [ $attempt -lt $MAX_START_RETRIES ]; then echo "    Waiting ${START_RETRY_DELAY}s before retrying..."; sleep $START_RETRY_DELAY; fi
    ((attempt++))
  done
  echo "--> Command failed after $MAX_START_RETRIES attempts."
  return 1
}
function wait_for_status() {
  local tpu_name="$1"; local zone="$2"; local target_status="$3"; local checks=0
  if [ "$DRY_RUN" = true ]; then echo "[DRY RUN] Would wait for '$tpu_name' to reach status '$target_status'."; return 0; fi
  echo "Waiting for '$tpu_name' to enter '$target_status' state..."
  while [ $checks -lt $MAX_STATUS_CHECKS ]; do
    local current_status=$(gcloud compute tpus tpu-vm describe "$tpu_name" --zone="$zone" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null)
    if [[ "$current_status" == "$target_status" ]]; then echo "Success: '$tpu_name' is now $target_status."; return 0; fi
    echo "  - Current status is '$current_status', checking again in ${STATUS_CHECK_DELAY}s..."
    sleep $STATUS_CHECK_DELAY
    ((checks++))
  done
  echo "ERROR: Timed out waiting for '$tpu_name' to reach '$target_status' state."
  return 1
}

# --- Main Script Logic ---
success_count=0

if [ "$DRY_RUN" = true ]; then
  echo "*** DRY RUN MODE ENABLED ***"
fi
echo "Fetching TPU VMs in project '$PROJECT_ID'..."
echo ""

# 💡 CORRECTED: Reverted to your original, working filter syntax
GCLOUD_ARGS=(
    "gcloud" "compute" "tpus" "tpu-vm" "list"
    "--project=$PROJECT_ID"
    "--zone=$ZONE_FILTER"
    "--filter=name:'${VM_NAME_FILTER}'" # Use colon (:) for prefix match
    "--format=value(name)"
)
VM_LIST=$("${GCLOUD_ARGS[@]}")


if [ -z "$VM_LIST" ]; then
  echo "Warning: No VMs found matching the specified filters."
  exit 0
fi

for tpu_name in $VM_LIST; do
  echo "--- Processing: $tpu_name ---"

  current_status=$(gcloud compute tpus tpu-vm describe "$tpu_name" --zone="$ZONE_FILTER" --project="$PROJECT_ID" --format="value(state)" 2>/dev/null)
  echo "Initial status of '$tpu_name' is '$current_status'."

  stop_needed=false
  if [[ "$current_status" == "RUNNING" ]]; then
      stop_needed=true
  elif [[ "$current_status" == "STOPPED" ]]; then
      echo "VM is already stopped. Proceeding directly to start."
  else
      echo "Warning: VM is in an unexpected state ('$current_status'). Skipping this VM."
      echo "-----------------------------------------------------" && echo ""
      continue
  fi

  if [ "$DRY_RUN" = true ]; then
    if [ "$stop_needed" = true ]; then
        echo "[DRY RUN] Would STOP $tpu_name."
        wait_for_status "$tpu_name" "$ZONE_FILTER" "STOPPED"
    fi
    echo "[DRY RUN] Would START $tpu_name with $MAX_START_RETRIES retries."
    wait_for_status "$tpu_name" "$ZONE_FILTER" "RUNNING"
    ((success_count++))
  else
    if [ "$stop_needed" = true ]; then
        echo "Attempting to STOP $tpu_name..."
        gcloud compute tpus tpu-vm stop "$tpu_name" --zone="$ZONE_FILTER" --project="$PROJECT_ID" --quiet
        if ! wait_for_status "$tpu_name" "$ZONE_FILTER" "STOPPED"; then
            echo "ERROR: Failed to confirm stop for '$tpu_name'. Skipping this VM."
            echo "-----------------------------------------------------" && echo ""
            continue
        fi
    fi

    echo "Attempting to START $tpu_name..."
    if retry_command gcloud compute tpus tpu-vm start "$tpu_name" --zone="$ZONE_FILTER" --project="$PROJECT_ID" --quiet; then
      if ! wait_for_status "$tpu_name" "$ZONE_FILTER" "RUNNING"; then
        echo "FATAL: 'start' command was sent, but '$tpu_name' FAILED TO REACH RUNNING STATE."
        echo "ABORTING SCRIPT to prevent further potential VM downtime."
        exit 1
      fi
    else
      echo "FATAL: FAILED TO SEND 'start' command for '$tpu_name' after $MAX_START_RETRIES attempts."
      echo "ABORTING SCRIPT due to persistent command failure."
      exit 1
    fi

    echo "Successfully started '$tpu_name'."
    ((success_count++))
  fi
  echo "-----------------------------------------------------"
  echo ""
done

DRY_RUN_MSG=""
if [ "$DRY_RUN" = true ]; then
  DRY_RUN_MSG=" (Dry Run)"
fi

echo "All VMs have been processed${DRY_RUN_MSG}."
echo "Final Summary: Succeeded: $success_count"
