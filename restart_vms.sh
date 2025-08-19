#!/bin/bash

# This script dynamically fetches and restarts a series of TPU VMs.
# It requires both a name pattern and a specific zone to be provided
# to ensure operations are targeted and safe.

# --- Default Configuration ---
DRY_RUN=false
VM_NAME_FILTER=""
ZONE_FILTER=""

# --- Argument Parsing ---
# Robust loop to handle flags and arguments in any order.
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--dry-run)
            DRY_RUN=true
            shift # consume flag
            ;;
        -z|--zone)
            if [[ -n "$2" && "$2" != -* ]]; then
                ZONE_FILTER="$2"
                shift 2 # consume flag and its value
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
                echo "Error: Multiple name filters provided ('$VM_NAME_FILTER' and '$1'). Please provide only one." >&2
                exit 1
            fi
            VM_NAME_FILTER="$1"
            shift # consume the filter
            ;;
    esac
done

# --- Usage Validation ---
# Exit if either the name filter or the zone is missing.
if [ -z "$VM_NAME_FILTER" ] || [ -z "$ZONE_FILTER" ]; then
  echo "Error: Both a VM name filter and a zone are required."
  echo ""
  echo "Usage: $0 -z <ZONE> <VM_NAME_FILTER> [-d|--dry-run]"
  echo ""
  echo "Arguments:"
  echo "  <VM_NAME_FILTER>    (Required) The name pattern of the VMs to target."
  echo "  -z, --zone <ZONE>   (Required) The specific zone to operate in."
  echo "  -d, --dry-run       (Optional) Show what would be done without making changes."
  echo ""
  echo "Example:"
  echo "  ./restart_vms.sh --zone us-east5-b vllm-tpu-v6e-8"
  exit 1
fi

# --- Configuration ---
MAX_RETRIES=3
RETRY_DELAY=15

# --- Function to execute a command with retries ---
function retry_command() {
  local attempt=1
  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Attempt $attempt of $MAX_RETRIES..."
    "$@"
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
      return 0 # Success
    fi
    echo "Command failed with exit code $exit_code."
    if [ $attempt -lt $MAX_RETRIES ]; then
      echo "Waiting ${RETRY_DELAY}s before retrying..."
      sleep $RETRY_DELAY
    fi
    ((attempt++))
  done
  echo "Command failed after $MAX_RETRIES attempts."
  return 1 # Failure
}

# --- Main Script Logic ---
success_count=0
failure_count=0
vms_found=0

if [ "$DRY_RUN" = true ]; then
  echo "*** DRY RUN MODE ENABLED ***"
  echo "The script will list actions but will not execute them."
  echo ""
fi

echo "Fetching list of TPU VMs..."
echo "   - Name Filter: '$VM_NAME_FILTER'"
echo "   - Zone Filter: '$ZONE_FILTER'"
echo ""

# Build the gcloud command and its arguments using a bash array for safety.
GCLOUD_ARGS=("gcloud" "compute" "tpus" "tpu-vm" "list")
GCLOUD_ARGS+=("--zone" "$ZONE_FILTER")
GCLOUD_ARGS+=("--filter" "name:'${VM_NAME_FILTER}'")
# We only need the name from the output now.
GCLOUD_ARGS+=("--format" "value(name)")


# Execute the command, reading only the TPU name from the output.
"${GCLOUD_ARGS[@]}" | while read -r tpu_name || [ -n "${tpu_name}" ]; do
  if [ -z "$tpu_name" ]; then
    continue
  fi

  ((vms_found++))
  # Use the ZONE_FILTER variable since we already know the zone.
  echo "--- Processing: $tpu_name in $ZONE_FILTER ---"

  if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would attempt to STOP $tpu_name."
    echo "[DRY RUN] Would attempt to START $tpu_name."
    echo "[DRY RUN] Simulated restart for '$tpu_name'."
    ((success_count++))
  else
    # Use the ZONE_FILTER variable for the stop and start commands.
    echo "Attempting to STOP $tpu_name..."
    if ! retry_command gcloud compute tpus tpu-vm stop "$tpu_name" --zone="$ZONE_FILTER"; then
      echo "ERROR: Failed to stop '$tpu_name'. Skipping this VM."
      ((failure_count++))
      echo "-----------------------------------------------------" && echo ""
      continue
    fi
    echo "Successfully stopped '$tpu_name'."

    echo "Attempting to START $tpu_name..."
    if ! retry_command gcloud compute tpus tpu-vm start "$tpu_name" --zone="$ZONE_FILTER"; then
      echo "ERROR: Failed to start '$tpu_name'. This VM may require manual intervention."
      ((failure_count++))
      echo "-----------------------------------------------------" && echo ""
      continue
    fi
    echo "Successfully restarted '$tpu_name'."
    ((success_count++))
  fi
  echo "-----------------------------------------------------"
  echo ""
done

if [ "$vms_found" -eq 0 ]; then
  echo "Warning: No VMs found matching the specified filters."
fi

DRY_RUN_MSG=""
if [ "$DRY_RUN" = true ]; then
  DRY_RUN_MSG=" (Dry Run)"
fi

echo "All VMs have been processed${DRY_RUN_MSG}."
echo "Final Summary: Successes: $success_count, Failures: $failure_count"
