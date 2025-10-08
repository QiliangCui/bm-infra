#!/bin/bash
set -euo pipefail

# Argument Explanation:
#
# 1. INPUT_CSV       - (Required) Path to the input CSV file to process.
#
# 2. CODE_HASH       - (Optional) A string representing the code version hash.
#
# 3. JOB_REFERENCE   - (Optional) Identifier for the job or run reference.
#
# 4. RUN_TYPE        - (Optional) Type of run or execution mode (e.g., HOURLY, AUTOTUNE).
#
# 5. REPO            - (Optional) Repository name or URL related to the job.
#
# 6. EXTRA_ENVS      - (Optional) Additional environment variables to set, formatted as
#                     a semicolon-separated list of key=value pairs.
#                     Example:
#                       VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=1
#
#                     These variables can be parsed and exported within the script as needed.
#
#  Others:
#   REPO_MAP        - (Optional) An environment variable to map repository URLs to local
#                     filesystem paths. This accelerates setup by using local mirrors
#                     instead of performing a fresh `git clone` from the internet.
#
#                     The variable must be a string of one or more mappings, separated
#                     by semicolons (`;`). Each mapping consists of the full repository
#                     URL, a colon (`||`), and the absolute path to the local mirror.
#
#                     Example:
#                       export REPO_MAP="https://github.com/vllm-project/vllm.git:repos/vllm;https||//github.com/vllm-project/tpu-inference.git||repos/tpu-inference"
#
#                     If this variable is not set, or a specific URL is not found in
#                     the map, the script will gracefully fall back to `git clone`.
#
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <INPUT_CSV> [CODE_HASH] [JOB_REFERENCE] [RUN_TYPE] [REPO] [EXTRA_ENVS]"
    exit 1
fi

INPUT_CSV="$1"
CODE_HASH="${2:-}"  # optional
JOB_REFERENCE="${3:-}"
RUN_TYPE="${4:-"MANUAL"}"
REPO="${5:-"DEFAULT"}"
EXTRA_ENVS="${6:-}"

# ==============================================================================
# PARSE THE REPO_MAP ENVIRONMENT VARIABLE (ONCE)
# ==============================================================================
# Declare a global associative array to hold the parsed repository mappings.
declare -A REPO_MAP_ASSOC

# Check if the REPO_MAP environment variable is set and not empty.
if [[ -n "${REPO_MAP:-}" ]]; then
  echo "Found REPO_MAP environment variable, parsing local repository paths..."

  # Temporarily change the Internal Field Separator (IFS) to ';' to split pairs
  OLD_IFS="$IFS"
  IFS=';'
  # Create an array of "key:value" pairs from the string
  pairs_array=($REPO_MAP)
  IFS="$OLD_IFS" # Restore IFS immediately

  # Loop through the pairs and populate the associative array
  for pair in "${pairs_array[@]}"; do
    key="${pair%%||*}"
    value="${pair#*||}"
    REPO_MAP_ASSOC["$key"]="$value"
  done
fi
# ==============================================================================

if [[ "$REPO" != "DEFAULT" && "$REPO" != "TPU_INFERENCE" && "$REPO" != "TPU_INFERENCE_TORCHAX" ]]; then
  echo "Error: REPO must be one of: DEFAULT, TPU_INFERENCE, or TPU_INFERENCE_TORCHAX, but got '$REPO'"
  exit 1
fi

IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

echo "Recreating artifacts directory"
rm -rf artifacts/
mkdir -p artifacts/

clone_and_get_hash() {
  local repo_url="$1"
  local dest_folder="$2"
  local target_hash="$3"

  # Check for a local path in the global associative array
  local local_repo_path="${REPO_MAP_ASSOC[$repo_url]:-}"

  if [[ -n "$local_repo_path" ]]; then
    echo "Found local mapping for '$repo_url'. Copying from '$local_repo_path'..." >&2
    if [ ! -d "$local_repo_path" ]; then
        echo "Error: Mapped path '$local_repo_path' does not exist." >&2
        return 1
    fi
    cp -a "$local_repo_path" "$dest_folder"
  else
    echo "No local mapping found. Cloning from '$repo_url'..." >&2
    git clone "$repo_url" "$dest_folder"
  fi

  pushd "$dest_folder" > /dev/null

  # If target hash is provided, reset to it
  if [[ -n "$target_hash" ]]; then
    echo "Resetting to $target_hash" >&2
    git reset --hard "$target_hash" >&2
  fi

  # Get and return the short hash
  local resolved_hash
  resolved_hash=$(git rev-parse --short HEAD)
  popd > /dev/null

  echo "$resolved_hash"
}

if [[ "${SKIP_BUILD_IMAGE:-0}" != "1" ]]; then

  # Clone and get hash
  VLLM_HASH=$(clone_and_get_hash "https://github.com/vllm-project/vllm.git" "artifacts/vllm" "$VLLM_HASH")
  echo "resolved VLLM_HASH: $VLLM_HASH"

  # A temp solution to patch a fix.
  if [[ "${LOCAL_PATCH:-0}" == "1" ]]; then
    echo "Update the vllm locally."

    echo "bash ./tools/patch_core.sh"
    bash ./tools/patch_core.sh

    echo "pushd ./artifacts/vllm"
    pushd ./artifacts/vllm
    git add .
    git config user.name "$USER"
    git config user.email "$USER@google.com"
    git commit -m "temp patch."

    NEW_VLLM_HASH=$(git rev-parse --short HEAD)

    # RealHash_BaseHash
    VLLM_HASH="${NEW_VLLM_HASH}_${VLLM_HASH}"
    popd
  fi

  echo "./scripts/scheduler/build_image.sh $VLLM_HASH"
  ./scripts/scheduler/build_image.sh "$VLLM_HASH"

  CODE_HASH=$VLLM_HASH

  # If additional image is needed
  if [ "$REPO" = "TPU_INFERENCE_TORCHAX" ]; then
    echo "build image for TPU_INFERENCE_TORCHAX"

    TPU_INFERENCE_HASH=$(clone_and_get_hash "https://github.com/vllm-project/tpu-inference.git" "artifacts/tpu-inference" "$TPU_INFERENCE_HASH")
    echo "resolved TPU_INFERENCE_HASH: $TPU_INFERENCE_HASH"

    TORCHAX_HASH=$(clone_and_get_hash "https://github.com/pytorch/xla.git" "artifacts/xla" "$TORCHAX_HASH")
    echo "resolved TORCHAX_HASH: $TORCHAX_HASH"

    ./scripts/scheduler/build_tpu_inference_image.sh "$VLLM_HASH" "$TPU_INFERENCE_HASH" "$TORCHAX_HASH"
    CODE_HASH="${VLLM_HASH}-${TPU_INFERENCE_HASH}-${TORCHAX_HASH}"
  elif [ "$REPO" = "TPU_INFERENCE" ]; then
    echo "build image for TPU_INFERENCE only"

    TPU_INFERENCE_HASH=$(clone_and_get_hash "https://github.com/vllm-project/tpu-inference.git" "artifacts/tpu-inference" "$TPU_INFERENCE_HASH")
    echo "resolved TPU_INFERENCE_HASH: $TPU_INFERENCE_HASH"

    ./scripts/scheduler/build_tpu_inference_image.sh "$VLLM_HASH" "$TPU_INFERENCE_HASH" ""
    CODE_HASH="${VLLM_HASH}-${TPU_INFERENCE_HASH}-"

  fi
else
  echo "Skipping build image"
fi

echo "./scripts/scheduler/schedule_run.sh $INPUT_CSV $CODE_HASH $JOB_REFERENCE $RUN_TYPE"
./scripts/scheduler/schedule_run.sh "$INPUT_CSV" "$CODE_HASH" "$JOB_REFERENCE" "$RUN_TYPE" "$EXTRA_ENVS"

echo "Runs created."

echo "========================================================="
echo "To get job status:"
echo "./scripts/manager/get_status.sh $JOB_REFERENCE"
echo
echo "To restart failed job:"
echo "./scripts/manager/reschedule_run.sh $JOB_REFERENCE"
echo "========================================================="
