#!/bin/bash
set -euo pipefail

# Check at least 1 argument is provided
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <INPUT_CSV> [CODE_HASH] [JOB_REFERENCE] [RUN_TYPE] [REPO]"
    exit 1
fi

INPUT_CSV="$1"
CODE_HASH="${2:-}"  # optional
JOB_REFERENCE="${3:-}"
RUN_TYPE="${4:-"MANUAL"}"
REPO="${5:-"DEFAULT"}"
TPU_COMMONS_TPU_BACKEND_TYPE="${6:-"torchax"}"

if [[ "$REPO" != "DEFAULT" && "$REPO" != "TPU_COMMONS" ]]; then
  echo "Error: REPO must be either DEFAULT or TPU_COMMONS, but got '$REPO'"
  exit 1
fi

if [[ "$TPU_COMMONS_TPU_BACKEND_TYPE" != "torchax" && "$TPU_COMMONS_TPU_BACKEND_TYPE" != "jax" ]]; then
  echo "Error: TPU_COMMONS_TPU_BACKEND_TYPE must be either torchax or jax, but got '$TPU_COMMONS_TPU_BACKEND_TYPE'"
  exit 1
fi

IFS='-' read -r VLLM_HASH TPU_COMMONS_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

echo "Recreating artifacts directory"
rm -rf artifacts/
mkdir -p artifacts/

clone_and_get_hash() {
  local repo_url="$1"
  local dest_folder="$2"
  local target_hash="$3"

  # Clone the repo
  git clone "$repo_url" "$dest_folder"
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

  echo "./scripts/scheduler/build_image.sh $VLLM_HASH"
  ./scripts/scheduler/build_image.sh "$VLLM_HASH"

  CODE_HASH=$VLLM_HASH

  # If additional image is needed
  if [ "$REPO" = "TPU_COMMONS" ]; then
    echo "build image for TPU_COMMON"

    TPU_COMMONS_HASH=$(clone_and_get_hash "https://github.com/vllm-project/tpu_commons.git" "artifacts/tpu_commons" "$TPU_COMMONS_HASH")
    echo "resolved TPU_COMMONS_HASH: $TPU_COMMONS_HASH"

    TORCHAX_HASH=$(clone_and_get_hash "https://github.com/pytorch/xla.git" "artifacts/xla" "$TORCHAX_HASH")
    echo "resolved TORCHAX_HASH: $TORCHAX_HASH"

    ./scripts/scheduler/build_tpu_commons_image.sh "$VLLM_HASH" "$TPU_COMMONS_HASH" "$TORCHAX_HASH" "$TPU_COMMONS_TPU_BACKEND_TYPE"
    CODE_HASH="${VLLM_HASH}-${TPU_COMMONS_HASH}-${TORCHAX_HASH}-${TPU_COMMONS_TPU_BACKEND_TYPE}"
  fi

else
  echo "Skipping build image"
fi

echo "./scripts/scheduler/schedule_run.sh $INPUT_CSV $CODE_HASH $JOB_REFERENCE $RUN_TYPE"
./scripts/scheduler/schedule_run.sh "$INPUT_CSV" "$CODE_HASH" "$JOB_REFERENCE" "$RUN_TYPE"

echo "Runs created."

echo "========================================================="
echo "To get job status:"
echo "./scripts/manager/get_status.sh $JOB_REFERENCE"
echo
echo "To restart failed job:"
echo "./scripts/manager/reschedule_run.sh $JOB_REFERENCE"
echo "========================================================="
