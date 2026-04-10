#!/bin/bash
set -euo pipefail

# End-to-end TT job creation: builds the base image locally from scratch,
# then builds and pushes the vllm image, then schedules the run.
#
# The base image (DockerfileTT) is built locally only — never pushed — so it
# does not overwrite the shared tt:latest used by other tests.
#
# Argument Explanation: same as create_tt_job.sh
#
# 1. INPUT_CSV       - (Required) Path to the input CSV file to process.
# 2. CODE_HASH       - (Optional) torchtpu-vllm commit hash to pin to.
# 3. JOB_REFERENCE   - (Optional) Identifier for the job or run reference.
# 4. RUN_TYPE        - (Optional) Type of run (e.g., HOURLY, AUTOTUNE). Default: MANUAL.
# 5. EXTRA_ENVS      - (Optional) Semicolon-separated key=value pairs.
# 6. Q_PURPOSE       - (Optional) Queue purpose. Default: bm.
#
# Environment variables:
#   REPO_MAP         - (Optional) Semicolon-separated repo-url||local-path pairs.

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <INPUT_CSV> [CODE_HASH] [JOB_REFERENCE] [RUN_TYPE] [EXTRA_ENVS] [Q_PURPOSE]"
    exit 1
fi

INPUT_CSV="$1"
CODE_HASH="${2:-}"
JOB_REFERENCE="${3:-}"
RUN_TYPE="${4:-"MANUAL"}"
EXTRA_ENVS="${5:-}"
Q_PURPOSE="${6:-"bm"}"

# ==============================================================================
# PARSE THE REPO_MAP ENVIRONMENT VARIABLE (ONCE)
# ==============================================================================
declare -A REPO_MAP_ASSOC

if [[ -n "${REPO_MAP:-}" ]]; then
  echo "Found REPO_MAP environment variable, parsing local repository paths..."
  OLD_IFS="$IFS"
  IFS=';'
  pairs_array=($REPO_MAP)
  IFS="$OLD_IFS"
  for pair in "${pairs_array[@]}"; do
    key="${pair%%||*}"
    value="${pair#*||}"
    REPO_MAP_ASSOC["$key"]="$value"
  done
fi
# ==============================================================================

TT_VLLM_HASH="$CODE_HASH"

echo "Recreating artifacts directory"
rm -rf artifacts/
mkdir -p artifacts/

clone_and_get_hash() {
  local repo_url="$1"
  local dest_folder="$2"
  local target_hash="$3"

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

  if [[ -n "$target_hash" ]]; then
    echo "Resetting to $target_hash" >&2
    git reset --hard "$target_hash" >&2
  fi

  local resolved_hash
  resolved_hash=$(git rev-parse --short HEAD)
  popd > /dev/null

  echo "$resolved_hash"
}

# Clone torchtpu-vllm and resolve hash
TT_VLLM_HASH=$(clone_and_get_hash "https://github.com/google-pytorch/torchtpu-vllm.git" "artifacts/torchtpu-vllm" "$TT_VLLM_HASH")
echo "resolved TT_VLLM_HASH: $TT_VLLM_HASH"

# Read TORCH_VERSION from torchtpu-vllm's pyproject.toml (same approach as ~/setup_tt.sh)
TORCH_VERSION=$(grep -oP '(?<=torch-tpu==)[^"]+' artifacts/torchtpu-vllm/pyproject.toml | tr -d '[:space:]')
if [[ -z "$TORCH_VERSION" ]]; then
    echo "Error: Could not determine torch_tpu version from artifacts/torchtpu-vllm/pyproject.toml"
    exit 1
fi
echo "torch-tpu version from pyproject.toml: ${TORCH_VERSION}"

# Compute tags
VLLM_HASH="95c0f928c"
SAFE_VERSION=$(echo "$TORCH_VERSION" | sed 's/+/--/g')

# Local-only base image tag — never pushed to registry
BASE_IMAGE_TAG="tt-e2e:${SAFE_VERSION}"

FULL_VLLM_IMAGE_BASE="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu"
CODE_HASH="${VLLM_HASH}-${TT_VLLM_HASH}-${SAFE_VERSION}.vllm"
IMAGE_TAG="${FULL_VLLM_IMAGE_BASE}:${CODE_HASH}"

echo "Base image tag (local only): $BASE_IMAGE_TAG"
echo "Final image tag: $IMAGE_TAG"

# ==============================================================================
# SKIP CHECK: if final image already exists remotely, skip all builds
# ==============================================================================
if gcloud artifacts docker tags list "$FULL_VLLM_IMAGE_BASE" \
    --project="$GCP_PROJECT_ID" \
    --format="value(tag)" \
  | grep -Fxq "$CODE_HASH"; then
    echo "Remote image $IMAGE_TAG already exists. Skipping all builds."
    echo "$CODE_HASH" > artifacts/CODE_HASH
else
    pushd artifacts/torchtpu-vllm

    # ==============================================================================
    # PHASE 1: Build base image locally (DockerfileTT) — no push
    # ==============================================================================
    if docker image inspect "$BASE_IMAGE_TAG" &>/dev/null; then
        echo "Local base image $BASE_IMAGE_TAG already exists. Skipping base build."
    else
        echo "Building base image $BASE_IMAGE_TAG locally (no push)..."
        export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)

        VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
        --no-cache \
        --secret id=google_token,env=GOOGLE_ACCESS_TOKEN \
        --build-arg max_jobs=16 \
        --build-arg USE_SCCACHE=1 \
        --build-arg GIT_REPO_CHECK=0 \
        --build-arg TORCH_VERSION="$TORCH_VERSION" \
        --tag "$BASE_IMAGE_TAG" \
        --progress plain \
        -f "../../docker/DockerfileTT" .
    fi

    # ==============================================================================
    # PHASE 2: Build vllm image (DockerfileTTV) using local base, then push
    # ==============================================================================
    if docker image inspect "$IMAGE_TAG" &>/dev/null; then
        echo "Local vllm image $IMAGE_TAG already exists. Pushing..."
        docker push "$IMAGE_TAG"
    else
        echo "Building vllm image $IMAGE_TAG..."
        VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
        --build-arg max_jobs=16 \
        --build-arg USE_SCCACHE=1 \
        --build-arg BASE_IMAGE="$BASE_IMAGE_TAG" \
        --build-arg GIT_REPO_CHECK=0 \
        --tag "$IMAGE_TAG" \
        --progress plain \
        -f "../../docker/DockerfileTTV" .

        docker push "$IMAGE_TAG"
    fi

    popd

    echo "$CODE_HASH" > artifacts/CODE_HASH
fi

# Load CODE_HASH
CODE_HASH_FILE="artifacts/CODE_HASH"
[[ -s "$CODE_HASH_FILE" ]] || { echo "Error: $CODE_HASH_FILE missing or empty" >&2; exit 1; }
CODE_HASH="$(<"$CODE_HASH_FILE")"
echo "CODE_HASH: $CODE_HASH"

echo "./scripts/scheduler/schedule_run.sh $INPUT_CSV $CODE_HASH $JOB_REFERENCE $RUN_TYPE $EXTRA_ENVS $Q_PURPOSE"
./scripts/scheduler/schedule_run.sh "$INPUT_CSV" "$CODE_HASH" "$JOB_REFERENCE" "$RUN_TYPE" "$EXTRA_ENVS" "$Q_PURPOSE"

echo "Runs created."

echo "========================================================="
echo "To get job status:"
echo "./scripts/manager/get_status.sh $JOB_REFERENCE"
echo
echo "To restart failed job:"
echo "./scripts/manager/reschedule_run.sh $JOB_REFERENCE"
echo "========================================================="
