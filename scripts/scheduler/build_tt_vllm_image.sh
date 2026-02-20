#!/bin/bash
set -euo pipefail

TT_VLLM_HASH=$1

# hard code to v0.13.0 for nwo
VLLM_HASH="72506c983"
echo "vllm hash $VLLM_HASH"

BASE_IMAGE=southamerica-west1-docker.pkg.dev/cloud-tpu-inference-test/vllm-tpu-bm/tt:latest

# Extract torch_tpu version

echo "Extracting torch_tpu version from $BASE_IMAGE..."
docker pull "$BASE_IMAGE" > /dev/null
TORCH_TPU_VERSION=$(docker run --rm "$BASE_IMAGE" pip show torch_tpu | grep Version | awk '{print $2}')

if [ -z "$TORCH_TPU_VERSION" ]; then
    echo "Error: Could not find torch_tpu version in $BASE_IMAGE"
    exit 1
fi

echo "torch_tpu version: $TORCH_TPU_VERSION"
SAFE_TORCH_TPU_VERSION=$(echo "$TORCH_VERSION" | sed 's/+/--/g')

CODE_HASH="${VLLM_HASH}-${TT_VLLM_HASH}-${SAFE_TORCH_TPU_VERSION}"
IMAGE_TAG="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu:$CODE_HASH"
echo "Image tag: $IMAGE_TAG"

# 1. Check if image exists remotely
# if gcloud artifacts docker tags list "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu" \
#     --project="$GCP_PROJECT_ID" \
#     --format="value(tag)" \
#   | grep -Fxq "$CODE_HASH"; then
#     echo "Remote image $IMAGE_TAG already exists. Skipping build and push."
#     exit 0
# fi

# 2. Check if image exists locally
if docker image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "Local image exists. Skipping build. Pushing..."
    docker push "$IMAGE_TAG"
    exit 0
fi

DOCKERFILE="../../docker/DockerfileTTV"
echo "Building with $DOCKERFILE"

pushd artifacts/torchtpu-vllm

VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
--build-arg max_jobs=16 \
--build-arg USE_SCCACHE=1 \
--build-arg BASE_IMAGE=$BASE_IMAGE \
--build-arg GIT_REPO_CHECK=0 \
--build-arg VLLM_COMMIT_HASH="$VLLM_HASH" \
--tag $IMAGE_TAG \
--progress plain \
-f "$DOCKERFILE" .

popd

docker push "$IMAGE_TAG"
