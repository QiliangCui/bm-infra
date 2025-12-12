#!/bin/bash
set -euo pipefail

VLLM_HASH=$1
TPU_INFERENCE_HASH=$2
TORCHAX_HASH=$3
CODE_HASH="${VLLM_HASH}-${TPU_INFERENCE_HASH}-${TORCHAX_HASH}"

IMAGE_TAG="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu:$CODE_HASH"

echo "Image tag: $IMAGE_TAG"

# 1. Check if image exists remotely
if gcloud artifacts docker tags list "$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu" \
    --project="$GCP_PROJECT_ID" \
    --format="value(tag)" \
  | grep -Fxq "$CODE_HASH"; then
    echo "Remote image $IMAGE_TAG already exists. Skipping build and push."
    exit 0
fi

# 2. Check if image exists locally
if docker image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "Local image exists. Skipping build. Pushing..."
    docker push "$IMAGE_TAG"
    exit 0
fi

pushd artifacts/tpu-inference

VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
 --build-arg max_jobs=16 \
 --build-arg USE_SCCACHE=1 \
 --build-arg GIT_REPO_CHECK=0 \
 --build-arg VLLM_COMMIT_HASH=$VLLM_HASH \
 --tag $IMAGE_TAG \
 --progress plain \
 -f docker/Dockerfile .

popd

docker push "$IMAGE_TAG"
