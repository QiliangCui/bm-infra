#!/bin/bash
set -euo pipefail

DOCKERFILE="../../docker/DockerfileTT"
echo "Building with $DOCKERFILE"

pushd artifacts/torchtpu-vllm

export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)

# 1. Get the Access Token
export GOOGLE_ACCESS_TOKEN=$(gcloud auth print-access-token)

# 2. Fetch the version of torch_tpu
# This runs a tiny container to check the version from the registry
echo "Determining torch_tpu version..."
TORCH_VERSION=$(docker run --rm \
    -e GOOGLE_ACCESS_TOKEN=$GOOGLE_ACCESS_TOKEN \
    python:3.12-slim-bookworm /bin/bash -c "
    pip install --quiet --no-deps \
    --index-url https://oauth2accesstoken:\${GOOGLE_ACCESS_TOKEN}@us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ \
    --pre torch_tpu > /dev/null 2>&1 && \
    pip show torch_tpu | grep Version | awk '{print \$2}'
")

if [ -z "$TORCH_VERSION" ]; then
    echo "Error: Could not determine torch_tpu version."
    exit 1
fi

SAFE_VERSION=$(echo "$TORCH_VERSION" | sed 's/+/--/g')

IMAGE_BASE="$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/tt"
IMAGE_VERSION_TAG="$IMAGE_BASE:$SAFE_VERSION"
IMAGE_LATEST_TAG="$IMAGE_BASE:latest"

echo "Image tag: $IMAGE_VERSION_TAG"
echo "Image tag: $IMAGE_LATEST_TAG"

VLLM_TARGET_DEVICE=tpu DOCKER_BUILDKIT=1 docker build \
--secret id=google_token,env=GOOGLE_ACCESS_TOKEN \
--build-arg max_jobs=16 \
--build-arg USE_SCCACHE=1 \
--build-arg GIT_REPO_CHECK=0 \
--tag $IMAGE_VERSION_TAG \
--tag $IMAGE_LATEST_TAG \
--progress plain \
-f "$DOCKERFILE" .

popd

# docker push "$IMAGE_TAG"
