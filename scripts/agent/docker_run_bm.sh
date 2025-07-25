#!/bin/bash

if [ ! -f "$1" ]; then
  echo "Error: The env file '$1' does not exist."
  exit 1  # Exit the script with a non-zero status to indicate an error
fi

ENV_FILE=$1

# For testing on local vm, use `set -a` to export all variables
source /etc/environment
source $ENV_FILE

remove_docker_container() { 
    docker rm -f tpu-test || true; 
    docker rm -f vllm-tpu || true;
    docker rm -f $CONTAINER_NAME || true;
}

trap remove_docker_container EXIT

# Remove the container that might not be cleaned up in the previous run.
remove_docker_container

image_tag=$GCP_REGION-docker.pkg.dev/$GCP_PROJECT_ID/vllm-tpu-bm/vllm-tpu:$CODE_HASH

IFS='-' read -r VLLM_HASH TPU_COMMON_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

echo "image tag: $image_tag"

docker pull $image_tag

if [ $? -ne 0 ]; then
  echo "Failed to pull the Docker image: $image_tag"
  exit 1
fi

LOG_ROOT=$(mktemp -d)
REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"

# If mktemp fails, set -e will cause the script to exit.
echo "Results will be stored in: $LOG_ROOT"

if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN is not set or is empty."  
  exit 1
fi

# Make sure mounted disk or dir exists
if [ ! -d "$DOWNLOAD_DIR" ]; then
    echo "Error: Folder $DOWNLOAD_DIR does not exist. This is useually a mounted drive. If no mounted drive, just create a folder."
    exit 1
fi

if ! mountpoint -q "$DOWNLOAD_DIR"; then
    echo "Error: $DOWNLOAD_DIR exists but is not a mounted directory."
    exit 1
fi

echo "Run model $MODEL"
echo

echo "starting docker...$CONTAINER_NAME"
echo    
docker run \
 -v $DOWNLOAD_DIR:$DOWNLOAD_DIR \
 --env-file $ENV_FILE \
 -e HF_TOKEN="$HF_TOKEN" \
 -e TARGET_COMMIT=$VLLM_HASH \
 -e MODEL=$MODEL \
 -e DATASET=$DATASET \
 -e WORKSPACE=/workspace \
 --name $CONTAINER_NAME \
 -d \
 --privileged \
 --network host \
 -v /dev/shm:/dev/shm \
 $image_tag tail -f /dev/null

# =============== temp solution ===============
if [ "$DATASET" = "custom-token" ] || [ "$DATASET" = "mmlu" ]; then
  echo "Temp solution: Syncing dataset for $DATASET"

  mkdir -p ./artifacts/dataset/

  if [ "$DATASET" = "custom-token" ]; then
    # Download flat files for custom-token
    gsutil -m cp gs://$GCS_BUCKET/dataset/*.* ./artifacts/dataset/
  elif [ "$DATASET" = "mmlu" ]; then
    # Download mmlu directory recursively
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/mmlu/* ./artifacts/dataset/
  fi

  echo "Copying dataset to container..."
  docker cp artifacts/dataset "$CONTAINER_NAME:/workspace/"

  echo docker cp scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"
  docker cp scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"

  echo docker cp scripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
  docker cp scripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
fi

# ===============  temp solution ===============

echo "copy script run_bm.sh to container..."
docker cp scripts/agent/run_bm.sh "$CONTAINER_NAME:/workspace/vllm/run_bm.sh"

echo "grant chmod +x"
echo
docker exec "$CONTAINER_NAME" chmod +x "/workspace/vllm/run_bm.sh"

echo "run script..."
echo
docker exec "$CONTAINER_NAME" /bin/bash -c "echo always > /sys/kernel/mm/transparent_hugepage/enabled && ./run_bm.sh"

echo "copy results and logs back..."
VLLM_LOG="$LOG_ROOT/$TEST_NAME"_vllm_log.txt
BM_LOG="$LOG_ROOT/$TEST_NAME"_bm_log.txt
docker cp "$CONTAINER_NAME:/workspace/vllm_log.txt" "$VLLM_LOG" 
docker cp "$CONTAINER_NAME:/workspace/bm_log.txt" "$BM_LOG"

echo "gsutil cp $LOG_ROOT/* $REMOTE_LOG_ROOT"
gsutil cp $LOG_ROOT/* $REMOTE_LOG_ROOT

throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
echo "throughput for $TEST_NAME at $VLLM_HASH: $throughput"

output_token_throughput=$(grep "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
total_token_throughput=$(grep "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
# Extract the JSON string for accuracy metrics. The sed command removes the 'AccuracyMetrics: ' prefix.
AccuracyMetricsJSON=$(grep "AccuracyMetrics:" "$BM_LOG" | sed 's/AccuracyMetrics: //')

#
# compare the throughput with EXPECTED_THROUGHPUT 
# and assert meeting the expectation
# Even if failed to get throughput, we still consider the docker run is good.
# The following script will report failure if the result out is not created.
# 
if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Failed to get the throughput"
  exit 1
fi

if (( $(echo "$throughput < $EXPECTED_THROUGHPUT" | bc -l) )); then
  echo "Error: throughput($throughput) is less than expected($EXPECTED_THROUGHPUT)"
  exit 0
fi

# write output
echo "Throughput=$throughput" > "artifacts/$RECORD_ID.result"

extract_value() {
  local section="$1"
  local label="$2"  # Mean, Median, or P99
  grep "$section (ms):" "$BM_LOG" | \
    awk -v label="$label" '$0 ~ label { print $NF }'
}

# Median values
MedianITL=$(extract_value "ITL" "Median")
MedianTPOT=$(extract_value "TPOT" "Median")
MedianTTFT=$(extract_value "TTFT" "Median")
MedianETEL=$(extract_value "E2EL" "Median")

# P99 values
P99ITL=$(extract_value "ITL" "P99")
P99TPOT=$(extract_value "TPOT" "P99")
P99TTFT=$(extract_value "TTFT" "P99")
P99ETEL=$(extract_value "E2EL" "P99")

cat <<EOF >> "artifacts/$RECORD_ID.result"
MedianITL=$MedianITL
MedianTPOT=$MedianTPOT
MedianTTFT=$MedianTTFT
MedianETEL=$MedianETEL
P99ITL=$P99ITL
P99TPOT=$P99TPOT
P99TTFT=$P99TTFT
P99ETEL=$P99ETEL
OutputTokenThroughput=$output_token_throughput
TotalTokenThroughput=$total_token_throughput
AccuracyMetrics='$AccuracyMetricsJSON'
EOF
