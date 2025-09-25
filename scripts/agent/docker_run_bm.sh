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

IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

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


# Check and trim
# Example value
# TARGET_COMMIT="bb81182d3_de509ae8e"
# TARGET_COMMIT="bb81182d3"
TARGET_COMMIT=$VLLM_HASH
if [[ "$TARGET_COMMIT" == *_* ]]; then
  TARGET_COMMIT="${TARGET_COMMIT%%_*}"
fi

echo "Run model $MODEL"
echo

echo "starting docker...$CONTAINER_NAME"
echo
docker run \
 -v $DOWNLOAD_DIR:$DOWNLOAD_DIR \
 --env-file $ENV_FILE \
 -e HF_TOKEN="$HF_TOKEN" \
 -e TARGET_COMMIT=$TARGET_COMMIT \
 -e MODEL=$MODEL \
 -e DATASET=$DATASET \
 -e WORKSPACE=/workspace \
 --name $CONTAINER_NAME \
 -d \
 --privileged \
 --network host \
 -v /dev/shm:/dev/shm \
 $image_tag tail -f /dev/null

# =============== temp solution start ===============

DATASETS=("custom-token" "mmlu" "mlperf" "bench-custom-token" "math500")
if [[ " ${DATASETS[*]} " == *" $DATASET "* ]]; then
  echo "Temp solution: Syncing dataset for $DATASET"

  mkdir -p ./artifacts/dataset/

  if [ "$DATASET" = "custom-token" ]; then
    # Download flat files for custom-token
    gsutil -m cp gs://$GCS_BUCKET/dataset/*.* ./artifacts/dataset/
  elif [ "$DATASET" = "mmlu" ]; then
    # Download mmlu directory recursively
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/mmlu/* ./artifacts/dataset/
  elif [ "$DATASET" = "mlperf" ]; then
    # Download single pkl file for MLPerf
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/mlperf/processed-data.pkl ./artifacts/dataset/
  elif [ "$DATASET" = "bench-custom-token" ]; then
    # Download flat files for custom-token
    gsutil -m cp -r gs://$GCS_BUCKET/bench-dataset/* ./artifacts/dataset/
  elif [ "$DATASET" = "math500" ]; then
    gsutil -m cp -r gs://$GCS_BUCKET/dataset/math500.jsonl ./artifacts/dataset/
  fi

  echo "Copying dataset to container..."
  docker cp artifacts/dataset "$CONTAINER_NAME:/workspace/"

  echo docker cp scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"
  docker cp scripts/agent/benchmark_serving.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_serving.py"

  echo docker cp scripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
  docker cp scripts/agent/benchmark_dataset.py "$CONTAINER_NAME:/workspace/vllm/benchmarks/benchmark_dataset.py"
fi


# =============== temp solution end ===============

# TODO(patemotter): split these into functions
if [ "$DATASET" = "sharegpt" ]; then
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp gs://$GCS_BUCKET/dataset/sharegpt/*.* ./artifacts/dataset/
  docker cp artifacts/dataset "$CONTAINER_NAME:/workspace/"
fi

if [[ "$DATASET" == "math500" ]]; then
  echo "Copying lm_eval directory to container..."
  docker cp lm_eval "$CONTAINER_NAME:/workspace/"
fi

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
PROFILE_FOLDER="$LOG_ROOT/$TEST_NAME"_profile
docker cp "$CONTAINER_NAME:/workspace/vllm_log.txt" "$VLLM_LOG"
docker cp "$CONTAINER_NAME:/workspace/bm_log.txt" "$BM_LOG"
docker cp "$CONTAINER_NAME:/workspace/profile/plugins/profile" "$PROFILE_FOLDER"

echo "gsutil cp $LOG_ROOT/* $REMOTE_LOG_ROOT"
gsutil cp -r $LOG_ROOT/* $REMOTE_LOG_ROOT

AccuracyMetricsJSON=$(grep "AccuracyMetrics:" "$BM_LOG" | sed 's/AccuracyMetrics: //')
echo "AccuracyMetricsJSON: $AccuracyMetricsJSON"

if [[ "$DATASET" == "math500" ]]; then
  # For lm_eval runs, we should focus on the accuracy results only
  echo "Accuracy-only benchmark, skipping performance metrics."
  echo "AccuracyMetrics=$AccuracyMetricsJSON" > "artifacts/$RECORD_ID.result"
else
  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  echo "throughput for $TEST_NAME at $VLLM_HASH: $throughput"

  output_token_throughput=$(grep "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  total_token_throughput=$(grep "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')

  if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    if [[ -n "$AccuracyMetricsJSON" ]]; then
      echo "Failed to get the throughput, but found accuracy metrics."
      echo "AccuracyMetrics=$AccuracyMetricsJSON" > "artifacts/$RECORD_ID.result"
      exit 0
    else
      echo "Failed to get the throughput and no accuracy metrics found."
      exit 1
    fi
  fi

  if (( $(echo "$throughput < $EXPECTED_THROUGHPUT" | bc -l) )); then
    echo "Error: throughput($throughput) is less than expected($EXPECTED_THROUGHPUT)"
  fi
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

  # Write results to file
  (
    printf '%s=%s\n' \
      "MedianITL" "$MedianITL" \
      "MedianTPOT" "$MedianTPOT" \
      "MedianTTFT" "$MedianTTFT" \
      "MedianETEL" "$MedianETEL" \
      "P99ITL" "$P99ITL" \
      "P99TPOT" "$P99TPOT" \
      "P99TTFT" "$P99TTFT" \
      "P99ETEL" "$P99ETEL" \
      "OutputTokenThroughput" "$output_token_throughput" \
      "TotalTokenThroughput" "$total_token_throughput" \
      "AccuracyMetrics" "$AccuracyMetricsJSON"
  ) >> "artifacts/$RECORD_ID.result"
fi

