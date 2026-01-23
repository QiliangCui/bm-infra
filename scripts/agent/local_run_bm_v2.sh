#!/bin/bash
set -euo pipefail

if [ ! -f "$1" ]; then
  echo "Error: The env file '$1' does not exist."
  exit 1
fi

ENV_FILE=$1
PYTHON_VERSION="3.12"
VLLM_FOLDER="../vllm"
VLLM_REPO="https://github.com/vllm-project/vllm"

# Load environment variables
source /etc/environment
set -a
source "$ENV_FILE"
set +a

# setup uv
export PATH="/home/bm-agent/.local/bin:$PATH"

ENV_NAME="vllm-bm-$CODE_HASH"

# setup the uv root dire
mkdir -p /mnt/disks/persist/bm-agent/uv
ENV_BASE="/mnt/disks/persist/bm-agent/uv"

ENV_PATH="$ENV_BASE/$ENV_NAME"

# Clone or update vllm repo
if [ ! -d "$VLLM_FOLDER" ] || [ -z "$(ls -A "$VLLM_FOLDER")" ]; then
  echo "Cloning VLLM repo..."
  git clone "$VLLM_REPO" "$VLLM_FOLDER"
fi

IFS='-' read -r VLLM_HASH TPU_INFERENCE_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

pushd "$VLLM_FOLDER"
git fetch origin
git reset --hard "$VLLM_HASH"
popd

# Check and create uv venv
if [ ! -d "$ENV_PATH" ]; then
  echo "Creating uv environment at $ENV_PATH..."
  uv venv "$ENV_PATH" --python "python$PYTHON_VERSION"
  uv pip install -p "$ENV_PATH/bin/python" --upgrade pip
  uv pip install -p "$ENV_PATH/bin/python" pandas datasets

  # Install lm_eval with dependencies, version is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
  uv pip install -p "$ENV_PATH/bin/python" "lm-eval[api,vllm,math]>=0.4.9.2"

  echo "Installing vllm and dependencies..."
  echo VLLM_USE_PRECOMPILED=1 uv pip install -p "$ENV_PATH/bin/python" -e "$VLLM_FOLDER" --torch-backend=cu128
  VLLM_USE_PRECOMPILED=1 uv pip install -p "$ENV_PATH/bin/python" -e "$VLLM_FOLDER" --torch-backend=cu128
fi

# Safety cleanup on exit
clean_up() {
   pkill -f vllm || true
   pkill -f VLLM || true
   ./scripts/agent/clean_old_vllm_envs_v2.sh || true
}
trap clean_up EXIT

# Prepare working dirs
TMP_WORKSPACE="/tmp/workspace"
LOG_ROOT=$(mktemp -d)
REMOTE_LOG_ROOT="gs://$GCS_BUCKET/job_logs/$RECORD_ID/"

rm -rf "$TMP_WORKSPACE"
mkdir -p "$TMP_WORKSPACE"

echo "Results will be stored in: $LOG_ROOT"

# Sanity checks
if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN is not set."
  exit 1
fi

if [ ! -d "$DOWNLOAD_DIR" ]; then
  echo "Error: Folder $DOWNLOAD_DIR does not exist."
  exit 1
fi

if ! mountpoint -q "$DOWNLOAD_DIR"; then
    echo "Error: $DOWNLOAD_DIR exists but is not a mounted directory."
    exit 1
fi

# Prepare script
echo "Copying and chmod-ing run_bm.sh..."
cp scripts/agent/run_bm.sh "$VLLM_FOLDER/run_bm.sh"
chmod +x "$VLLM_FOLDER/run_bm.sh"

echo "Copying bench_serving directory..."
mkdir -p "$VLLM_FOLDER/scripts/agent"
cp -r scripts/agent/bench_serving "$VLLM_FOLDER/scripts/agent/"

# prepare datasets
if [ "$DATASET" = "sharegpt" ]; then
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp gs://$GCS_BUCKET/dataset/sharegpt/*.* ./artifacts/dataset/
  cp -r artifacts/dataset "$TMP_WORKSPACE/"
fi

# Run benchmark
echo "Running model benchmark..."
bash -c "
  set -e
  source '$ENV_PATH/bin/activate'
  cd '$VLLM_FOLDER'
  WORKSPACE='$TMP_WORKSPACE' \
  HF_TOKEN='$HF_TOKEN' \
  TARGET_COMMIT='$VLLM_HASH' \
  MODEL='$MODEL' \
  ./run_bm.sh
" || true # let it got through code below to upload oogs

# Copy results
VLLM_LOG="$LOG_ROOT/${TEST_NAME}_vllm_log.txt"
BM_LOG="$LOG_ROOT/${TEST_NAME}_bm_log.txt"
cp "$TMP_WORKSPACE/vllm_log.txt" "$VLLM_LOG" || true
cp "$TMP_WORKSPACE/bm_log.txt" "$BM_LOG" || true

# Upload to GCS
gsutil cp "$LOG_ROOT"/* "$REMOTE_LOG_ROOT"

# Parse throughput
throughput=$(grep 'Request throughput (req/s):' "$BM_LOG" | sed 's/[^0-9.]//g') || true
echo "Throughput: $throughput" || true

# Check throughput
if [[ -z "$throughput" || ! "$throughput" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Failed to parse throughput"
  exit 0
fi

if [[ -n "${EXPECTED_THROUGHPUT:-}" ]]; then
  if (( $(echo "$throughput < $EXPECTED_THROUGHPUT" | bc -l) )); then
    echo "Error: Throughput ($throughput) < Expected ($EXPECTED_THROUGHPUT)"
    exit 0
  fi
else
  echo "No EXPECTED_THROUGHPUT set, skipping threshold check."
fi

# Write result file
mkdir -p artifacts
echo "Throughput=$throughput" > "artifacts/$RECORD_ID.result"

output_token_throughput=$(grep "Output token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
total_token_throughput=$(grep "Total Token throughput (tok/s):" "$BM_LOG" | sed 's/[^0-9.]//g')

extract_value() {
  local section="$1"
  local label="$2"
  grep "$section (ms):" "$BM_LOG" | awk -v label="$label" '$0 ~ label { print $NF }'
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
EOF
