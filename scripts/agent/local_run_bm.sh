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
TPU_COMMONS_FOLDER="../tpu_commons"
TPU_COMMONS_REPO="https://github.com/vllm-project/tpu_commons.git"
CONDA="/mnt/disks/persist/bm-agent/miniconda3/bin/conda"

# Load environment
source /etc/environment
set -a
source "$ENV_FILE"
set +a

ENV_NAME="vllm-bm-$CODE_HASH"

# Clone or update vllm repo
if [ ! -d "$VLLM_FOLDER" ] || [ -z "$(ls -A "$VLLM_FOLDER")" ]; then
  echo "Cloning VLLM repo..."
  git clone "$VLLM_REPO" "$VLLM_FOLDER"
fi

IFS='-' read -r VLLM_HASH TPU_COMMON_HASH TORCHAX_HASH _ <<< "$CODE_HASH"

pushd "$VLLM_FOLDER"
git fetch origin
git reset --hard "$VLLM_HASH"
popd

# Check and create conda env
if ! $CONDA env list | grep -Fq "$ENV_NAME"; then
  echo "Creating conda environment '$ENV_NAME'..."
  $CONDA create -y -n "$ENV_NAME" python="$PYTHON_VERSION"
  
  echo "Installing vllm and dependencies..."
  $CONDA run -n "$ENV_NAME" pip install --upgrade pip
  $CONDA run -n "$ENV_NAME" pip install pandas datasets
  $CONDA run -n "$ENV_NAME" bash -c "cd '$VLLM_FOLDER' && VLLM_USE_PRECOMPILED=1 pip install --editable ."

  # Check if TPU_COMMON_HASH is set and not empty
  if [[ -n "$TPU_COMMON_HASH" ]]; then
    echo "TPU_COMMON_HASH is set to '$TPU_COMMON_HASH'. Cloning and installing tpu_commons..."

    # Clone or update tpu_commons repo
    if [ ! -d "$TPU_COMMONS_FOLDER" ]; then
        git clone "$TPU_COMMONS_REPO" "$TPU_COMMONS_FOLDER"
    fi

    echo "Checking out correct tpu_commons commit..."
    pushd "$TPU_COMMONS_FOLDER"
    git fetch origin
    git reset --hard "$TPU_COMMON_HASH"
    popd

    # Install tpu_commons in the new conda environment
    echo "Installing tpu_commons package into '$ENV_NAME'..."
    $CONDA run -n "$ENV_NAME" bash -c "sed -i.bak 's#jax==0.7.1.dev20250813#jax==0.7.2.dev20250821#g' '$TPU_COMMONS_FOLDER/requirements.txt'"
    $CONDA run -n "$ENV_NAME" bash -c "sed -i.bak 's#jaxlib==0.7.1.dev20250813#jaxlib==0.7.2.dev20250821#g' '$TPU_COMMONS_FOLDER/requirements.txt'"
    $CONDA run -n "$ENV_NAME" bash -c "cd '$TPU_COMMONS_FOLDER' && pip install -r requirements.txt && pip install -e ."
    $CONDA run -n "$ENV_NAME" bash -c "cd '$TPU_COMMONS_FOLDER' && pip install -r requirements_benchmarking.txt"
    $CONDA run -n "$ENV_NAME" bash -c "pip install numba"
    $CONDA run -n "$ENV_NAME" bash -c "mkdir -p ../shared-wheels && gsutil cp gs://libtpu-tpu7x-releases/wheels/libtpu/libtpu-0.0.22.dev20250821+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl ../shared-wheels/"
    $CONDA run -n "$ENV_NAME" bash -c "pip install ../shared-wheels/libtpu-0.0.22.dev20250821+tpu7x-cp312-cp312-manylinux_2_31_x86_64.whl"
    echo "tpu_commons installation complete."

    $CONDA run -n "$ENV_NAME" bash -c "gsutil cp gs://amangu-multipods/code/device.py /mnt/disks/persist/bm-agent/miniconda3/envs/$ENV_NAME/lib/python3.12/site-packages/tpu_info/device.py"
    $CONDA run -n "$ENV_NAME" bash -c "gsutil cp gs://amangu-multipods/code/tuned_block_sizes.py $TPU_COMMONS_FOLDER/tpu_commons/kernels/ragged_paged_attention/v3/tuned_block_sizes.py"
    echo "Local v7x changes complete."

  fi
fi

# Safety cleanup on exit
clean_up() { 
   pkill -f VLLM || true
   pkill -f "vllm serve" || true
   ./scripts/agent/clean_old_vllm_envs.sh || true
}
trap clean_up EXIT

# Do a cleanup before starting
clean_up

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

if [ "$DATASET" = "sharegpt" ]; then  
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp gs://$GCS_BUCKET/dataset/sharegpt/*.* ./artifacts/dataset/
  cp -r artifacts/dataset "$TMP_WORKSPACE/"
fi

if [ "$DATASET" = "bench-custom-token" ]; then  
  echo "Copying dataset to container..."
  mkdir -p ./artifacts/dataset/
  gsutil cp -r gs://$GCS_BUCKET/bench-dataset-copy/${MODEL##*/} ./artifacts/dataset/
  cp -r artifacts/dataset "$TMP_WORKSPACE/"
fi

# Run benchmark
echo "Running model benchmark..."
$CONDA run -n "$ENV_NAME" bash -c "
  set -e
  cd '$VLLM_FOLDER'
  WORKSPACE='$TMP_WORKSPACE' \
  HF_TOKEN='$HF_TOKEN' \
  TARGET_COMMIT='$VLLM_HASH' \
  MODEL='$MODEL' \
  ./run_bm.sh
"

# Copy results
VLLM_LOG="$LOG_ROOT/${TEST_NAME}_vllm_log.txt"
BM_LOG="$LOG_ROOT/${TEST_NAME}_bm_log.txt"
cp "$TMP_WORKSPACE/vllm_log.txt" "$VLLM_LOG"
cp "$TMP_WORKSPACE/bm_log.txt" "$BM_LOG"

# Parse throughput
throughput=$(grep 'Request throughput (req/s):' "$BM_LOG" | sed 's/[^0-9.]//g')
echo "Throughput: $throughput"

# Parse Token throughput (tok/s)
output_token_throughput=$(grep 'Output token throughput (tok/s):' "$BM_LOG" | sed 's/[^0-9.]//g')
echo "OutputTokenThroughput: $output_token_throughput"
total_token_throughput=$(grep 'Total Token throughput (tok/s):' "$BM_LOG" | sed 's/[^0-9.]//g')
echo "TotalTokenThroughput: $total_token_throughput"

# Upload to GCS
gsutil cp "$LOG_ROOT"/* "$REMOTE_LOG_ROOT"

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
echo "Throughput=$throughput" > "artifacts/$RECORD_ID.result"
echo "OutputTokenThroughput=$output_token_throughput" >> "artifacts/$RECORD_ID.result"
echo "TotalTokenThroughput=$total_token_throughput" >> "artifacts/$RECORD_ID.result"

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
EOF
