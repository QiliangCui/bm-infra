#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -ex

# Change to the script's directory to ensure relative paths work correctly.
cd "$(dirname "$0")"

# --- Configuration ---
export LOG_DIR=./results
export MODEL_NAME=$MODEL
export TASK_NAME=mlperf
export OUTPUT_PREFIX=${TASK_NAME}_$(echo $MODEL_NAME | sed 's/\//-/g')

export OUTPUT_BASE_PATH=$LOG_DIR/$OUTPUT_PREFIX.json
export ACCURACY_JSON_PATH=/workspace/mlperf_accuracy.json

echo "Running lm_eval, output will be timestamped in: $LOG_DIR"

mkdir -p "$LOG_DIR"

CMD=(
    lm_eval
    --model vllm
    --model_args "pretrained=$MODEL_NAME,tensor_parallel_size=${TP_SIZE:-8},dtype=auto,max_model_len=2048"
    --tasks "$TASK_NAME"
    --include_path .
    --batch_size auto
    --log_samples
    --limit "${NumPrompts:-1000}"
    --output_path "$OUTPUT_BASE_PATH"
    --apply_chat_template
)

# Execute the command, allowing stderr for error visibility
if ! "${CMD[@]}"; then
    echo "Error: lm_eval command failed. See output above for details."
    exit 1
fi

echo "Finding the latest output file in $LOG_DIR with prefix ${OUTPUT_PREFIX}..."

# Find the most recently modified file in the output directory that starts with the correct prefix
LATEST_FILE=$(find "$LOG_DIR" -type f -name "${OUTPUT_PREFIX}_*.json" -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)

# Check if a file was actually found
if [ -z "$LATEST_FILE" ]; then
    echo "Error: No matching output file found. Exiting."
    exit 1
fi

echo "Found and using file: $LATEST_FILE"

echo "Parsing results and writing to $ACCURACY_JSON_PATH..."
python parse_lm_eval_mlperf_results.py "$LATEST_FILE" > "$ACCURACY_JSON_PATH"