#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -ex

# Change to the script's directory to ensure relative paths work correctly.
cd "$(dirname "$0")"

# --- Configuration ---
export LOG_DIR=./results
export MODEL_NAME=$MODEL
export TASK_NAME=mmlu_llama
export OUTPUT_PREFIX=${TASK_NAME}_$(echo $MODEL_NAME | sed 's#/#-#g')

export OUTPUT_BASE_PATH=$LOG_DIR/$OUTPUT_PREFIX.json
export ACCURACY_JSON_PATH=/workspace/mmlu_tpu_accuracy.json

echo "Running lm_eval for task: $TASK_NAME on TPU"
echo "Output will be timestamped in: $LOG_DIR"

mkdir -p "$LOG_DIR"

MODEL_ARGS="pretrained=$MODEL_NAME"
MODEL_ARGS+=",tensor_parallel_size=${TENSOR_PARALLEL_SIZE:-1}"
MODEL_ARGS+=",dtype=auto"
MODEL_ARGS+=",max_num_seqs=${MAX_NUM_SEQS:-512}"
MODEL_ARGS+=",max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}"
MODEL_ARGS+=",gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9}"
MODEL_ARGS+=",max_model_len=${MAX_MODEL_LEN:-8192}"
MODEL_ARGS+=",enable_prefix_caching=False"
MODEL_ARGS+=",download_dir=${DOWNLOAD_DIR:-/mnt/disks/persist}"

if [[ "${ENABLE_EXPERT_PARALLEL:-False}" == "True" ]]; then
    MODEL_ARGS+=",enable_expert_parallel=True"
fi

CMD=(
    lm_eval
    --model vllm
    --model_args "$MODEL_ARGS"
    --tasks "$TASK_NAME"
    --num_fewshot 0
    --apply_chat_template
    --batch_size auto
    --output_path "$OUTPUT_BASE_PATH"
    --limit 100
)

if ! "${CMD[@]}"; then
    echo "Error: lm_eval command failed. See output above for details."
    exit 1
fi

echo "Finding the latest output file in $LOG_DIR with prefix ${OUTPUT_PREFIX}..."

LATEST_FILE=$(find "$LOG_DIR" -type f -name "${OUTPUT_PREFIX}_*.json" -printf "%T@ %p\n" | sort -nr | head -n 1 | cut -d' ' -f2-)

if [ -z "$LATEST_FILE" ]; then
    echo "Error: No matching output file found. Exiting."
    exit 1
fi

echo "Found and using file: $LATEST_FILE"

echo "Parsing results and writing to $ACCURACY_JSON_PATH..."
python ../mmlu/parse_lm_eval_mmlu_results.py "$LATEST_FILE" > "$ACCURACY_JSON_PATH"
