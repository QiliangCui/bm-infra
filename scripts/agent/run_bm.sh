#!/bin/bash

set -euo pipefail

# Datasets using lm-evaluation-harness `lm_eval`.
LM_EVAL_DATASETS=("math500" "mmlu" "mlperf")

# Datasets that use the internal python performance benchmark script `python benchmark_serving.py`.
BM_INFRA_DATASETS=("custom-token" "bench-custom-token")

# All other datasets will use the standard `vllm bench serve` command.

# TODO: Move to image building.
# Ingore the error because in case of using uv, the packages are installed outside this script.
pip install pandas || true
pip install datasets || true
pip install evaluate==0.4.5 || true
pip install rouge-score==0.1.2 || true
# Install lm_eval with math dependencies, commit is same as https://github.com/vllm-project/vllm/blob/main/.buildkite/scripts/hardware_ci/run-tpu-v1-test.sh#L64
pip install "lm-eval[math] @ git+https://github.com/EleutherAI/lm-evaluation-harness.git@206b7722158f58c35b7ffcd53b035fdbdda5126d" || true


VLLM_LOG="$WORKSPACE/vllm_log.txt"
BM_LOG="$WORKSPACE/bm_log.txt"
BEST_BM_LOG="$WORKSPACE/best_bm_log.txt"
PROFILE_FOLDER="$WORKSPACE/profile"


if [ -n "$TARGET_COMMIT" ]; then
  head_hash=$(git rev-parse HEAD)
  resolved_target=$(git rev-parse "$TARGET_COMMIT" 2>/dev/null)

  if [ -z "$resolved_target" ]; then
    echo "Error: target commit '$TARGET_COMMIT' is not a valid Git object" | tee -a $VLLM_LOG
    exit 1
  fi

  if [ "$resolved_target" != "$head_hash" ]; then
    echo "Error: target commit '$TARGET_COMMIT' does not match HEAD: $head_hash" | tee -a $VLLM_LOG
    exit 1
  fi
fi

echo "model: $MODEL"
echo

# Helper function to check if a value is in an array
contains_element () {
  local e match="$1"
  shift
  for e; do [[ "$e" == "$match" ]] && return 0; done
  return 1
}

# Run accuracy benchmark via lm_eval
if contains_element "$DATASET" "${LM_EVAL_DATASETS[@]}"; then
  echo "DATASET ($DATASET) is an accuracy benchmark. Running lm_eval path."
  /workspace/lm_eval/$DATASET/run.sh
  printf "AccuracyMetrics: " > /workspace/bm_log.txt
  cat "/workspace/${DATASET}_accuracy.json" | tr -d '\n' >> /workspace/bm_log.txt
  echo "" >> /workspace/bm_log.txt
  echo "Finished running $DATASET benchmark."
  exit 0
fi


# create a log and profile folder
mkdir -p "$WORKSPACE/log"
mkdir -p "$PROFILE_FOLDER"

if [ "$DATASET" = "sonnet" ]; then
  echo "Create sonnet_4x.txt"
  echo "" > benchmarks/sonnet_4x.txt
  for _ in {1..4}
    do
     cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
  done
fi

#
# start vllm service in backend
#
echo "lanching vllm..."
echo "logging to $VLLM_LOG"
echo

if [[ -z "${EXTRA_ARGS:-}" ]]; then
  # If it is unset or empty, we initialize it as an empty string.
  # This makes the append operation (+=) safe to use later.
  EXTRA_ARGS=""
fi

if [[ -n "${ADDITIONAL_CONFIG:-}" ]]; then
  echo "Adding --additional_config=${ADDITIONAL_CONFIG} to EXTRA_ARGS for running vllm serve ..."
  EXTRA_ARGS+=" --additional_config='${ADDITIONAL_CONFIG}'"
fi

if [[ "$MODEL" == "google/gemma-3-27b-it" ]]; then
  echo "google/gemma-3-27b-it"
  EXTRA_ARGS+=" --limit-mm-per-prompt {\"image\":0}"
fi

if [[ "$MODEL" == "deepseek-ai/DeepSeek-R1" ]]; then
  echo "deepseek-ai/DeepSeek-R1"
  EXTRA_ARGS+=" --hf-config=deepseek-ai/DeepSeek-R1 --hf_overrides '{\"architectures\": [\"DeepseekV3ForCausalLM\"]}' --gpu-memory-utilization 0.91"
fi

# TODO: Remove this fragile string matching way of passing extra flags. This is done in despirite times.
# Implement EXTRA_FLAGS support, which can be passed dynamically from the csv.
if [[ "$MODEL" == *"Qwen/Qwen3"* && "${ADDITIONAL_CONFIG:-}" == *"float8"* ]]; then
  echo "$MODEL with float8 config detected."
  EXTRA_ARGS+=" --kv-cache-dtype=fp8 --gpu-memory-utilization=0.98"
fi

if [[ "$MODEL" == *"unsloth/gpt-oss-120b-BF16"* ]]; then
  echo "$MODEL model detected."
  EXTRA_ARGS+=" --gpu-memory-utilization=0.98"
fi

echo "Printing the vllm serve command used to start the server:"
echo "VLLM_USE_V1=1 VLLM_TORCH_PROFILER_DIR=\"$PROFILE_FOLDER\" vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS \
 --async-scheduling > \"$VLLM_LOG\" 2>&1 &"

eval "VLLM_USE_V1=1 VLLM_TORCH_PROFILER_DIR=\"$PROFILE_FOLDER\" vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS \
 --async-scheduling > \"$VLLM_LOG\" 2>&1 &"


echo "wait for 20 minutes.."
echo
for i in {1..120}; do
    # TODO: detect other type of errors.
    if grep -Fq "raise RuntimeError" "$VLLM_LOG"; then
        echo "Detected RuntimeError, exiting."
        exit 1
    elif grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

EXPECTED_ETEL=${EXPECTED_ETEL:-3600000}
NUM_PROMPTS=${NUM_PROMPTS:-1000}
PREFIX_LEN=${PREFIX_LEN:-0}

PROFILE_FLAG=""
# Check if the PROFILE variable is numerically equal to 1
if [[ "${PROFILE:-0}" -eq 1 ]]; then
  PROFILE_FLAG="--profile"
fi

run_benchmark(){
  echo "running benchmark..."
  echo "logging to $BM_LOG"
  echo

  local request_rate="$1"
  local command_to_run
  local ARGS=()

  # Determine benchmark command to use
  if contains_element "$DATASET" "${BM_INFRA_DATASETS[@]}"; then
    command_to_run=("python" "benchmarks/benchmark_serving.py")
  else
    command_to_run=("vllm" "bench" "serve")
  fi

  # TODO: Remove this hardcoding before merging to mains.
  command_to_run=("vllm" "bench" "serve")

  # Common arguments
  ARGS+=(
    --backend vllm
    --model "$MODEL"
    --request-rate "$request_rate"
    --dataset-name "$DATASET"
    --num-prompts "$NUM_PROMPTS"
    --percentile-metrics "ttft,tpot,itl,e2el"
    --ignore-eos
    $PROFILE_FLAG
  )

  # Dataset-specific arguments
  case "$DATASET" in
    sonnet)
      ARGS+=(--dataset-path "benchmarks/sonnet_4x.txt" --sonnet-input-len "$INPUT_LEN" --sonnet-output-len "$OUTPUT_LEN")
      ;;
    random)
      ARGS+=(--random-input-len "$INPUT_LEN" --random-output-len "$OUTPUT_LEN")
      ;;
    mmlu)
      ARGS+=(--dataset-path "/workspace/dataset" --mmlu-num-shots 0 --mmlu-method "HELM")
      ;;
    mlperf)
      ARGS+=(--dataset-path "/workspace/dataset/processed-data.pkl" --mlperf-input-len "$INPUT_LEN" --max-model-len "$MAX_MODEL_LEN")
      ;;
    custom-token)
      local dataset_path="$WORKSPACE/dataset/${MODEL##*/}_${INPUT_LEN}_${OUTPUT_LEN}_tp${TENSOR_PARALLEL_SIZE}.json"
      ARGS+=(--dataset-path "$dataset_path")
      ;;
    bench-custom-token)
      local dataset_path="$WORKSPACE/dataset/${MODEL##*/}/inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}.jsonl"
      echo "dataset_path: $dataset_path"
      # The original script set dataset-name to 'custom' for this case
      ARGS[7]="custom" # This replaces the --dataset-name value in the array
      ARGS+=(--dataset-path "$dataset_path" --custom-output-len "$OUTPUT_LEN" --skip-chat-template)
      ;;
    sharegpt)
      local dataset_path="$WORKSPACE/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"
      if [ "$INPUT_LEN" -gt 0 ]; then
        echo "Please set INPUT_LEN to 0 for sharegpt dataset because it is not used." > "$BM_LOG" 2>&1
        exit 1
      fi
      ARGS+=(--dataset-path "$dataset_path")
      if [ "$OUTPUT_LEN" -ne 0 ]; then
        ARGS+=(--sharegpt-output-len "$OUTPUT_LEN")
      fi
      ;;
    hf)
      # Override backend for this specific case
      ARGS[1]="openai-chat" # Replaces --backend value
      ARGS+=(--dataset-path "lmarena-ai/VisionArena-Chat" --endpoint "/v1/chat/completions")
      ;;
    *)
      echo "Error: unsupported dataset '$DATASET'" > "$BM_LOG" 2>&1
      exit 1
      ;;
  esac

  # Execute the command
  "${command_to_run[@]}" "${ARGS[@]}" > "$BM_LOG" 2>&1

  echo "completed..."
  echo

  throughput=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
  p99_e2el=$(grep "P99 E2EL (ms):" "$BM_LOG" | awk '{print $NF}')
  echo "throughput: $throughput, P99 E2EL:$p99_e2el"
  echo
  echo "$throughput $p99_e2el"
}

read throughput p99_e2el < <(run_benchmark "inf" | tail -n 1)

echo "throughput:$throughput"
echo "p99_e2el:$p99_e2el"

# Step 1: check if initial run meets the E2EL requirement
p99_int=$(printf "%.0f" "$p99_e2el")
goal_int=$(printf "%.0f" "$EXPECTED_ETEL")

if (( p99_int <= goal_int )); then
  echo "Initial run: P99 E2EL ($p99_e2el ms) <= EXPECTED_ETEL ($EXPECTED_ETEL ms), good enough. Exiting 0."
  exit 0
fi

echo "Initial run failed: P99 E2EL ($p99_e2el ms) > EXPECTED_ETEL ($EXPECTED_ETEL ms)"
echo "Starting binary search to lower request rate..."

# Step 2: Begin binary search
low=0
high=$(printf "%.0f" "$throughput")
goal=$EXPECTED_ETEL

# Round goal to nearest int
goal_int=$(printf "%.0f" "$goal")

best_rate=0
best_throughput=0
best_e2el=0

while (( high - low > 0 )); do
  mid=$(( (low + high + 1) / 2 ))
  echo "Trying request_rate=$mid"

  read throughput p99_e2el < <(run_benchmark "$mid" | tail -n 1)

  # Convert p99_e2el to integer
  p99_int=$(printf "%.0f" "$p99_e2el")

  if (( p99_int <= goal_int )); then
    echo "PASS: p99_e2el=$p99_e2el <= $goal"
    best_rate=$mid
    best_throughput=$throughput
    best_e2el=$p99_e2el
    low=$mid

    # Backup best log
    cp "$BM_LOG" "$BEST_BM_LOG"
  else
    echo "FAIL: p99_e2el=$p99_e2el > $goal"
    high=$((mid - 1))
  fi
done

if (( best_rate == 0 )); then
  echo "Could not find a valid request_rate >= 1 that meets EXPECTED_ETEL=$EXPECTED_ETEL" | tee -a "$BM_LOG"
  exit 1
fi

# Restore the best log to BM_LOG
cp "$BEST_BM_LOG" "$BM_LOG"

echo
echo "======================================"
echo "✓ Final best request_rate: $best_rate"
echo "✓ Throughput: $best_throughput"
echo "✓ P99 E2EL: $best_e2el"
echo "======================================"
