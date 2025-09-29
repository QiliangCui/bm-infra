#!/bin/bash

set -euo pipefail

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

#
# create a log and profile folder
#
mkdir -p "$WORKSPACE/log"
mkdir -p "$PROFILE_FOLDER"

# TODO: Move to image building.
# Ingore the error because in case of using uv, the packages are installed outside this script.
pip install pandas || true
pip install datasets || true
pip install evaluate==0.4.5 || true
pip install rouge-score==0.1.2 || true

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
  EXTRA_ARGS+=" --hf-config=deepseek-ai/DeepSeek-R1 --hf_overrides '{\"architectures\": [\"DeepseekV3ForCausalLM\"]}' --gpu-memory-utilization 0.7"
fi

if [[ "$MODEL" == "Qwen/Qwen3-32B" ]]; then
  echo "Qwen/Qwen3-32B"
  EXTRA_ARGS+=" --kv-cache-dtype=fp8 --gpu-memory-utilization=0.96"
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
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS > \"$VLLM_LOG\" 2>&1 &"

eval "VLLM_USE_V1=1 VLLM_TORCH_PROFILER_DIR=\"$PROFILE_FOLDER\" vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN $EXTRA_ARGS > \"$VLLM_LOG\" 2>&1 &"


echo "wait for 20 minutes.."
echo
# sleep 1200
# wait for 10 minutes...
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
  #
  # run test
  #
  echo "run benchmark test..."
  echo "logging to $BM_LOG"
  echo
  
  request_rate="$1"
  if [ "$DATASET" = "sonnet" ]; then
    python benchmarks/benchmark_serving.py \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name sonnet \
      --dataset-path benchmarks/sonnet_4x.txt \
      --sonnet-input-len $INPUT_LEN \
      --sonnet-output-len $OUTPUT_LEN \
      --num-prompts ${NUM_PROMPTS} \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1

  elif [ "$DATASET" = "random" ]; then
    python benchmarks/benchmark_serving.py \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name random \
      --random-input-len $INPUT_LEN \
      --random-output-len $OUTPUT_LEN \
      --num-prompts ${NUM_PROMPTS} \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1
  elif [ "$DATASET" = "mmlu" ]; then
    python benchmarks/benchmark_serving.py \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name mmlu \
      --dataset-path /workspace/dataset \
      --mmlu-num-shots 0 \
      --mmlu-method HELM \
      --num-prompts ${NUM_PROMPTS} \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1
  elif [ "$DATASET" = "mlperf" ]; then
    python benchmarks/benchmark_serving.py \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name mlperf \
      --dataset-path /workspace/dataset/processed-data.pkl \
      --mlperf-input-len $INPUT_LEN \
      --max-model-len $MAX_MODEL_LEN \
      --num-prompts ${NUM_PROMPTS} \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1
  elif [ "$DATASET" = "custom-token" ]; then
    dataset_path="$WORKSPACE/dataset/${MODEL##*/}_${INPUT_LEN}_${OUTPUT_LEN}_tp${TENSOR_PARALLEL_SIZE}.json"
    python benchmarks/benchmark_serving.py \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name custom-token \
      --dataset-path $dataset_path \
      --num-prompts ${NUM_PROMPTS} \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1
  elif [ "$DATASET" = "bench-custom-token" ]; then
    dataset_path="$WORKSPACE/dataset/${MODEL##*/}/inlen${INPUT_LEN}_outlen${OUTPUT_LEN}_prefixlen${PREFIX_LEN}.jsonl"
    echo "dataset_path: $dataset_path"
    vllm bench serve \
      --backend vllm \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name custom \
      --dataset-path $dataset_path \
      --num-prompts ${NUM_PROMPTS} \
      --custom-output-len $OUTPUT_LEN \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --custom-skip-chat-template \
      --ignore-eos > "$BM_LOG" 2>&1
  elif [ "$DATASET" = "sharegpt" ]; then
    dataset_path="$WORKSPACE/dataset/ShareGPT_V3_unfiltered_cleaned_split.json"

    if [ "$INPUT_LEN" -gt 0 ]; then
      echo "Please set INPUT_LEN to 0 for sharegpt dataset because it is not used." > "$BM_LOG" 2>&1      
      exit 1
    fi
    
    ARGS=(
      --backend vllm
      --model "$MODEL"
      --request-rate "$request_rate"
      --dataset-name sharegpt
      --dataset-path "$dataset_path"
      --num-prompts "$NUM_PROMPTS"
      --percentile-metrics ttft,tpot,itl,e2el
      --ignore-eos
      $PROFILE_FLAG
    )

    if [ "$OUTPUT_LEN" -ne 0 ]; then
      ARGS+=(--sharegpt-output-len "$OUTPUT_LEN")
    fi    

    python benchmarks/benchmark_serving.py "${ARGS[@]}" > "$BM_LOG" 2>&1

  # only used for VL model
  elif [ "$DATASET" = "hf" ]; then
    dataset_path="lmarena-ai/VisionArena-Chat" 
    python benchmarks/benchmark_serving.py \
      --backend openai-chat \
      --model $MODEL \
      --request-rate $request_rate \
      --dataset-name hf \
      --dataset-path $dataset_path \
      --num-prompts "$NUM_PROMPTS" \
      --endpoint /v1/chat/completions \
      --percentile-metrics ttft,tpot,itl,e2el \
      $PROFILE_FLAG \
      --ignore-eos > "$BM_LOG" 2>&1

  else
    echo "Error: unsupported dataset '$DATASET'" > "$BM_LOG" 2>&1
    exit 1
  fi

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
