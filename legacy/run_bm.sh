#!/bin/bash

TAG=""
VLLM_LOG="$WORKSPACE/vllm_log.txt"
BM_LOG="$WORKSPACE/bm_log.txt"
RESULT="$WORKSPACE/result.txt"
HASH_FILE="$WORKSPACE/hash.txt"
TABLE_FILE="$WORKSPACE/table.txt"

echo "running tag $TAG"
echo "result file$ $RESULT"
echo "model: $MODEL"
echo

#
# create a log folder
#
mkdir "$WORKSPACE/log"

cd "$VLLM_CODE"

if [ "$SYNC_TO_HEAD" -eq 1 ]; then
    echo "sync code to latest"
    echo
    git checkout main
    git reset --hard
    git pull    

    echo "pip uninstall -y torch torch_xla jax jaxlib libtpu_nightly"
    echo
    pip uninstall -y torch torch_xla jax jaxlib libtpu_nightly

    echo "pip install -r $REQUIREMENTS"
    pip install -r $REQUIREMENTS

else
    echo "skip sync..."
fi

pip install pandas
pip install datasets


echo "time:$TAG" >> "$RESULT"
current_hash=$(git rev-parse HEAD)
echo "hash:$current_hash" >> "$RESULT"
echo "$current_hash" > $HASH_FILE

#
# create sonnet_4x
#
echo "Create sonnet_4x.txt"
echo "" > benchmarks/sonnet_4x.txt
for _ in {1..4}
 do
  cat benchmarks/sonnet.txt >> benchmarks/sonnet_4x.txt
done

#
# start vllm service in backend
#
echo "lanching vllm..."
echo "logging to $VLLM_LOG"
echo

# --gpu-memory-utilization 0.98 \
LIBTPU_INIT_ARGS=“--xla_tpu_force_1d_allreduce_at_chunk_count=1” VLLM_USE_V1=1 vllm serve $MODEL \
 --seed 42 \
 --disable-log-requests \
 --port 8004 \
 --max-num-seqs $MAX_NUM_SEQS \
 --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
 --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
 --no-enable-prefix-caching \
 --download_dir $DOWNLOAD_DIR \
 --max-model-len $MAX_MODEL_LEN > "$VLLM_LOG" 2>&1 &

echo "wait for 10 minutes.."
echo
# sleep 600
# wait for 10 minutes...
for i in {1..60}; do        
    if grep -Fq "Application startup complete" "$VLLM_LOG"; then
        echo "Application started"
        break
    else
        echo "wait for 10 seconds..."
        sleep 10
    fi
done

#
# run test
#
echo "run benchmark test..."
echo "logging to $BM_LOG"
echo
vllm bench serve \
    --backend vllm \
    --model $MODEL  \
    --dataset-name sonnet \
    --dataset-path benchmarks/sonnet_4x.txt \
    --sonnet-input-len 1800 \
    --sonnet-output-len 128 \
    --ignore-eos \
    --port 8004 > "$BM_LOG"

echo "complelted..."
echo

through_put=$(grep "Request throughput (req/s):" "$BM_LOG" | sed 's/[^0-9.]//g')
echo "through put: $through_put"
echo "through put: $through_put" >> "$RESULT"
echo

echo "$TAG,$current_hash,$through_put" >> "$TABLE_FILE"

echo "pkill -f vllm"
echo
sudo pkill -f vllm
sleep 10
