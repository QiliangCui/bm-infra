#!/bin/bash

# Benchmark generation is paused. The v6e fleet that consumes these queues is
# being migrated off the us-east5-a reservation, so publishing more work would
# only fill Pub/Sub with records no agent will claim -- messages expire after
# the subscriptions' 7 day retention and leave their Spanner rows stranded in
# CREATED forever.
#
# The guard lives here rather than in hourly_run_wrapper.sh so the wrapper still
# runs its `git pull`: that pull is the only thing that carries a change from
# this repo onto the scheduler VM, and gating before it would strand the VM on
# this commit with no way back short of touching the box.
#
# To resume, either flip the default below back to 1 and let the hourly pull
# pick it up, or export BM_SCHEDULER_ENABLED=1 in /etc/environment on the
# scheduler VM -- both units read it via EnvironmentFile.
if [[ "${BM_SCHEDULER_ENABLED:-0}" != "1" ]]; then
  echo "bm scheduler is paused (BM_SCHEDULER_ENABLED != 1). No jobs created."
  exit 0
fi

TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
HOUR_NOW=$(TZ="$TIMEZONE" date +%H)

# ===================================================================
# Clone code all at once and export the folder to REPO_MAP.
# In this way, all the create_job.sh below share the same git code.s

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh

# Retry git clone to ride through transient GitHub HTTP 5xx errors. A failed
# clone leaves repos/ incomplete and breaks every downstream job that depends
# on the missing path, so any clone failure must be fatal after retries.
clone_with_retry() {
  local url="$1" dest="$2"
  local max_attempts=4 attempt=1 sleep_s=5
  while true; do
    rm -rf "$dest"
    if git clone "$url" "$dest"; then
      return 0
    fi
    if (( attempt >= max_attempts )); then
      echo "ERROR: Failed to clone $url after $max_attempts attempts" >&2
      return 1
    fi
    echo "Clone attempt $attempt/$max_attempts failed for $url; retrying in ${sleep_s}s..." >&2
    sleep "$sleep_s"
    attempt=$(( attempt + 1 ))
    sleep_s=$(( sleep_s * 2 ))
  done
}

rm -rf repos/
mkdir -p repos/

clone_with_retry https://github.com/vllm-project/vllm.git repos/vllm || exit 1
clone_with_retry https://github.com/vllm-project/tpu-inference.git repos/tpu-inference || exit 1
clone_with_retry https://github.com/pytorch/xla.git repos/xla || exit 1

map_entries=(
  "https://github.com/vllm-project/vllm.git||repos/vllm"
  "https://github.com/vllm-project/tpu-inference.git||repos/tpu-inference"
  "https://github.com/pytorch/xla.git||repos/xla"
)

# Join the array elements with a semicolon
# We temporarily change the Internal Field Separator (IFS) to ';'
OLD_IFS="$IFS"
IFS=';'
REPO_MAP_STRING="${map_entries[*]}"
IFS="$OLD_IFS" # Restore IFS immediately

# Now export the final, correctly formatted string
export REPO_MAP="$REPO_MAP_STRING"
# ===================================================================

# torch xla
# echo "./scripts/scheduler/create_job.sh ./cases/hourly.csv \"\" $TAG HOURLY"
# ./scripts/scheduler/create_job.sh ./cases/hourly.csv "" $TAG HOURLY

# echo "./scripts/scheduler/create_job.sh ./cases/hourly_xla_meta.csv \"\" $TAG HOURLY_XLA_META DEFAULT \"PROFILE=0\""
# ./scripts/scheduler/create_job.sh ./cases/hourly_xla_meta.csv "" $TAG HOURLY_XLA_META DEFAULT "PROFILE=0"

# echo "./scripts/scheduler/create_job.sh ./cases/hourly_customer1.csv \"\" $TAG CUSTOMER1_HOURLY"
# ./scripts/scheduler/create_job.sh ./cases/hourly_customer1.csv "" $TAG CUSTOMER1_HOURLY

# Run gpu_1 on even hours, gpu_2 on odd hours
# Because I don't have enough h100-8 now.
if (( 10#$HOUR_NOW % 2 == 0 )); then
  echo "./scripts/scheduler/create_job.sh ./cases/hourly_gpu_1.csv \"\" $TAG HOURLY"
  ./scripts/scheduler/create_job.sh ./cases/hourly_gpu_1.csv "" $TAG HOURLY

  echo "./scripts/scheduler/create_job.sh ./cases/hourly_gpu_customer1.csv \"\" $TAG CUSTOMER1_HOURLY"
  ./scripts/scheduler/create_job.sh ./cases/hourly_gpu_customer1.csv "" $TAG CUSTOMER1_HOURLY
else
  echo "./scripts/scheduler/create_job.sh ./cases/hourly_gpu_2.csv \"\" $TAG HOURLY"
  ./scripts/scheduler/create_job.sh ./cases/hourly_gpu_2.csv "" $TAG HOURLY
fi

# Run b200-8
# todo: this can be merged into hourly run.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_b200.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly_b200.csv "" $TAG HOURLY



# Run TPU Inference + JAX test.
# Eventually, JAX and vLLM should run the same test case.
# for now, we start from v6e-1.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv \"\" $TAG HOURLY_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax\""
./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv "" $TAG HOURLY_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax"

# Run JAX with new model design
./scripts/scheduler/create_job.sh ./cases/hourly_jax_new.csv "" $TAG HOURLY_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True"

# Run Torchax + jax backend
echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax.csv \"\" $TAG HOURLY_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax.csv "" $TAG HOURLY_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"

echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax_customer1.csv \"\" $TAG CUSTOMER1_HOURLY_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax_customer1.csv "" $TAG CUSTOMER1_HOURLY_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"


if [[ "$HOUR_NOW" == "00" || "$HOUR_NOW" == "12" ]]; then
  :
  # vLLM
  # echo "./scripts/scheduler/create_job.sh ./cases/autotune.csv \"\" $TAG AUTOTUNE"
  # ./scripts/scheduler/create_job.sh ./cases/autotune.csv "" $TAG AUTOTUNE

  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_xla_meta.csv \"\" $TAG AUTOTUNE_XLA_META DEFAULT \"PROFILE=0\""
  # ./scripts/scheduler/create_job.sh ./cases/autotune_xla_meta.csv "" $TAG AUTOTUNE_XLA_META DEFAULT "PROFILE=0"

  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_customer1.csv \"\" $TAG CUSTOMER1_AUTOTUNE"
  # ./scripts/scheduler/create_job.sh ./cases/autotune_customer1.csv "" $TAG CUSTOMER1_AUTOTUNE

  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv \"\" $TAG AUTOTUNE_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax\""
  # ./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv "" $TAG AUTOTUNE_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax"
fi

# Too many autotune that can't be scheduled in one hour
if [[ "$HOUR_NOW" == "01" || "$HOUR_NOW" == "13" ]]; then
  # Run Torchax + jax backend
  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax.csv \"\" $TAG AUTOTUNE_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
  # ./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax.csv "" $TAG AUTOTUNE_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"

  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax_customer1.csv \"\" $TAG CUSTOMER1_AUTOTUNE_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
  # ./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax_customer1.csv "" $TAG CUSTOMER1_AUTOTUNE_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"

  # JAX accuracy
  echo "./scripts/scheduler/create_job.sh ./cases/accuracy_jax.csv \"\" $TAG JAX_ACCURACY TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;\""
  ./scripts/scheduler/create_job.sh ./cases/accuracy_jax.csv "" $TAG JAX_ACCURACY TPU_INFERENCE "TPU_BACKEND_TYPE=jax;"

fi

# Too many autotune that can't be scheduled in one hour, separating these runs from autotune above.
if [[ "$HOUR_NOW" == "03" || "$HOUR_NOW" == "15" ]]; then
#   # Run comparison benchmarks
  echo "./scripts/scheduler/create_job.sh ./cases/nightly_jax.csv \"\" $TAG BENCH_COMP_TPU TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True\""
  ./scripts/scheduler/create_job.sh ./cases/nightly_jax.csv "" $TAG BENCH_COMP_TPU TPU_INFERENCE "TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True"
fi

if [[ "$HOUR_NOW" == "02" ]]; then
  # B200 not enough hardware to run it twice a day.
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_b200.csv \"\" $TAG AUTOTUNE"
  ./scripts/scheduler/create_job.sh ./cases/autotune_b200.csv "" $TAG AUTOTUNE
fi

# if [[ "$HOUR_NOW" == "14" ]]; then
#   # Run ali tunes.
#   echo "./scripts/scheduler/create_job.sh ./cases/autotune_ali.csv \"\" $TAG AUTOTUNE"
#   ./scripts/scheduler/create_job.sh ./cases/autotune_ali.csv "" $TAG AUTOTUNE
# fi

echo LOCAL_PATCH=1 ./scripts/scheduler/create_job.sh ./cases/hourly_disagg.csv "" $TAG HOURLY_DISAGG TPU_INFERENCE "PREFILL_SLICES=2;DECODE_SLICES=2;TPU_BACKEND_TYPE=jax"
LOCAL_PATCH=1 ./scripts/scheduler/create_job.sh ./cases/hourly_disagg.csv "" $TAG HOURLY_DISAGG TPU_INFERENCE "PREFILL_SLICES=2;DECODE_SLICES=2;TPU_BACKEND_TYPE=jax"

# torchax v7
echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax_v7.csv \"\" $TAG HOURLY_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\" tt"
./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax_v7.csv "" $TAG HOURLY_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm" tt

# Run 480B benchmark cases three times daily (12am, 6am, 12pm PT) to avoid overloading queue
if [[ "$HOUR_NOW" == "00" || "$HOUR_NOW" == "06" || "$HOUR_NOW" == "12" ]]; then
  echo "Running three times daily 480B benchmarks..."

  echo "./scripts/scheduler/create_job.sh ./cases/three_times_daily_torchax_v7.csv \"\" $TAG DAILY_AX_JAX TPU_INFERENCE \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\" tt"
  ./scripts/scheduler/create_job.sh ./cases/three_times_daily_torchax_v7.csv "" $TAG DAILY_AX_JAX TPU_INFERENCE "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm" tt
fi
