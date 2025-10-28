#!/bin/bash
TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
HOUR_NOW=$(TZ="$TIMEZONE" date +%H)

# ===================================================================
# Clone code all at once and export the folder to REPO_MAP.
# In this way, all the create_job.sh below share the same git code.s

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh

rm -rf repos/
mkdir -p repos/

git clone https://github.com/vllm-project/vllm.git repos/vllm
git clone https://github.com/vllm-project/tpu-inference.git repos/tpu-inference
git clone https://github.com/pytorch/xla.git repos/xla

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
echo "./scripts/scheduler/create_job.sh ./cases/hourly.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly.csv "" $TAG HOURLY

echo "./scripts/scheduler/create_job.sh ./cases/hourly_xla_meta.csv \"\" $TAG HOURLY_XLA_META DEFAULT \"PROFILE=0\""
./scripts/scheduler/create_job.sh ./cases/hourly_xla_meta.csv "" $TAG HOURLY_XLA_META DEFAULT "PROFILE=0"

echo "./scripts/scheduler/create_job.sh ./cases/hourly_customer1.csv \"\" $TAG CUSTOMER1_HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly_customer1.csv "" $TAG CUSTOMER1_HOURLY

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

