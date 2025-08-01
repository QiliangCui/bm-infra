#!/bin/bash
TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
HOUR_NOW=$(TZ="$TIMEZONE" date +%H)

echo "./scripts/scheduler/create_job.sh ./cases/hourly.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly.csv "" $TAG HOURLY

# Run b200-8
# todo: this can be merged into hourly run.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_b200.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly_b200.csv "" $TAG HOURLY

# Run gpu_1 on even hours, gpu_2 on odd hours
# Because I don't have enough h100-8 now.
if (( 10#$HOUR_NOW % 2 == 0 )); then
  echo "./scripts/scheduler/create_job.sh ./cases/hourly_gpu_1.csv \"\" $TAG HOURLY"
  ./scripts/scheduler/create_job.sh ./cases/hourly_gpu_1.csv "" $TAG HOURLY
else
  echo "./scripts/scheduler/create_job.sh ./cases/hourly_gpu_2.csv \"\" $TAG HOURLY"
  ./scripts/scheduler/create_job.sh ./cases/hourly_gpu_2.csv "" $TAG HOURLY
fi

# Run TPU Commons + TorchAX test.
# Eventually, TorchAx and vLLM should run the same test case.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax.csv \"\" $TAG HOURLY_TORCHAX TPU_COMMONS_TORCHAX \"TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=0\""
./scripts/scheduler/create_job.sh ./cases/hourly_torchax.csv "" $TAG HOURLY_TORCHAX TPU_COMMONS_TORCHAX "TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=0"

# Torchax spmd
echo "./scripts/scheduler/create_job.sh cases/hourly_torchaxspmd.csv \"\" $TAG HOURLY_TORCHAX TPU_COMMONS_TORCHAX \"TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=1\""
./scripts/scheduler/create_job.sh cases/hourly_torchaxspmd.csv "" $TAG HOURLY_TORCHAX TPU_COMMONS_TORCHAX "TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=1"

# Run TPU Commons + JAX test.
# Eventually, JAX and vLLM should run the same test case.
# for now, we start from v6e-1.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv \"\" $TAG HOURLY_JAX TPU_COMMONS \"TPU_BACKEND_TYPE=jax\""
./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv "" $TAG HOURLY_JAX TPU_COMMONS "TPU_BACKEND_TYPE=jax"

# Run JAX with new model design
./scripts/scheduler/create_job.sh ./cases/hourly_jax_new.csv "" $TAG HOURLY_JAX TPU_COMMONS "TPU_BACKEND_TYPE=jax;NEW_MODEL_DESIGN=True"

# Run JAX with mmlu dataset
./scripts/scheduler/create_job.sh ./cases/hourly_jax_mmlu.csv "" $TAG HOURLY_JAX_MMLU TPU_COMMONS "TPU_BACKEND_TYPE=jax"

# Run Torchax + jax backend
echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax.csv \"\" $TAG HOURLY_AX_JAX TPU_COMMONS \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
./scripts/scheduler/create_job.sh ./cases/hourly_torchax_jax.csv "" $TAG HOURLY_AX_JAX TPU_COMMONS "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"

if [[ "$HOUR_NOW" == "00" || "$HOUR_NOW" == "12" ]]; then
  # vLLM
  echo "./scripts/scheduler/create_job.sh ./cases/autotune.csv \"\" $TAG AUTOTUNE"
  ./scripts/scheduler/create_job.sh ./cases/autotune.csv "" $TAG AUTOTUNE

  # Torchax
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_torchax.csv \"\" $TAG AUTOTUNE_TORCHAX TPU_COMMONS_TORCHAX \"TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=0\""
  ./scripts/scheduler/create_job.sh ./cases/autotune_torchax.csv "" $TAG AUTOTUNE_TORCHAX TPU_COMMONS_TORCHAX "TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=0"

  # Torchax spmd
  echo "./scripts/scheduler/create_job.sh cases/autotune_torchaxspmd.csv \"\" $TAG AUTOTUNE_TORCHAX TPU_COMMONS_TORCHAX \"TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=1\""
  ./scripts/scheduler/create_job.sh cases/autotune_torchaxspmd.csv "" $TAG AUTOTUNE_TORCHAX TPU_COMMONS_TORCHAX "TPU_BACKEND_TYPE=torchax;VLLM_TORCHAX_ENABLED=1;VLLM_XLA_USE_SPMD=1"

  echo "./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv \"\" $TAG AUTOTUNE_JAX TPU_COMMONS \"TPU_BACKEND_TYPE=jax\""
  ./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv "" $TAG AUTOTUNE_JAX TPU_COMMONS "TPU_BACKEND_TYPE=jax"

  # Run Torchax + jax backend
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax.csv \"\" $TAG AUTOTUNE_AX_JAX TPU_COMMONS \"TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm\""
  ./scripts/scheduler/create_job.sh ./cases/autotune_torchax_jax.csv "" $TAG AUTOTUNE_AX_JAX TPU_COMMONS "TPU_BACKEND_TYPE=jax;MODEL_IMPL_TYPE=vllm"

  # Adhoc
  # echo "./scripts/scheduler/create_job.sh ./cases/autotune_adhoc.csv \"\" $TAG AUTOTUNE "
  # ./scripts/scheduler/create_job.sh ./cases/autotune_adhoc.csv "" $TAG AUTOTUNE
fi

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh
