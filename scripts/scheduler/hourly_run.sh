#!/bin/bash
TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
HOUR_NOW=$(TZ="$TIMEZONE" date +%H)

echo "./scripts/scheduler/create_job.sh ./cases/hourly.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh ./cases/hourly.csv "" $TAG HOURLY

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
# for now, we start from v6e-1.
echo "./scripts/scheduler/create_job.sh ./cases/hourly_torchax.csv \"\" $TAG HOURLY_TORCHAX TPU_COMMONS"
./scripts/scheduler/create_job.sh ./cases/hourly_torchax.csv "" $TAG HOURLY_TORCHAX TPU_COMMONS

# Torchax spmd
echo "./scripts/scheduler/create_job.sh cases/hourly_torchaxspmd.csv \"\" $TAG HOURLY_TORCHAX TPU_COMMONS" torchaxspmd
./scripts/scheduler/create_job.sh cases/hourly_torchaxspmd.csv "" $TAG HOURLY_TORCHAX TPU_COMMONS torchaxspmd

# Run TPU Commons + JAX test.
# Eventually, JAX and vLLM should run the same test case.
# for now, we start from v6e-1.
# TODO(b/429439832): we ideally don't want to pin the vLLM version, but we want to keep it in sync with the TPU Commons-needed vLLM version
echo "./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv \"3c545c0c3\" $TAG HOURLY_JAX TPU_COMMONS jax"
./scripts/scheduler/create_job.sh ./cases/hourly_jax.csv "3c545c0c3" $TAG HOURLY_JAX TPU_COMMONS jax


if [[ "$HOUR_NOW" == "00" || "$HOUR_NOW" == "12" ]]; then
  # vLLM
  echo "./scripts/scheduler/create_job.sh ./cases/autotune.csv \"\" $TAG AUTOTUNE"
  ./scripts/scheduler/create_job.sh ./cases/autotune.csv "" $TAG AUTOTUNE

  # Torchax
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_torchax.csv \"\" $TAG AUTOTUNE_TORCHAX TPU_COMMONS"
  ./scripts/scheduler/create_job.sh ./cases/autotune_torchax.csv "" $TAG AUTOTUNE_TORCHAX TPU_COMMONS

  # Torchax spmd
  echo "./scripts/scheduler/create_job.sh cases/autotune_torchaxspmd.csv \"\" $TAG AUTOTUNE_TORCHAX TPU_COMMONS" torchaxspmd
  ./scripts/scheduler/create_job.sh cases/autotune_torchaxspmd.csv "" $TAG AUTOTUNE_TORCHAX TPU_COMMONS torchaxspmd

  # TODO(b/429439832): we ideally don't want to pin the vLLM version, but we want to keep it in sync with the TPU Commons-needed vLLM version
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv \"3c545c0c3\" $TAG AUTOTUNE_JAX TPU_COMMONS jax"  
  ./scripts/scheduler/create_job.sh ./cases/autotune_jax.csv "3c545c0c3" $TAG AUTOTUNE_JAX TPU_COMMONS jax

  # Adhoc
  echo "./scripts/scheduler/create_job.sh ./cases/autotune_adhoc.csv \"\" $TAG AUTOTUNE "
  ./scripts/scheduler/create_job.sh ./cases/autotune_adhoc.csv "" $TAG AUTOTUNE
fi

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh
