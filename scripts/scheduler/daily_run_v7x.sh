#!/bin/bash
TIMEZONE="America/Los_Angeles"
TAG="$(TZ="$TIMEZONE" date +%Y%m%d_%H%M%S)"
HOUR_NOW=$(TZ="$TIMEZONE" date +%H)

# ===================================================================
# Clone code all at once and export the folder to REPO_MAP.
# In this way, all the create_job.sh below share the same git code.s
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
export SKIP_BUILD_IMAGE=1
# ===================================================================

# Ironwood qwen & Llama
echo "./scripts/scheduler/create_job.sh ./cases/daily_qwen_llama_tpu7x_2.csv \"\" $TAG DAILY TPU_INFERENCE"
./scripts/scheduler/create_job.sh ./cases/daily_qwen_llama_tpu7x_2.csv "" $TAG DAILY TPU_INFERENCE

# Ironwood Deepseek
echo "./scripts/scheduler/create_job.sh ./cases/daily_deepseek_tpu7x_8.csv \"\" $TAG DAILY TPU_INFERENCE \"JAX_RANDOM_WEIGHTS=true;VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax\""
./scripts/scheduler/create_job.sh ./cases/daily_deepseek_tpu7x_8.csv "" $TAG DAILY TPU_INFERENCE "JAX_RANDOM_WEIGHTS=true;VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax"

# Ironwood Deepseek Accuracy
echo "./scripts/scheduler/create_job.sh ./cases/accuracy_jax_v7x.csv \"\" $TAG JAX_ACCURACY TPU_INFERENCE \"VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;\""
./scripts/scheduler/create_job.sh ./cases/accuracy_jax_v7x.csv "" $TAG JAX_ACCURACY TPU_INFERENCE "VLLM_MLA_DISABLE=1;NEW_MODEL_DESIGN=True;TPU_BACKEND_TYPE=jax;"

# GPT OSS
echo "./scripts/scheduler/create_job.sh ./cases/daily_gpt_oss_120b_tpu7x_8.csv \"\" $TAG DAILY TPU_INFERENCE \"USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm\""
./scripts/scheduler/create_job.sh ./cases/daily_gpt_oss_120b_tpu7x_8.csv "" $TAG DAILY TPU_INFERENCE "USE_MOE_EP_KERNEL=0;MODEL_IMPL_TYPE=vllm"
# Qwen 3-480B
echo "./scripts/scheduler/create_job.sh ./cases/daily_qwen3_480B_FP8_tpu7x_8.csv \"\" $TAG DAILY TPU_INFERENCE"
./scripts/scheduler/create_job.sh ./cases/daily_qwen3_480B_FP8_tpu7x_8.csv "" $TAG DAILY TPU_INFERENCE

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh
