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
git clone https://github.com/vllm-project/tpu_commons.git repos/tpu_commons
git clone https://github.com/pytorch/xla.git repos/xla

map_entries=(
  "https://github.com/vllm-project/vllm.git||repos/vllm"
  "https://github.com/vllm-project/tpu_commons.git||repos/tpu_commons"
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

# Ironwood
echo "./scripts/scheduler/create_job.sh ./cases/hourly.csv \"\" $TAG HOURLY"
./scripts/scheduler/create_job.sh gs://amangu-multipods/ironwood/cases/hourly_ironwood.csv "d328f7894f140fdc643dc1aa5fe80f4596e6f418-5797c31acb0010cf8c54ba9218bacf96d8a1260e-" $TAG HOURLY TPU_COMMONS

echo "./scripts/cleanup_docker.sh"
./scripts/cleanup_docker.sh
