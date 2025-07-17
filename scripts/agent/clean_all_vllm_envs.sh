#!/bin/bash
set -euo pipefail

# Constants
NOW=$(date +%s)
CONDA="/mnt/disks/persist/bm-agent/miniconda3/bin/conda"
ENV_BASE="$($CONDA info --base)/envs"

echo "Scanning for old 'vllm-bm-*' environments older than 1 day in $ENV_BASE..."

for env_path in "$ENV_BASE"/vllm-bm-*; do
  [ -d "$env_path" ] || continue
  env_name=$(basename "$env_path")

  # Get last modification time
  mtime=$(stat -c "%Y" "$env_path")
  age=$((NOW - mtime))

  
  echo "Deleting old env: $env_name (age: $((age / 3600)) hours)"
  $CONDA remove -n "$env_name" --all -y || echo "Failed to remove $env_name"
  
done
