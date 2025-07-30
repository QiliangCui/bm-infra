#!/bin/bash
set -euo pipefail

# Constants
THRESHOLD_SECONDS=$((60 * 60 * 24))  # 1 day
NOW=$(date +%s)
ENV_BASE="/mnt/disks/persist/bm-agent/uv"
ENV_PREFIX="venv-"

echo "Scanning for old '${ENV_PREFIX}*' environments older than 1 day in $ENV_BASE..."

for env_path in "$ENV_BASE"/${ENV_PREFIX}*; do
  [ -d "$env_path" ] || continue
  env_name=$(basename "$env_path")

  # Get last modification time
  mtime=$(stat -c "%Y" "$env_path")
  age=$((NOW - mtime))

  if [ "$age" -gt "$THRESHOLD_SECONDS" ]; then
    echo "Deleting old env: $env_name (age: $((age / 3600)) hours)"
    rm -rf "$env_path" || echo "Failed to remove $env_name"
  else
    echo "Keeping env: $env_name (recently modified)"
  fi
done
