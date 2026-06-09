#!/bin/bash

set -euo pipefail

docker_root=$(docker info -f '{{.DockerRootDir}}')
if [ -z "$docker_root" ]; then
  echo "Failed to determine Docker root directory."
  exit 1
fi
echo "Docker root directory: $docker_root"
# Check disk usage of the filesystem where Docker's root directory is located
disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
# Define the threshold
threshold=70
if [[ "$disk_usage" =~ ^[0-9]+$ ]] && [ "$disk_usage" -gt "$threshold" ]; then
  echo "Disk usage($disk_usage%) is above $threshold%. Cleaning up Docker images and volumes..."
  # Remove dangling images (those that are not tagged and not used by any container)
  docker image prune -f
  # Remove unused volumes / force the system prune for old images as well.
  docker volume prune -f && docker system prune --force --filter "until=12h" --all
  echo "Docker images and volumes cleanup completed."
else
  echo "Disk usage($disk_usage%) is below $threshold%. No cleanup needed."
fi

# Clean up JAX/PyTorch TPU shared memory compilation cache to prevent RESOURCE_EXHAUSTED / No space left on device errors
echo "Checking shared memory (/dev/shm) usage..."
shm_usage=$(df /dev/shm | tail -1 | awk '{print $5}' | sed 's/%//')
echo "Shared memory usage is $shm_usage%"

# Always clean files older than 1 day to prevent slow buildup
echo "Cleaning torch_tpu_cache files older than 1 day..."
docker run --rm -v /dev/shm:/dev/shm alpine find /dev/shm/torch_tpu_cache -mindepth 1 -mtime +1 -delete 2>/dev/null || true

if [[ "$shm_usage" =~ ^[0-9]+$ ]] && [ "$shm_usage" -gt 70 ]; then
  echo "Shared memory usage ($shm_usage%) is above 70%. Forcing complete clear of torch_tpu_cache..."
  docker run --rm -v /dev/shm:/dev/shm alpine find /dev/shm/torch_tpu_cache -mindepth 1 -delete 2>/dev/null || true
  echo "Shared memory cleanup completed."
fi