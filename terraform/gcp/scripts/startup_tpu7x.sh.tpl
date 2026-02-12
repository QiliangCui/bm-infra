#!/bin/bash 

# Set env vars system-wide
grep -q "^GCP_PROJECT_ID=" /etc/environment || echo "GCP_PROJECT_ID=${project_id}" | sudo tee -a /etc/environment
grep -q "^GCP_INSTANCE_ID=" /etc/environment || echo "GCP_INSTANCE_ID=${spanner_instance}" | sudo tee -a /etc/environment
grep -q "^GCP_DATABASE_ID=" /etc/environment || echo "GCP_DATABASE_ID=${spanner_db}" | sudo tee -a /etc/environment
grep -q "^GCP_REGION=" /etc/environment || echo "GCP_REGION=${region}" | sudo tee -a /etc/environment
grep -q "^GCP_INSTANCE_NAME=" /etc/environment || echo "GCP_INSTANCE_NAME=${instance_name}" | sudo tee -a /etc/environment
grep -q "^GCS_BUCKET=" /etc/environment || echo "GCS_BUCKET=${gcs_bucket}" | sudo tee -a /etc/environment
grep -q "^GCP_QUEUE=" /etc/environment || echo "GCP_QUEUE=vllm-${purpose}-queue-${accelerator_type}" | sudo tee -a /etc/environment
grep -q "^LOCAL_RUN_BM=" /etc/environment || echo "LOCAL_RUN_BM=1" | sudo tee -a /etc/environment
if ! grep -q "^HF_TOKEN=" /etc/environment; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --project=${project_id} --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)"
fi

if [[ -z "${USERNAME}" ]]; then
  USERNAME="bm-agent"
fi

if ! id -u ${USERNAME} >/dev/null 2>&1; then
  echo "sudo useradd -m -s /bin/bash ${USERNAME}"
  sudo useradd -m -s /bin/bash ${USERNAME}
fi

apt-get update
#                              apt-get install -y curl build-essential jq
DEBIAN_FRONTEND=noninteractive apt-get install -yq curl build-essential jq

# use the local docker
# curl -o- https://get.docker.com/ | bash -

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
/root/.cargo/bin/cargo install minijinja-cli
cp /root/.cargo/bin/minijinja-cli /usr/bin/minijinja-cli
chmod 777 /usr/bin/minijinja-cli

# Mount persistent disk

sudo mkdir -p /mnt/disks/persist

# Robustly waiting for the persistent disk to appear.
# This loop willwiat for up to 5 minutes (300 seconds)
WAIT_SECONDS=300
for ((i=0; i<WAIT_SECONDS; i++)); do
  # The '-b' flag checks if it is a block device
  if [ -b /dev/${persistent_device_name} ]; then
    echo "✅ Disk /dev/${persistent_device_name} found!"
    break
  fi
  sleep 1
done

# Check if the loop timed out
if [ ! -b /dev/${persistent_device_name} ]; then
  echo "❌ Error: Timed out waiting for disk /dev/${persistent_device_name} to appear after $WAIT_SECONDS seconds."
  # Log the available block devices for debugging
  echo "Available block devices:"
  lsblk
  exit 1
fi

# Format if not already formatted
if ! blkid /dev/${persistent_device_name}; then
  echo "Formatting /dev/${persistent_device_name} as ext4..."
  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/${persistent_device_name}
fi

# Only mount if not already mounted (first boot or recovery)
if ! mountpoint -q /mnt/disks/persist; then
  sudo mount /dev/${persistent_device_name} /mnt/disks/persist
fi

# Add ownership of USERNAME on /mnt/disks/persist
sudo chown -R ${USERNAME}:${USERNAME} /mnt/disks/persist

# Install miniconda for local bm run.
sudo -u ${USERNAME} -i bash <<'EOF'
set -euo pipefail

# Miniconda version and install directory
MINICONDA_VERSION=latest  # adjust if needed
MINICONDA_DIR="/mnt/disks/persist/bm-agent/miniconda3"
MINICONDA_SCRIPT="Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_SCRIPT"

# Download Miniconda installer if not exists
if [ ! -f "$HOME/$MINICONDA_SCRIPT" ]; then
  echo "Downloading Miniconda installer..."
  curl -fsSL "$MINICONDA_URL" -o "$HOME/$MINICONDA_SCRIPT"
fi

# Install Miniconda silently if not installed
if [ ! -d "$MINICONDA_DIR" ]; then
  echo "Installing Miniconda to $MINICONDA_DIR..."
  bash "$HOME/$MINICONDA_SCRIPT" -b -p "$MINICONDA_DIR"
fi

# Initialize conda for bash shell
eval "$($MINICONDA_DIR/bin/conda shell.bash hook)" || true

# Add conda init to .bashrc if not already there
if ! grep -q "conda initialize" "$HOME/.bashrc"; then
  echo "Adding conda initialize to .bashrc..."
  "$MINICONDA_DIR/bin/conda" init bash
fi

# Accept Terms of Service for required channels
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
"$MINICONDA_DIR/bin/conda" tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "Miniconda installation complete."

EOF

jq ". + {\"data-root\": \"/mnt/disks/persist\"}" /etc/docker/daemon.json > /tmp/daemon.json.tmp && mv /tmp/daemon.json.tmp /etc/docker/daemon.json
systemctl stop docker
systemctl daemon-reload
systemctl start docker

# Docker change the permissions. So resetting it again.
sudo chown -R ${USERNAME}:${USERNAME} /mnt/disks/persist

sudo usermod -aG docker ${USERNAME}

# Run the commands below as bm-agent user:
sudo -u ${USERNAME} -i bash << EOBM
gcloud auth configure-docker ${region}-docker.pkg.dev --quiet
rm -rf bm-infra
git clone -b ${branch_name} https://github.com/QiliangCui/bm-infra.git

EOBM
cp /home/${USERNAME}/bm-infra/service/bm-agent/bm-agent.service /etc/systemd/system/bm-agent.service
systemctl stop bm-agent.service
systemctl daemon-reload
systemctl enable bm-agent.service
systemctl start bm-agent.service
