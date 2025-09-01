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

apt-get update
apt-get install -y curl build-essential jq

curl -o- https://get.docker.com/ | bash -

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
/root/.cargo/bin/cargo install minijinja-cli
cp /root/.cargo/bin/minijinja-cli /usr/bin/minijinja-cli
chmod 777 /usr/bin/minijinja-cli

# Mount persistent disk

sudo mkdir -p /mnt/disks/persist
sudo chmod 777 /mnt/disks/persist

# Format if not already formatted
if ! blkid /dev/${persistent_device_name}; then
  echo "Formatting /dev/${persistent_device_name} as ext4..."
  sudo mkfs.ext4 -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard /dev/${persistent_device_name}
fi

# Add to /etc/fstab using UUID
disk_uuid=$(blkid -s UUID -o value /dev/${persistent_device_name}
if ! grep -q "/mnt/disks/persist" /etc/fstab; then
  echo "UUID=$disk_uuid /mnt/disks/persist ext4 defaults,discard 0 2" | sudo tee -a /etc/fstab
fi

# Only mount if not already mounted (first boot or recovery)
if ! mountpoint -q /mnt/disks/persist; then
  sudo mount /dev/${persistent_device_name} /mnt/disks/persist  
fi


# Install miniconda for local bm run.
sudo -u bm-agent -i bash <<'EOF'
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

useradd -m -s /bin/bash bm-agent
sudo usermod -aG docker bm-agent

# Run the commands below as bm-agent user:
sudo -u bm-agent -i bash << EOBM
gcloud auth configure-docker ${region}-docker.pkg.dev --quiet
rm -rf bm-infra
git clone https://github.com/QiliangCui/bm-infra.git
pushd bm-infra
git pull
git reset --hard ${branch_hash}
popd
EOBM
cp /home/bm-agent/bm-infra/service/bm-agent/bm-agent.service /etc/systemd/system/bm-agent.service
systemctl stop bm-agent.service
systemctl daemon-reload
systemctl enable bm-agent.service
systemctl start bm-agent.service
