#!/bin/bash     

# Set env vars system-wide
grep -q "^GCP_PROJECT_ID=" /etc/environment || echo "GCP_PROJECT_ID=${project_id}" | sudo tee -a /etc/environment
grep -q "^GCP_INSTANCE_ID=" /etc/environment || echo "GCP_INSTANCE_ID=${spanner_instance}" | sudo tee -a /etc/environment
grep -q "^GCP_DATABASE_ID=" /etc/environment || echo "GCP_DATABASE_ID=${spanner_db}" | sudo tee -a /etc/environment
grep -q "^GCP_REGION=" /etc/environment || echo "GCP_REGION=${region}" | sudo tee -a /etc/environment
grep -q "^GCP_INSTANCE_NAME=" /etc/environment || echo "GCP_INSTANCE_NAME=${instance_name}" | sudo tee -a /etc/environment
grep -q "^GCS_BUCKET=" /etc/environment || echo "GCS_BUCKET=${gcs_bucket}" | sudo tee -a /etc/environment
grep -q "^GCP_QUEUE=" /etc/environment || echo "GCP_QUEUE=vllm-${purpose}-queue-${accelerator_type}" | sudo tee -a /etc/environment
if ! grep -q "^HF_TOKEN=" /etc/environment; then
  gcloud secrets versions access latest --secret=bm-agent-hf-token --project=${project_id} --quiet | \
  sudo tee -a /etc/environment > /dev/null <<< "HF_TOKEN=$(cat)"
fi

apt-get update
apt-get install -y curl build-essential jq

curl -o- https://get.docker.com/ | bash -

# curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# /root/.cargo/bin/cargo install minijinja-cli
# cp /root/.cargo/bin/minijinja-cli /usr/bin/minijinja-cli
# chmod 777 /usr/bin/minijinja-cli

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
git checkout moe
# git pull
# git reset --hard ${branch_hash}
popd
EOBM
cp /home/bm-agent/bm-infra/service/bm-tune-agent/bm-tune-agent.service /etc/systemd/system/bm-tune-agent.service
systemctl stop bm-tune-agent.service
systemctl daemon-reload
systemctl enable bm-tune-agent.service
systemctl start bm-tune-agent.service
