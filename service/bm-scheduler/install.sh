#!/bin/bash
set -euo pipefail

if ! id -u bm-scheduler >/dev/null 2>&1; then
    echo "sudo useradd -m -s /bin/bash bm-scheduler"
    sudo useradd -m -s /bin/bash bm-scheduler
fi

echo "sudo usermod -aG docker bm-scheduler"
sudo usermod -aG docker bm-scheduler

echo "sudo apt-get update && sudo apt-get install -y jq"
sudo apt-get update && sudo apt-get install -y jq

echo "sudo -u bm-scheduler -i..."
sudo -u bm-scheduler -i bash <<EOF
echo "gcloud auth configure-docker $GCP_REGION-docker.pkg.dev --quiet"
gcloud auth configure-docker $GCP_REGION-docker.pkg.dev --quiet

echo "rm -rf bm-infra"
rm -rf bm-infra

echo "machine github.com
login $GITHUB_USERNAME
password $GITHUB_PERSONAL_ACCESS_TOKEN" | tee -a /home/bm-scheduler/.netrc

echo "git clone https://github.com/QiliangCui/bm-infra.git"

if [[ -z "${BRANCH_NAME}" ]]; then
  export BRANCH_NAME=main
fi

git clone --branch "${branch_name}" https://github.com/QiliangCui/bm-infra.git

EOF

echo "sudo cp /home/bm-scheduler/bm-infra/service/bm-scheduler/bm-scheduler.service /etc/systemd/system/bm-scheduler.service"
sudo cp /home/bm-scheduler/bm-infra/service/bm-scheduler/bm-scheduler.service /etc/systemd/system/bm-scheduler.service

echo "sudo systemctl daemon-reload"
sudo systemctl daemon-reload

echo "sudo systemctl stop bm-scheduler.service"
sudo systemctl stop bm-scheduler.service

# add to crontab
(crontab -l 2>/dev/null | grep -v 'bm-scheduler.service'; echo "0 * * * * sudo /bin/systemctl restart bm-scheduler.service") | crontab -
