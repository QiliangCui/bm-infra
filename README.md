# Benchmark Infrastructure and Tooling

## As a user to submit job to run on the agents

### Prepare enviornment

#### Insert following variables into /etc/environment.

```
echo 'GCP_PROJECT_ID=cloud-tpu-inference-test
GCP_INSTANCE_ID=vllm-bm-inst
GCP_DATABASE_ID=vllm-bm-runs
GCP_REGION=southamerica-west1
GCS_BUCKET=vllm-cb-storage2' | sudo tee -a /etc/environment
```

#### Docker setup

```
yes | gcloud auth configure-docker $GCP_REGION-docker.pkg.dev

sudo usermod -aG docker $USER

newgrp docker
```

#### Install jq for parsing json

```
sudo apt-get update && sudo apt-get install -y jq
```

### Submit a job to run.

1. login to gcp: `gcloud auth login`.
1. create a test case file like ./cases/case1.csv. Save it to a file like ~/my_test.csv
1. go to this source code folder.
1. Run `./scripts/scheduler/create_job.sh <INPUT_CSV_PATH> [CODE_HASH] [JOB_REFERENCE] [RUN_TYPE]`
   - INPUT_CSV_PATH: the test case file. This can either be a local filepath or a GCS storage URI (gs://\<path>)
   - CODE_HASH: the [vllm](https://github.com/vllm-project/vllm) code hash you want to run. use "" to indicate latest.
   - JOB_REFERENCE: A string that you can use later to find the job in database.
   - RUN_TYPE: default is "MANUAL". No need to set this usually.
   - REPO: which backend framework to use, default is using vLLM ("DEFAULT") but can also be "TPU_INFERENCE"
   - TPU_INFERENCE_TPU_BACKEND_TYPE: which TPU Inference TPU_BACKEND_TYPE to use -- can be "torchax" (default) or "jax"

Example:

```
./scripts/scheduler/create_job.sh ./configs/case1.csv

./scripts/scheduler/create_job.sh ~/my_test.csv da9b523ce1fd5c27bfd18921ba0388bf2e8e4618 my_first_test

./scripts/scheduler/create_job.sh gs://bm-infra/my_case.csv
```

To see job status

```
./scripts/manager/get_status.sh [JOB_REFERENCE]

For example
./scripts/manager/get_status.sh my_first_test

```

Write some script to query the database as `./scripts/manager/get_status.sh` or go to the spanner to query and see more result.

### Submit a job to run with dumping profile.

Use the command above with a "PROFILE=1" as ExtraEnv. For example,

```
./scripts/scheduler/create_job.sh cases/case1.csv "309c1bb82" cuiq-0804-xprof MANUAL DEFAULT "PROFILE=1"
```

After running, use command below to get the profile path on gcs

```
./scripts/manager/get_profile.sh cuiq-0804-xprof MANUAL

```

### Scan a range of vllm commits


Run `./scripts/manager/scan_commits.sh <INPUT_CSV> <START_HASH[-END_HASH]> [JOB_REFERENCE] [RUN_TYPE]`
- INPUT_CSV: test case csv.
- START_HASH: scan starting from this commits.
- END_HASH: scan till this commits(inclusive). If not provided, scan to latests.
- JOB_REFERENCE: job reference for searching the job later. The script will append a number after your JOB_REFERENCE. See example below
- RUN_TYPE: don't set it.

Example

```
# to scan between c8134bea15826876e37694834ad87d9c4bdfb26b and 3da2313d781f73c4b3b6bd57a130f85b7c0f0ca4
./scripts/manager/scan_commits.sh ~/my_test.csv c8134bea15826876e37694834ad87d9c4bdfb26b-3da2313d781f73c4b3b6bd57a130f85b7c0f0ca4 find_regression
```
The job reference will be like `find_regression_1`, `find_regression_2`... in the database. `find_regression_1` will be the first commit and `find_regression_2` will be the next commit.


## Manually Setup a BM Agent - The machine to query the queue and run jobs.

Not that the job will be run as a user "bm-agent" instead of yourself.

### Prepare enviornment

```
echo 'GCP_PROJECT_ID=cloud-tpu-inference-test
GCP_INSTANCE_ID=vllm-bm-inst
GCP_DATABASE_ID=vllm-bm-runs
GCP_REGION=southamerica-west1
GCS_BUCKET=vllm-cb-storage2
GCP_QUEUE=vllm-bm-queue-<debug-1, debug-2>
HF_TOKEN=<your hugging face token>
GCP_INSTANCE_NAME=<your instance name>
LOCAL_RUN_BM=<0:run with docker, 1: run with VM and conda, 2: run with VM and uv>
GITHUB_USERNAME=<user name - for only private repo>
GITHUB_PERSONAL_ACCESS_TOKEN=<access token - for only private repo>
'| sudo tee -a /etc/environment
```

Note: if you want to connect to "real" job queue, use the real device name like h100-8, v6e-8. But it means your machine will pull message from the real job. Usually, debug queue should be good enough for developing and debug.

### Attach a disk and mount to /mnt/disks/persist

```
# verify the mounted disk

mountpoint /mnt/disks/persist

```

### Install BM-Agent Service
If it is not a mounted disk, don't do following step.
Jobs will fail without a mounted disk.

Install the bm-agent service.

```
./service/bm-agent/install.sh
```

it installs a service bm-agent. It starts automatically to query the job queue and start to work on it.


Use the command below to control them.

```
# check status
sudo systemctl status bm-agent.service

# stop
sudo systemctl stop bm-agent.service

# disable so that it won't auto start.
sudo systemctl disable bm-agent.service

# see logs
sudo journalctl -u bm-agent -n 300 -f

```

## Deploy and Install everything with TF

### install terraform

```
# 1. Install required packages
sudo apt-get update && sudo apt-get install -y gnupg software-properties-common curl

# 2. Add HashiCorp GPG key
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg

# 3. Add the official HashiCorp Linux repo
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/hashicorp.list

# 4. Update and install Terraform
sudo apt-get update && sudo apt-get install terraform -y
```

### deploy

```
pushd terraform/gcp

terraform init

terraform plan

terraform apply

popd
```

### To increase or decrease the capacity

Change the machine number number in ./terraform/gcp/main.tf

## The Code Hash

Format is `A[_B]-[C]-[D]`.

A: The vllm repo hash.

B: If exists, it is the vllm main branch official head. If it exists, it indicate the A is a local commit.

C: TPU_INFERENCE branch hash.

D: Torch XLA branch hash.

## After creating new project

### Enable API

```
gcloud services enable spanner.googleapis.com --project=<new project>
gcloud services enable storage.googleapis.com --project=<new project>
gcloud services enable pubsub.googleapis.com --project=<new project>
gcloud services enable secretmanager.googleapis.com --project=<new project>
```

### Give permission to access spanner, pubsub and gcs

```
gcloud projects add-iam-policy-binding cloud-tpu-inference-test \
  --member="serviceAccount:<service-account>@developer.gserviceaccount.com" \
  --role="roles/storage.objectViewer" \
  --role="roles/pubsub.subscriber" \
  --role="roles/spanner.databaseUser"
```

### Give permission to GCS

```
gsutil iam ch \
  serviceAccount:<service-account>@developer.gserviceaccount.com:objectAdmin \
  gs://vllm-cb-storage2
```

### Give permission to access artifact registry

```
gcloud artifacts repositories add-iam-policy-binding vllm-tpu-bm \
  --location=southamerica-west1 \
  --project=cloud-tpu-inference-test \
  --member="serviceAccount:<service-account>@developer.gserviceaccount.com" \
  --role="roles/artifactregistry.reader"
```

### give permission to secrete

```
gcloud secrets add-iam-policy-binding bm-agent-hf-token \
  --member="serviceAccount:<service-account>@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```


