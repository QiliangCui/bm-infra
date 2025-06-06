# bm-infra

```
export GCP_PROJECT_ID=cloud-tpu-inference-test
export GCP_INSTANCE_ID=vllm-bm-inst
export GCP_DATABASE_ID=vllm-bm-runs
export GCP_REGION=southamerica-west1
export GCS_BUCKET=vllm-cb-storage2
export GCS_INSTANCE_NAME=cuiq-infer-v6e-1-1
export GCP_QUEUE=vllm-bm-queue-v6e-1
export HF_TOKEN=<>
```

### Create and Delete Detabase

Create instance

```
gcloud spanner instances create $GCP_INSTANCE_ID \
   --description="vllm benchmark run record." \
   --nodes=1 \
   --project=$GCP_PROJECT_ID \
   --config=regional-$REGION
```

Create database

```
gcloud spanner databases create $GCP_DATABASE_ID \
 --project=$GCP_PROJECT_ID \
 --ddl-file=database/vllm_bm.sdl \
 --instance=$GCP_INSTANCE_ID 
```

Delete database

```
gcloud spanner databases delete $GCP_DATABASE_ID \
 --instance=$GCP_INSTANCE_ID \
 --project=$GCP_PROJECT_ID
```

### Create the Pub/sub queue.

Create pubsub

```
gcloud pubsub topics create vllm-bm-queue-v6e-1 \
  --project="$GCP_PROJECT_ID"

gcloud pubsub topics create vllm-bm-queue-v6e-8 \
  --project="$GCP_PROJECT_ID"

```

Delete pubsub top

```
gcloud pubsub topics delete vllm-bm-queue-v6e-1 \
 --project=YOUR_PROJECT_ID

gcloud pubsub topics delete vllm-bm-queue-v6e-8 \
 --project=YOUR_PROJECT_ID
```

### Test Run command

Insert runs to database

```
./scripts/insert_run.sh ./configs/case1.csv a408820f2fcdd4025f05f8a43dc15604fe534367
```

Trigger run by Record id

```
./script/run_job.sh 5b5040f7-c815-4a87-ab8e-54a49fd49916
```