#!/bin/bash
#
# This script is used to add the results of gpu runs into spanner.
# Before adding check if the data for GPU is per chip or per VM. Based on that you need to multiply. Here we are taking per VM numbers.

set -euo pipefail

current_datetime=$(date +"%Y%m%d-%H%M%S")
RECORD_ID="manual-addition-${current_datetime}"
MODEL="meta-llama/Llama-3.1-8B-Instruct"
MAX_MODEL_LEN=5120
INPUT_LEN=1024
OUTPUT_LEN=4096
PREFIX_LEN=0
OUTPUT_TOKEN_THROUGHPUT=<Number>


RUN_TYPE="ONETIME"
MAX_NUM_SEQS=1024
MAX_NUM_BATCHED_TOKENS=4096
TENSOR_PARALLEL_SIZE=1
JOB_REFERENCE=$(date +"%Y%m%d_%H%M%S")
EXTRA_ENVS=""
ADDITIONAL_CONFIG='{"quantization": "fp8"}'
VERY_LARGE_EXPECTED_ETEL=3600000
CODEHASH="UNKNOWN"
DATASET="UNKNOWN"
DEVICE="gb200-1"

echo "gcloud spanner databases execute-sql $GCP_DATABASE_ID \
    --instance=$GCP_INSTANCE_ID \
    --project=$GCP_PROJECT_ID \
    --sql=\"INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen, Dataset,
      InputLen, OutputLen, LastUpdate, CreatedBy,JobReference, ExpectedETEL, NumPrompts, ModelTag, PrefixLen, 
      OutputTokenThroughput, ExtraEnvs, AdditionalConfig 
    ) VALUES (
      '$RECORD_ID', 'COMPLETED', PENDING_COMMIT_TIMESTAMP(), '$DEVICE', '$MODEL', '$RUN_TYPE', '$CODEHASH',
      $MAX_NUM_SEQS,
      $MAX_NUM_BATCHED_TOKENS,
      $TENSOR_PARALLEL_SIZE,
      $MAX_MODEL_LEN,
      '$DATASET',
      $INPUT_LEN,
      $OUTPUT_LEN,
      PENDING_COMMIT_TIMESTAMP(),
      '$USER',
      '$JOB_REFERENCE',
      ${EXPECTED_ETEL:-$VERY_LARGE_EXPECTED_ETEL},
      ${NUM_PROMPTS:-1000},
      '${MODELTAG:-PROD}',
      ${PREFIX_LEN:-0},
      $OUTPUT_TOKEN_THROUGHPUT,
      '$EXTRA_ENVS',
      '$ADDITIONAL_CONFIG'
    );\""

eval "gcloud spanner databases execute-sql $GCP_DATABASE_ID \
    --instance=$GCP_INSTANCE_ID \
    --project=$GCP_PROJECT_ID \
    --sql=\"INSERT INTO RunRecord (
      RecordId, Status, CreatedTime, Device, Model, RunType, CodeHash,
      MaxNumSeqs, MaxNumBatchedTokens, TensorParallelSize, MaxModelLen, Dataset,
      InputLen, OutputLen, LastUpdate, CreatedBy,JobReference, ExpectedETEL, NumPrompts, ModelTag, PrefixLen, 
      OutputTokenThroughput, ExtraEnvs, AdditionalConfig 
    ) VALUES (
      '$RECORD_ID', 'COMPLETED', PENDING_COMMIT_TIMESTAMP(), '$DEVICE', '$MODEL', '$RUN_TYPE', '$CODEHASH',
      $MAX_NUM_SEQS,
      $MAX_NUM_BATCHED_TOKENS,
      $TENSOR_PARALLEL_SIZE,
      $MAX_MODEL_LEN,
      '$DATASET',
      $INPUT_LEN,
      $OUTPUT_LEN,
      PENDING_COMMIT_TIMESTAMP(),
      '$USER',
      '$JOB_REFERENCE',
      ${EXPECTED_ETEL:-$VERY_LARGE_EXPECTED_ETEL},
      ${NUM_PROMPTS:-1000},
      '${MODELTAG:-PROD}',
      ${PREFIX_LEN:-0},
      $OUTPUT_TOKEN_THROUGHPUT,
      '$EXTRA_ENVS',
      '$ADDITIONAL_CONFIG'
    );\""

if [ $? -ne 0 ]; then
    echo "Failure to add data entry to spanner db."
fi

echo "Script Ended"