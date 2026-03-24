#!/bin/bash

SKIP_BUILD_IMAGE=1 ./scripts/scheduler/create_job.sh ./cases/tune_qwen480b_8k1k.csv "" "AD_HOC_QWEN480B_8K1K" "DAILY" "TPU_INFERENCE" "USE_BENCHMARK_SERVING=1;USE_BATCHED_RPA_KERNEL=1"
SKIP_BUILD_IMAGE=1 ./scripts/scheduler/create_job.sh ./cases/tune_qwen480b_8k1k.csv "" "AD_HOC_QWEN480B_8K1K" "DAILY" "TPU_INFERENCE" "USE_BENCHMARK_SERVING=1"