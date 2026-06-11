# RFC: Unified Inference Benchmarking Infrastructure

## Status

Draft

## Motivation

Today we have two generations of inference benchmarking infrastructure.

The v1 infrastructure in `bm-infra` is lightweight and easy to operate manually. A scheduler script runs on a GCE instance, reads benchmark cases, writes run state to Spanner, publishes work to Pub/Sub, and long-running TPU workers pull work and execute benchmark jobs. This path is useful for early testing and onboarding, but it is script-heavy and operationally hard to observe. Scheduler logs are mostly `echo` output on a GCE VM, so debugging often requires SSHing into the scheduler host.

The v2 infrastructure in this repo uses Buildkite for scheduling, dispatch, logs, and CI-style visibility. It introduces a more structured JSON benchmark case format and a reusable runner, but it is harder to use for one-off experiments if all users must first wire their repo into official Buildkite CI.

We want to unify the benchmark definition and execution path while keeping the useful operational properties:

- Buildkite visibility, history, queueing, retries, and logs.
- Lightweight one-off testing for personal development.
- A path for testing new repos before they have official CI.
- A repo/runtime-neutral benchmark runner that can support vLLM today and InferenceX or other inference stacks later.
- BigQuery-backed metrics for analytics and dashboarding.

## Goals

1. Use one benchmark job model across CI, scheduled benchmarks, and local/manual runs.
2. Make Buildkite the primary remote dispatcher.
3. Support direct execution on a TPU VM for local debugging.
4. Allow one-off personal/dev benchmark jobs through a separate Buildkite cluster/queue without using production CI capacity.
5. Support future inference runtimes beyond vLLM, including InferenceX.
6. Store benchmark metrics in BigQuery for analytics.
7. Store logs, profiles, and large artifacts in GCS and link to them from metric records.
8. Treat Pub/Sub/Spanner dispatch as legacy or compatibility infrastructure rather than the new primary architecture.
9. Keep repo-specific build, serve, and benchmark scripts owned by the target repo instead of centralizing every special case.

## Non-Goals

- Build a full benchmark UI in the first iteration.
- Replace Buildkite.
- Preserve Pub/Sub as a first-class new dispatcher if Buildkite dev clusters are sufficient.
- Require every one-off experiment to use a polished benchmark case format.
- Rewrite all existing shell scripts at once.
- Make `tpu-inference-dev` the generic multi-repo benchmark launcher.
- Move every repo's custom benchmark logic into the central runner repo.

## Current State

### v1: Pub/Sub and Spanner Dispatcher

The v1 system roughly works as:

```text
CSV/script scheduler on GCE
  -> insert RunRecord into Spanner
  -> publish Pub/Sub message with RecordId
  -> TPU worker pulls message
  -> worker reads Spanner
  -> worker runs benchmark scripts
  -> worker updates Spanner
```

Strengths:

- Easy to submit simple jobs.
- Does not require a Buildkite pipeline for each target repo.
- Useful for initial testing and onboarding.

Weaknesses:

- Scheduler observability is poor.
- Script/CSV-heavy configuration is hard to validate.
- Benchmark behavior can diverge from the Buildkite path.
- Spanner is not ideal for metric analytics.
- Running a worker pool duplicates functionality Buildkite already provides.

### v2: Buildkite Benchmarking

The v2 benchmark path roughly works as:

```text
JSON benchmark case
  -> Buildkite dynamic pipeline generation
  -> Buildkite TPU queue
  -> benchmark runner on TPU agent
  -> parse results
  -> upload logs/artifacts
  -> write results
```

Strengths:

- Structured benchmark cases.
- Better CI visibility and logs.
- Buildkite queueing and retry behavior.
- Easier scheduled jobs.

Weaknesses:

- Current dev pipeline is still tied to the `tpu-inference` repo.
- `tpu-inference-dev` is a sandbox pipeline, but it currently targets the same queues as normal CI unless changed.
- The runner and case model still contain vLLM-specific assumptions.

## Proposed Direction

Use Buildkite for remote dispatch and direct TPU execution for local/manual debugging.

The new architecture should have two supported execution paths:

```text
Remote/shared execution:
  Buildkite pipeline -> TPU Buildkite queue -> shared runner -> BigQuery/GCS

Local/direct execution:
  ssh to TPU head VM -> shared runner -> BigQuery/GCS or local output
```

Scheduled benchmarks should use native Buildkite scheduled builds:

```text
Buildkite schedule
  -> Buildkite benchmark pipeline
  -> TPU Buildkite queue
  -> shared runner
```

## Buildkite Cluster and Queue Model

We should separate official CI capacity from ad hoc/dev capacity.

```text
Production / CI cluster
  official tpu-inference CI
  scheduled/nightly benchmarks
  protected queues
  protected secrets
  production signal

Dev / benchmark-playground cluster
  branch-based experiments
  one-off benchmark jobs
  pre-onboarding for new repos
  separate agent token
  separate queues
  scoped secrets
```

This lets trusted users run arbitrary branch-specific Buildkite YAML or scripts without risking production CI queues or secrets.

Example queue names:

```text
tpu_v7x_2_dev_queue
tpu_v7x_8_dev_queue
tpu_v7x_16_dev_queue
```

For multi-host slices such as a future `tpu7x-16`, the desired model is one Buildkite executor per TPU slice:

```text
tpu7x-16 slice
  head VM: runs buildkite-agent
  worker VM(s): no Buildkite agent, controlled by head VM
```

The runner on the head VM can discover worker IPs and use SSH/Ray, similar to the existing `run_multihost.sh` flow.

## Pipeline Strategy

### Official `tpu-inference` Pipelines

Use these for reviewed and official signal:

```text
tpu-inference-ci
  PR/main validation

tpu-inference-benchmark
  scheduled benchmark cases

tpu-inference-dev
  tpu-inference-specific sandbox experiments
```

The existing `tpu-inference-dev` pipeline should remain a sandbox for this repo. It should not become the generic launcher for arbitrary repos.

### Generic Benchmark Playground Pipeline

For arbitrary personal development and pre-onboarding of new repos, create a separate generic pipeline tied to a small runner repo, for example:

```text
inference-benchmark-runner
```

This pipeline is tied to one repository because Buildkite pipelines require a repository, but it can be operationally multi-repo by accepting parameters:

```text
TARGET_REPO
TARGET_REF
TARGET_IMAGE
CASE_URI
RUNTIME
TPU_QUEUE
```

However, for trusted users and quick iteration, we do not need to over-design a generic arbitrary-command API immediately. A simple and acceptable workflow is:

```text
1. Create a branch in the runner repo.
2. Modify the branch's Buildkite YAML/script.
3. Trigger the benchmark-playground pipeline on that branch.
4. The branch-specific script runs on the dev Buildkite cluster/queue.
```

This keeps one-off experiments reproducible in git and avoids building a large submission API prematurely.

## Shared Benchmark Runner

The core benchmark runner should be repo-neutral. It may be incubated in this repo initially, but it should not permanently assume ownership by `tpu-inference`.

The central runner should standardize orchestration and reporting. It should not become a dumping ground for every repo's custom benchmark scripts.

Potential long-term homes:

- `bm-infra`, modernized
- a new `inference-benchmark-runner` repo
- a new `tpu-benchmark-infra` repo

The runner should eventually be a Python package with roughly this shape:

```text
benchmarking/
  case_spec.py
  validation.py
  runner.py
  report.py

  adapters/
    runtime_container.py
    runtime_repo_script.py
    runtime_existing_server.py
    runtime_vllm.py
    runtime_inferencex.py
    workload_openai_compatible.py
    workload_repo_script.py
    workload_vllm_bench.py
    workload_lm_eval.py

  storage/
    bigquery.py
    gcs.py
```

The runner should distinguish between dispatch and execution context. For example:

```bash
python -m bm_runner run \
  --case case.json \
  --target-image us-docker.pkg.dev/proj/inferencex:abc123 \
  --runtime inferencex \
  --execution-context buildkite
```

Here `--execution-context buildkite` means "this benchmark is currently running inside Buildkite, so collect Buildkite metadata and links." It does not mean "submit another Buildkite build."

The runner can also auto-detect Buildkite:

```text
BUILDKITE=true -> execution_context=buildkite
otherwise      -> execution_context=local
```

## Ownership and Extension Model

The central runner should own generic, stable capabilities:

- case schema and validation
- Buildkite pipeline generation helpers
- execution lifecycle
- environment and artifact layout
- BigQuery/GCS reporting
- common runtime adapters:
  - run a container
  - run a repo-provided script
  - connect to an existing server
  - launch stable shared runtimes such as vLLM or InferenceX
- common workload adapters:
  - OpenAI-compatible load generation
  - `vllm bench`
  - `lm_eval`
  - repo-provided workload script

Individual target repos should own repo-specific behavior:

- how to build their image
- how to install dependencies
- how to launch their server when it is not covered by a common adapter
- experimental flags and model-specific quirks
- custom benchmark scripts
- repo-specific benchmark cases

This avoids a central repo bottleneck where every new target repo must first add special code to the central runner.

The central runner should provide extension points instead of accumulating one-off adapters. The most important extension points are:

### `container` Runtime

Pull or use a target image, run a command, and wait for health.

```json
{
  "target": {
    "runtime": "container",
    "image": "us-docker.pkg.dev/proj/inferencex/server:abc123",
    "command": "inferencex serve --model Qwen/Qwen3-4B --port 8000",
    "health": {
      "type": "http",
      "url": "http://localhost:8000/health"
    }
  }
}
```

### `repo_script` Runtime

Clone a target repo/ref and run a script owned by that repo.

```json
{
  "target": {
    "runtime": "repo_script",
    "repo": "git@github.com:org/inferencex.git",
    "ref": "abc123",
    "script": "benchmarks/scripts/serve.sh"
  }
}
```

The central runner passes standard environment variables to the script, such as:

```text
BM_RUN_ID
BM_ARTIFACT_DIR
BM_LOG_DIR
BM_TARGET_REPO_DIR
BM_MODEL
BM_INPUT_LEN
BM_OUTPUT_LEN
BM_PORT
```

### `repo_script` Workload

Run a repo-owned benchmark client and require it to emit normalized result JSON.

```json
{
  "workload": {
    "type": "repo_script",
    "script": "benchmarks/scripts/run_workload.sh",
    "result_file": "artifacts/results.json"
  }
}
```

The result file contract should be intentionally small:

```json
{
  "status": "COMPLETED",
  "metrics": [
    {
      "name": "request_throughput",
      "value": 12.3,
      "unit": "req/s",
      "kind": "throughput",
      "aggregation": "mean"
    },
    {
      "name": "e2el",
      "value": 1234.0,
      "unit": "ms",
      "kind": "latency",
      "aggregation": "p99",
      "dimensions": {
        "phase": "end_to_end"
      }
    }
  ],
  "artifacts": [
    {
      "name": "server_log",
      "path": "artifacts/server.log",
      "type": "log"
    }
  ]
}
```

InferenceX may become a first-class central adapter if it is intended to be a stable, LLM-inference-generic interface. The bar for adding a central adapter should be that multiple repos or many cases benefit from the same abstraction. Repo-specific one-offs should remain repo scripts.

## Benchmark Job Model

The structured case format should describe a benchmark independently of the dispatcher.

Example:

```json
{
  "api_version": "benchmark/v1",
  "name": "inferencex-qwen3-smoke",
  "metadata": {
    "run_type": "dev",
    "owner": "inferencex",
    "tags": {
      "model_tag": "prod"
    }
  },
  "target": {
    "runtime": "inferencex",
    "args": {
      "model": "Qwen/Qwen3-4B",
      "max_model_len": 8192,
      "tensor_parallel_size": 8
    }
  },
  "workload": {
    "type": "openai_compatible_load",
    "args": {
      "input_len": 1024,
      "output_len": 128,
      "num_prompts": 1000,
      "request_rate": "auto"
    }
  },
  "expectations": {
    "p99_e2el_ms": 30000
  },
  "reporting": {
    "bigquery_table": "inference_benchmarking.benchmark_runs"
  }
}
```

One-off branch-based jobs do not need to start here. They can use custom scripts first and graduate into structured cases once the benchmark becomes repeatable.

## BigQuery Metrics Storage

Benchmark results should move to BigQuery.

Spanner is useful for strongly consistent operational state, but benchmark metrics are analytics-heavy:

- historical trends
- regression analysis
- comparisons across model, runtime, hardware, and commit
- dashboarding
- aggregation over many dimensions
- flexible metric additions

Use BigQuery for benchmark results and metrics, and GCS for logs, profiles, and raw artifacts.

### Proposed Tables

Prefer a small number of stable columns plus JSON fields for extensibility. BigQuery should be easy to query for common dashboards, but adding a new runtime, argument, metric dimension, or artifact should not require a schema migration.

`benchmark_runs`: one row per benchmark run.

```text
run_id STRING
job_reference STRING
run_type STRING
status STRING
created_at TIMESTAMP
started_at TIMESTAMP
finished_at TIMESTAMP

source_repo STRING
source_ref STRING
source_commit STRING
target_image STRING

execution_context STRING     -- buildkite, local
buildkite_org STRING
buildkite_pipeline STRING
buildkite_build_id STRING
buildkite_job_id STRING
buildkite_url STRING

device STRING
queue STRING
runtime STRING
workload_type STRING
model STRING
model_tag STRING

case_config JSON
run_config JSON          -- normalized config after defaults/device resolution
dimensions JSON          -- flexible tags: model family, backend, dtype, region, etc.
environment JSON         -- selected environment variables
source_metadata JSON     -- PR number, branch, author, image digest, etc.
artifacts JSON           -- log/profile/output URIs and metadata

created_by STRING
run_by STRING
error_message STRING
```

`benchmark_metrics`: one row per metric per run.

```text
run_id STRING
metric_name STRING
metric_value_float FLOAT64
metric_value_string STRING
metric_value_json JSON
metric_unit STRING
metric_kind STRING       -- latency, throughput, accuracy, resource, custom
aggregation STRING       -- p50, p90, p99, mean, max, raw
dimensions JSON          -- phase, dataset, batch size, request rate, etc.
metadata JSON            -- parser/version/source details
created_at TIMESTAMP
```

Numeric metrics should use `metric_value_float`. Non-numeric metrics can use `metric_value_string` or `metric_value_json`. This allows accuracy details, histograms, per-category results, and runtime-specific outputs without changing the table schema.

Optional `benchmark_events`: one row per lifecycle event. This is useful for debugging long runs without scraping logs.

```text
run_id STRING
event_time TIMESTAMP
event_type STRING        -- started, server_ready, workload_started, failed, etc.
severity STRING
message STRING
metadata JSON
```

Common dashboards should use BigQuery views that flatten selected metrics and dimensions:

```text
source_repo
source_commit
runtime
workload_type
model
device
queue
input_len
output_len
num_prompts
request_throughput
output_token_throughput
total_token_throughput
median_ttft_ms
p99_ttft_ms
median_tpot_ms
p99_tpot_ms
median_e2el_ms
p99_e2el_ms
accuracy
```

The flattened view can extract standard dimensions from `run_config` or `dimensions`, for example:

```sql
JSON_VALUE(run_config, '$.workload.args.input_len') AS input_len
```

This keeps the storage schema flexible while still giving analysts stable columns in views.

## Scheduling

Use Buildkite's native scheduled builds for benchmark cron jobs:

```text
Buildkite schedule
  -> Buildkite benchmark pipeline
  -> TPU Buildkite queue
  -> shared runner
```

Benefits:

- schedule history in Buildkite
- build history, logs, retries, and artifacts in one place
- no extra scheduler service to own
- no scheduler GCE VM
- no SSH for scheduler debugging

## Setup Guide

This section describes how to set up the proposed system in practical terms.

### 1. Set Up the Central Runner Repo

Create or choose a central repo, for example `inference-benchmark-runner`.

Recommended contents:

```text
bm_runner/
  cli.py
  case_spec.py
  runner.py
  report.py
  adapters/
  storage/

.buildkite/
  bootstrap.sh
  playground.yml
```

Expose stable CLI entry points:

```bash
bm validate CASE
bm run CASE
bm generate-buildkite CASE_OR_DIR
```

or module form:

```bash
python -m bm_runner validate CASE
python -m bm_runner run CASE
```

Keep the central runner versioned. Repos using structured cases should pin the runner version in a file or dependency declaration:

```text
.buildkite/bm_runner.version
```

Example:

```text
8f4c2d8
```

Then CI installs the exact runner version before validating cases:

```bash
BM_RUNNER_SHA="$(cat .buildkite/bm_runner.version)"
pip install "git+https://github.com/org/inference-benchmark-runner.git@${BM_RUNNER_SHA}"
python -m bm_runner validate .buildkite/benchmark/cases/**/*.json
```

During runner development, users can override with an editable checkout:

```bash
pip install -e ../inference-benchmark-runner
python -m bm_runner validate case.json
```

### 2. Set Up the Buildkite Dev Cluster and Queues

Create a separate Buildkite cluster for ad hoc/dev benchmarking.

Configure TPU VMs with the dev cluster's agent token and dev queue tags:

```ini
token="<dev cluster agent token>"
name="tpu-v7x-8-dev-%spawn"
tags="queue=tpu_v7x_8_dev_queue"
```

For single-host TPU resources, each TPU VM can run one Buildkite agent.

For multi-host TPU resources, only the head VM should run the Buildkite agent:

```text
tpu7x-16 slice
  head VM: queue=tpu_v7x_16_dev_queue
  worker VM(s): no Buildkite agent
```

The head VM runner is responsible for worker discovery and startup.

### 3. Create the Benchmark Playground Pipeline

Create a Buildkite pipeline such as:

```text
inference-benchmark-runner
```

Configure its static pipeline step to upload the branch's playground pipeline:

```yaml
steps:
  - label: ":pipeline: Upload benchmark playground"
    agents:
      queue: cpu
    command: "bash .buildkite/bootstrap.sh"
```

The bootstrap can be small:

```bash
#!/usr/bin/env bash
set -euo pipefail
buildkite-agent pipeline upload .buildkite/playground.yml
```

For one-off personal development:

```text
1. Create a branch in the runner repo.
2. Edit .buildkite/playground.yml or scripts called by it.
3. Trigger the Buildkite pipeline on that branch.
4. The branch's pipeline runs on dev TPU queues.
```

This is intentionally branch-based and git-backed. It avoids a large arbitrary-command submission API while keeping experiments reproducible.

### 4. Add Structured Cases to a Target Repo

A target repo that wants repeatable benchmark cases can add:

```text
benchmarks/
  cases/
    smoke.json
  scripts/
    build_image.sh
    serve.sh
    run_workload.sh
```

For `tpu-inference`, the existing location can remain:

```text
.buildkite/benchmark/cases/
```

The target repo should not duplicate central schema logic. It should call the central validator:

```bash
python -m bm_runner validate benchmarks/cases/smoke.json
```

If the target repo uses repo-owned scripts, the case should reference those scripts:

```json
{
  "target": {
    "runtime": "repo_script",
    "script": "benchmarks/scripts/serve.sh"
  },
  "workload": {
    "type": "repo_script",
    "script": "benchmarks/scripts/run_workload.sh",
    "result_file": "artifacts/results.json"
  }
}
```

The central runner clones/checks out the target repo, runs the scripts, reads the normalized result JSON, uploads artifacts, and writes BigQuery rows.

### 5. Run a One-Off Benchmark Before Repo Onboarding

Before a repo has official Buildkite setup, use the benchmark playground pipeline.

Typical branch-based flow:

```text
1. In the runner repo, create branch mh/test-inferencex.
2. Edit .buildkite/playground.yml to clone the target repo or use a target image.
3. Point the step at a dev TPU queue.
4. Trigger the Buildkite build on branch mh/test-inferencex.
```

Example playground step:

```yaml
steps:
  - label: "InferenceX smoke benchmark"
    agents:
      queue: tpu_v7x_8_dev_queue
    timeout_in_minutes: 240
    command: |
      git clone git@github.com:org/inferencex.git target
      git -C target checkout abc123
      cd target
      bash benchmarks/scripts/build_image.sh
      python -m bm_runner run benchmarks/cases/smoke.json \
        --execution-context buildkite
```

Once the benchmark becomes repeatable, move the case and scripts into the target repo and pin a central runner version.

### 6. Run Directly on a TPU VM

For local debugging:

```bash
ssh <tpu-head-vm>
git clone <target-repo>
cd <target-repo>
pip install "git+https://github.com/org/inference-benchmark-runner.git@<sha>"
python -m bm_runner validate benchmarks/cases/smoke.json
python -m bm_runner run benchmarks/cases/smoke.json --execution-context local
```

For multi-host runs, execute from the head VM. The runner or repo script should discover/control worker VMs.

### 7. Configure BigQuery and GCS

Create a BigQuery dataset, for example:

```text
inference_benchmarking
```

Create tables:

```text
benchmark_runs
benchmark_metrics
benchmark_events
```

Create a GCS bucket or prefix for artifacts:

```text
gs://inference-benchmark-artifacts/
```

Grant the Buildkite dev and production agent service accounts:

```text
roles/bigquery.dataEditor on the dataset
roles/bigquery.jobUser on the project
roles/storage.objectAdmin on the artifact bucket or prefix
```

The runner should write structured results to BigQuery and upload logs/profiles/raw outputs to GCS.

Example BigQuery DDL:

```sql
CREATE TABLE IF NOT EXISTS `PROJECT.inference_benchmarking.benchmark_runs` (
  run_id STRING NOT NULL,
  job_reference STRING,
  run_type STRING,
  status STRING,
  created_at TIMESTAMP,
  started_at TIMESTAMP,
  finished_at TIMESTAMP,

  source_repo STRING,
  source_ref STRING,
  source_commit STRING,
  target_image STRING,

  execution_context STRING,
  buildkite_org STRING,
  buildkite_pipeline STRING,
  buildkite_build_id STRING,
  buildkite_job_id STRING,
  buildkite_url STRING,

  device STRING,
  queue STRING,
  runtime STRING,
  workload_type STRING,
  model STRING,
  model_tag STRING,

  case_config JSON,
  run_config JSON,
  dimensions JSON,
  environment JSON,
  source_metadata JSON,
  artifacts JSON,

  created_by STRING,
  run_by STRING,
  error_message STRING
)
PARTITION BY DATE(created_at)
CLUSTER BY runtime, device, model, run_type;

CREATE TABLE IF NOT EXISTS `PROJECT.inference_benchmarking.benchmark_metrics` (
  run_id STRING NOT NULL,
  metric_name STRING NOT NULL,
  metric_value_float FLOAT64,
  metric_value_string STRING,
  metric_value_json JSON,
  metric_unit STRING,
  metric_kind STRING,
  aggregation STRING,
  dimensions JSON,
  metadata JSON,
  created_at TIMESTAMP
)
PARTITION BY DATE(created_at)
CLUSTER BY metric_name, metric_kind, aggregation;

CREATE TABLE IF NOT EXISTS `PROJECT.inference_benchmarking.benchmark_events` (
  run_id STRING NOT NULL,
  event_time TIMESTAMP,
  event_type STRING,
  severity STRING,
  message STRING,
  metadata JSON
)
PARTITION BY DATE(event_time)
CLUSTER BY event_type, severity;
```

## Alerting

Alerting should be split by signal type. Buildkite is the best source for execution failures, while BigQuery is the best source for benchmark result regressions and historical comparisons.

### Alert Sources

Use Buildkite notifications for job-level failures:

- pipeline failed
- step failed
- agent lost
- timeout
- infrastructure/setup failure

Use BigQuery-driven checks for metric-level regressions:

- throughput below threshold
- latency above threshold
- accuracy below threshold
- missing metrics
- unexpected status distribution
- sustained regression compared with historical baseline

Use GCS/BigQuery links in alerts so users can jump directly to:

- Buildkite build/job URL
- server log URI
- workload/client log URI
- profile URI
- BigQuery run row

### Alert Policy

Do not page on every ad hoc/dev benchmark failure. Dev and playground jobs are expected to fail during experimentation.

Recommended policy:

```text
production CI failure
  -> notify owning team / oncall based on existing Buildkite policy

scheduled official benchmark failure
  -> notify benchmark owners
  -> page only if the benchmark is release-blocking or explicitly marked critical

metric regression in official benchmark
  -> notify benchmark owners
  -> page only after persistence threshold is met

dev/playground benchmark failure
  -> notify build creator only
```

The case config should support alert metadata:

```json
{
  "alerting": {
    "enabled": true,
    "severity": "warning",
    "owners": ["tpu-inference-benchmarks"],
    "channels": ["slack:vllm#tpu-ci-notifications"],
    "page": false,
    "rules": [
      {
        "metric": "request_throughput",
        "aggregation": "mean",
        "operator": "<",
        "threshold": 10.0,
        "unit": "req/s"
      },
      {
        "metric": "e2el",
        "aggregation": "p99",
        "operator": ">",
        "threshold": 30000,
        "unit": "ms"
      }
    ]
  }
}
```

### Regression Rules

Support both static thresholds and baseline comparisons.

Static threshold example:

```text
p99_e2el_ms <= 30000
request_throughput >= 10
accuracy >= 0.75
```

Baseline comparison example:

```text
current request_throughput must be within 95% of trailing 7-day median
current p99_e2el must be within 110% of trailing 7-day median
```

For noisy benchmarks, alerts should require persistence:

```text
alert only if 2 of the last 3 scheduled runs fail the rule
```

This avoids alerting on a single flaky run.

### Alert Evaluation

Initial implementation can be simple:

```text
Buildkite scheduled benchmark
  -> runner writes BigQuery rows
  -> final Buildkite step runs alert evaluation query
  -> step posts Slack/email notification or fails the build
```

Longer-term implementation can move alert evaluation to a scheduled query or small service if needed, but that is not required for the first version.

Example BigQuery query shape:

```sql
WITH latest AS (
  SELECT
    r.run_id,
    r.buildkite_url,
    m.metric_name,
    m.aggregation,
    m.metric_value_float
  FROM `PROJECT.inference_benchmarking.benchmark_runs` r
  JOIN `PROJECT.inference_benchmarking.benchmark_metrics` m
    USING (run_id)
  WHERE r.run_id = @run_id
)
SELECT *
FROM latest
WHERE metric_name = 'request_throughput'
  AND aggregation = 'mean'
  AND metric_value_float < 10.0;
```

If the query returns rows, the alert evaluator can mark the benchmark as regressed and include the returned metric details in the notification.

### Alert State

Alert state can initially live in BigQuery:

```text
benchmark_alerts
  alert_id STRING
  run_id STRING
  rule_id STRING
  severity STRING
  status STRING            -- open, acknowledged, resolved
  created_at TIMESTAMP
  resolved_at TIMESTAMP
  message STRING
  metadata JSON
```

This table is optional for the first implementation, but useful once we want deduplication, persistence thresholds, or dashboards for open regressions.

## Pub/Sub and Spanner

Pub/Sub and Spanner should not be part of the new primary architecture if the dev Buildkite cluster/queue workflow is sufficient.

Keep Pub/Sub/Spanner only as:

- legacy compatibility during migration
- fallback if Buildkite cannot support a required non-CI worker-pool use case
- temporary bridge for existing v1 jobs

The main use case previously served by Pub/Sub was remote job dispatch to TPU workers without Buildkite. A separate Buildkite dev cluster and branch-based dynamic pipelines should cover that use case with better logs, history, permissions, and queueing.

## Migration Plan

### Phase 1: Buildkite Dev Isolation

- Create or identify a separate Buildkite cluster for dev/ad hoc benchmarking.
- Create dedicated dev TPU queues.
- Ensure dev queues use separate agent tokens and appropriately scoped secrets.
- Avoid using production CI queues by default for branch-based experiments.

### Phase 2: Generic Benchmark Runner Pipeline

- Create a small `inference-benchmark-runner` or equivalent repo/pipeline.
- Configure the pipeline to run branch-specific Buildkite YAML/scripts.
- Support basic branch-based one-off jobs on dev queues.
- Document how users trigger a build for personal development or new repo onboarding.

### Phase 3: Shared Runner Library

- Extract the reusable parts of the current benchmark runner.
- Keep vLLM support as the first adapter.
- Add `container`, `repo_script`, and OpenAI-compatible workload adapters before adding many repo-specific adapters.
- Add explicit execution context handling for Buildkite and local runs.
- Preserve direct TPU execution for local debugging.

### Phase 4: BigQuery Reporting

- Add a BigQuery writer.
- Write BigQuery metrics from Buildkite and local runner paths.
- Upload logs/profiles/artifacts to GCS.
- Add flexible `JSON` fields for config, dimensions, artifacts, metadata, and non-scalar metric values.
- Add compatibility views for common dashboard queries.

### Phase 5: Runtime Expansion

- Add InferenceX runtime/workload support.
- Validate that InferenceX can be run through:
  - branch-based dev Buildkite jobs
  - prebuilt image jobs
  - structured benchmark cases

### Phase 6: Pub/Sub Deprecation Decision

- Audit remaining v1 Pub/Sub users.
- Decide whether to keep it as legacy only, remove it, or keep a compatibility shim.
- Avoid adding new features to the Pub/Sub path unless a Buildkite cluster cannot satisfy the need.

## Open Questions

1. Which repo should own the long-term shared runner?
2. Which Buildkite cluster should host dev/ad hoc benchmark jobs?
3. What secrets should be available in the dev benchmark cluster?
4. Do we need `tpu7x-16` dev queues, and if so, how do we ensure only the head VM runs the Buildkite agent?
5. Are Buildkite UI/API triggers enough for manual submissions?
6. How long do we dual-write or preserve Spanner for existing dashboards?
7. What is the minimum InferenceX adapter contract, and should it be central or repo-owned initially?
8. What exact result JSON contract should repo-owned workload scripts emit?

## Recommendation

Adopt Buildkite as the primary remote dispatcher and direct TPU execution as the local/manual path.

Create a separate Buildkite dev or benchmark-playground cluster for branch-based one-off jobs and pre-onboarding of new repos. This replaces the main value of Pub/Sub while providing better logs, history, permissions, queueing, and operational visibility.

Use Buildkite scheduled builds for recurring benchmark runs.

Move benchmark metrics to BigQuery and store logs/artifacts in GCS.

Treat Pub/Sub/Spanner dispatch as legacy compatibility infrastructure unless a concrete non-Buildkite worker-pool requirement remains.
