# Partner API validation kit

Scripts that implement the API-only validation plan against an OpenAI-
compatible inference endpoint. Each probe corresponds to a section of the
plan; results land as JSON under `out/`. A final report generator rolls
everything up into one markdown document for sharing.

## Layout

```
validation_kit/
├── README.md                       This file
├── run_all.sh                      End-to-end orchestrator
├── common.py                       OpenAI streaming client, timing, prompts, stats
├── probes/
│   ├── probe_cache.py              Response / prefix cache detection
│   ├── probe_spec_decoding.py      SD / MTP / Eagle / Medusa / Lookahead detection
│   ├── probe_roofline.py           HBM-bandwidth ceiling sanity check
│   ├── probe_precision_sleuth.py   Output divergence vs BF16 reference
│   ├── probe_kv_capacity.py        Advanced KV cache slots & concurrency limits
│   ├── probe_prefill_scaling.py    Advanced prefill chunk size scaling
│   ├── probe_scheduler_batching.py Advanced continuous batching verification
│   ├── probe_disaggregation.py     Advanced prefill-decode workload disaggregation
│   └── probe_spec_reverse.py       Advanced speculative drafting reverse-engineer
├── benchmarks/
│   ├── bench_pareto.py             Throughput-latency frontier sweep
│   ├── perturb_benchmarks.py       GSM-Symbolic-lite + MMLU answer permutations
│   ├── accuracy_harness.py         Standard + custom prompt accuracy
│   └── compare_bf16_fp4.py         Side-by-side BF16 vs FP4 accuracy + divergence
└── reports/
    └── generate_report.py          Markdown summary report grouped by severity
```

## Quick start

```bash
pip install httpx transformers

export PARTNER_BASE_URL=https://partner.example.com/v1
export PARTNER_API_KEY=sk-...
export PARTNER_MODEL=gpt-oss-120b
export PARTNER_TOKENIZER=openai/gpt-oss-120b   # for tokens-per-chunk

# Optional: BF16 reference for precision sleuthing + accuracy comparison
export REFERENCE_BASE_URL=http://your-vllm:8000/v1
export REFERENCE_MODEL=openai/gpt-oss-120b

./run_all.sh --chip-count 8 --hardware ironwood --precision default
```

Add `--skip-pareto` to skip the long throughput sweep, `--skip-accuracy`
to skip standards, `--skip-bf16-compare` to disable BF16-reference probes
even if the env vars are set.

After everything runs, **open `out/validation_report.md`** — that's the
boss-facing summary.

## Triage workflow

The probes are ordered from cheap-and-high-signal to expensive. A
typical workflow:

1. **Initial pass:** `./run_all.sh --chip-count N --skip-pareto --skip-accuracy`
   — under 10 minutes, catches cache / SD / impossible roofline numbers.
   Read `out/validation_report.md`. If CRITICAL findings appear, you have
   enough to push back on the partner before doing more.
2. **Full validation:** `./run_all.sh --chip-count N` — adds the Pareto
   sweep (~15 min) and accuracy. Re-read the report.
3. **With BF16 reference:** add `REFERENCE_BASE_URL` and `REFERENCE_MODEL`
   — adds precision sleuthing and BF16-vs-FP4 accuracy diff.

## What each probe catches

| Probe | What it catches | Wall time | Needs reference? |
|---|---|---|---|
| `probe_cache.py` | Response cache, prefix cache | ~2 min | No |
| `probe_spec_decoding.py` | SD, MTP, Eagle, Medusa, Lookahead | ~5 min | No |
| `probe_roofline.py` | Claims exceeding HBM-bandwidth ceiling | ~3 min | No |
| `probe_precision_sleuth.py` | Quantization aggressiveness; checkpoint mismatch | ~3 min | **Yes** |
| `probe_kv_capacity.py` | Advanced active KV cache slots & memory capacity limits | ~15 min | No |
| `probe_prefill_scaling.py` | Advanced prefill compute TTFT scaling & chunk detection | ~10 min | No |
| `probe_scheduler_batching.py` | Advanced iteration-level continuous batching check | ~5 min | No |
| `probe_disaggregation.py` | Advanced prefill-decode workload decoupling disaggregation | ~8 min | No |
| `probe_spec_reverse.py` | Advanced speculative drafting window reverse-engineer | ~8 min | No |
| `bench_pareto.py` | Throughput-latency frontier vs SLA | ~15 min | No |
| `accuracy_harness.py` | FP4 calibration overfit, surface-pattern matching | varies | No (uses just partner) |
| `compare_bf16_fp4.py` | FP4 vs BF16 accuracy gap; per-prompt regressions | varies | **Yes** |
| `generate_report.py` | Rolls everything into markdown | seconds | No |

## Configuration knobs that matter

**`--hardware`** sets the assumed HBM bandwidth for the roofline check.
Choices: `ironwood` (7.37 TB/s), `b200` (8 TB/s), `b300`, `h200`, `h100`,
`mi300x`, `mi355x`.

**`--precision`** sets the assumed bytes-per-token for gpt-oss-120b:
- `default` — 5.8 GB/tok. BF16 non-MoE + MXFP4 MoE + FP8 KV. Matches the model as released.
- `aggressive` — 3.8 GB/tok. FP8 attention/embed + MXFP4 MoE.
- `heroic` — 2.9 GB/tok. Uniform ~4-bit everywhere.

The roofline probe reports implied MBU at **all three** scenarios so you
see which precision the partner's numbers are consistent with. If they
claim default precision but the numbers only close at heroic, that's a
contradiction worth pursuing.

**`--chip-count`** is the partner's claimed chip count. Without it the
roofline analysis can't run.

## Probe details

### 1. Cache (probe_cache.py)

Three sub-tests; compares TTFT distributions:
- T1: identical request repeated N times. Response-cache hit → fast.
- T2: shared 700-token prefix, varying suffix. Prefix-cache hit → fast TTFT.
- T3: fully random prefix every request. No cache possible — baseline.

Verdicts on T3/T1 > 1.8 (response cache), T3/T2 > 1.5 (prefix cache),
or bimodal T1 distribution.

### 2. Spec decoding (probe_spec_decoding.py)

Three independent signals; any one is sufficient:
1. **Entropy ratio**: low-entropy "repeat this" vs high-temp random
   continuation at fixed I/O shapes. >1.5× decode tok/s ratio = SD/MTP.
2. **Tokens-per-chunk > 1**: tokenize each SSE chunk via `transformers`.
   Pure autoregressive = always 1.
3. **Sawtooth inter-chunk intervals**.

Pass `--no-tokenize` to skip the tokenizer dependency; entropy still works.

### 3. Roofline (probe_roofline.py)

Measures batch=1 decode tok/s with cache-busted random prompts, then
compares against `HBM_BW × chip_count / bytes_per_token`. Reports implied
MBU at all three precision scenarios. **MBU > 1.0 = physically impossible.**

### 4. Precision sleuthing (probe_precision_sleuth.py)

Sends same prompts to partner and to a BF16 reference at temperature=0.
Reports:
- Identical-output rate
- Mean longest common prefix (LCP) ratio
- First-divergence position

Strongest behavioral signal of actual quantization aggressiveness. Useful
contradiction check against the partner's stated precision.

### 5. KV Cache Slots & Capacity Sweep (probe_kv_capacity.py)

Fires concurrent streams with extremely large prompt sizes (e.g. 8k input / 256 output) scaling concurrency up to 128 to empirically map the serving framework's active HBM cache boundaries. Sudden, exponential degradation of decode speeds (`TPOT`) or request dropouts register active KV allocation limits or scheduling preemption bounds.

### 6. Prefill Compute Scaling (probe_prefill_scaling.py)

Measures Time-To-First-Token (`TTFT`) across sequential prompt sizes (128 up to 8k+) at single-user concurrency. Flat TTFT curves across large size increments capture active **Chunked Prefill** scheduling mechanisms, while stepwise linear latency climbs pinpoint the exact chunked prefill block bounds (e.g., 4,096 blocks).

### 7. Continuous Batching Verification (probe_scheduler_batching.py)

Launches a heavy, long-running background generation workload to fully occupy scheduling slots, then shoots high-priority short requests at millisecond intervals. Instantly returned short requests imply iteration-level continuous batching scheduler policies, whereas serialized waits confirm static batching configurations.

### 8. Workload Disaggregation & Jitter Analysis (probe_disaggregation.py)

Measures the standard deviation of active decode stream interval TPOT (jitter) during a quiet baseline compared to active periods featuring massive concurrent parallel prompt-prefill bursts. A jitter contention ratio approaching 1.00x confirms a Workload-Disaggregated architecture (physically separate prefill/decode node pools).

### 9. Speculative Drafting Reverse-Engineering (probe_spec_reverse.py)

Streams predictable, low-entropy sequences at `temperature=0.0` to evaluate multi-token SSE HTTP chunk bursts. Autoregressive streaming is strictly limited to 1 token per iteration. Observing single-chunk payloads containing multiple tokens reverse-engineers active speculative tree validation limits ($k = \text{max\_burst} - 1$).

### 10. Pareto sweep (bench_pareto.py)

Sweeps client-side concurrency over `{1, 4, 8, 16, 32, 64, 128}`, 90s per
point. Reports p50/p95/p99 TTFT and TPOT, aggregate and per-user tok/s,
and **useful throughput at fixed SLA** (p95 TPOT ≤ {50, 100, 300} ms).
The SLA numbers are what to compare against InferenceMAX B200 publications.

### 6. Accuracy (accuracy_harness.py + perturb_benchmarks.py)

Two paths:

**(a) Standard benchmarks** via `lm-evaluation-harness` — `accuracy_harness.py`
emits commands; you run them. Suite: MMLU, GSM8K-CoT, MMLU-Pro. Add
HumanEval+ via EvalPlus and LiveBench separately.

**(b) Perturbed + custom held-out** via `perturb_benchmarks.py`:
- `gsm8k_standard.jsonl` — original GSM8K
- `gsm8k_perturbed.jsonl` — integers multiplied by scale; answer rescaled
- `gsm8k_filler.jsonl` — original prompts with irrelevant prefix
- `mmlu_standard.jsonl` — original MMLU
- `mmlu_permuted.jsonl` — same content, shuffled answer-letter order

Then `accuracy_harness.py` runs them and reports gaps. Gap > 3% on any
perturbed set is a red flag.

```bash
python -m benchmarks.perturb_benchmarks \
    --gsm8k /path/to/gsm8k_test.jsonl \
    --mmlu /path/to/mmlu_test.jsonl \
    --limit 200 \
    --out-dir prompt_sets/

python -m benchmarks.accuracy_harness \
    --prompts prompt_sets/*.jsonl \
    --output-dir out/accuracy
```

### 7. BF16-vs-FP4 comparison (compare_bf16_fp4.py)

Runs the same prompt sets against both partner and reference endpoints,
reports:
- Accuracy on each
- Accuracy gap (BF16 − FP4)
- Cross-tabulation (both right / both wrong / FP4 regressed / FP4 recovered)
- Output-identity rate at temperature=0
- Per-prompt disagreement examples

Auto-detects "standard vs perturbed" pairs in the input filenames and
reports **differential degradation**: if FP4 degrades more on perturbed
than on standard, the quantization is likely benchmark-tuned.

### 8. Final report (reports/generate_report.py)

Reads all JSON outputs, classifies verdicts by severity (CRITICAL /
WARNING / INFO), and emits a single markdown report. Run automatically
as the last step of `run_all.sh`, or manually:

```bash
python -m reports.generate_report --out-dir out --output report.md
```

Severity rules:
- **CRITICAL** — "physically impossible," "multi-token decoding detected,"
  output divergence inconsistent with stated precision
- **WARNING** — "likely," borderline numbers, 1-3% gaps, partial signals
- **INFO** — no red flags

Output is structured so a non-technical reader can read just the
"Summary" and "CRITICAL findings" sections and understand the situation.

## Output schema

Every probe emits one JSON file under `out/` with at least:
- endpoint config (which run was which)
- raw per-request measurements
- summary statistics
- a `verdicts` array of human-readable findings

Verdicts are designed to be greppable:

```bash
grep -h '"verdicts"' out/*.json -A 20 | grep -E 'DETECTED|EXCEEDS|LIKELY'
```

But normally you just want `out/validation_report.md`.

## What this kit does NOT do

These need server-side access (Docker image or partner cooperation):

- TPU MFU, HBM utilization, duty cycle from the chip side
- XProf traces
- Verification of actual chip count, hardware generation, parallelism strategy
- Verification of actual quantization (we *infer* it via divergence + roofline math)
- Detection of disaggregated prefill/decode beyond gross latency patterns

The kit gives you the strongest possible signals from the client side. If
they're insufficient, that's itself a finding to escalate.
