#!/usr/bin/env bash
# Run the full API-only validation suite end-to-end.
#
# Required env vars:
#   PARTNER_BASE_URL  -- OpenAI-compatible endpoint URL ending in /v1
#   PARTNER_API_KEY   -- API key (or "dummy" if none required)
#   PARTNER_MODEL     -- model name string the endpoint expects
#
# Optional env vars (enables BF16 comparison probes):
#   REFERENCE_BASE_URL   -- your own BF16 vLLM (or similar) endpoint
#   REFERENCE_API_KEY    -- API key for the reference endpoint
#   REFERENCE_MODEL      -- model name the reference expects
#
# Required CLI args:
#   --chip-count N    -- number of chips the partner says they're using
#
# Optional CLI args:
#   --hardware        -- ironwood (default), b200, b300, h200, h100, mi300x, mi355x
#   --precision       -- default (5.8 GB/tok) | aggressive | heroic
#   --skip-pareto     -- skip the throughput-latency sweep (~15 min)
#   --skip-accuracy   -- skip the accuracy harness
#   --skip-bf16-compare -- skip BF16-vs-FP4 comparison even if REFERENCE_* is set
#
# Example with BF16 reference comparison:
#   export PARTNER_BASE_URL=https://partner.example.com/v1
#   export PARTNER_API_KEY=sk-...
#   export PARTNER_MODEL=gpt-oss-120b
#   export REFERENCE_BASE_URL=http://localhost:8000/v1
#   export REFERENCE_MODEL=openai/gpt-oss-120b
#   ./run_all.sh --chip-count 8 --hardware ironwood --precision default

set -euo pipefail

CHIP_COUNT=""
HARDWARE="ironwood"
PRECISION="default"
SKIP_PARETO=0
SKIP_ACCURACY=0
SKIP_BF16_COMPARE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chip-count) CHIP_COUNT="$2"; shift 2 ;;
    --hardware) HARDWARE="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --skip-pareto) SKIP_PARETO=1; shift ;;
    --skip-accuracy) SKIP_ACCURACY=1; shift ;;
    --skip-bf16-compare) SKIP_BF16_COMPARE=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$CHIP_COUNT" ]]; then
  echo "ERROR: --chip-count is required (the partner's claimed chip count)"
  exit 1
fi

: "${PARTNER_BASE_URL:?must be set}"
: "${PARTNER_API_KEY:?must be set}"
: "${PARTNER_MODEL:?must be set}"

HAS_REFERENCE=0
if [[ -n "${REFERENCE_BASE_URL:-}" && -n "${REFERENCE_MODEL:-}" ]]; then
  HAS_REFERENCE=1
  REFERENCE_API_KEY="${REFERENCE_API_KEY:-dummy}"
fi

mkdir -p out

echo "=========================================================="
echo " Partner API validation suite"
echo " Partner:    $PARTNER_BASE_URL ($PARTNER_MODEL)"
if [[ "$HAS_REFERENCE" -eq 1 ]]; then
  echo " Reference:  $REFERENCE_BASE_URL ($REFERENCE_MODEL)"
fi
echo " Hardware:   $HARDWARE (claim: $CHIP_COUNT chips, $PRECISION precision)"
echo "=========================================================="

# Tests are ordered: cheap and high-signal first.

echo
echo ">>> [1] Cache detection"
python -m probes.probe_cache --n 30 --output out/cache.json

echo
echo ">>> [2] Spec-decoding / MTP detection"
python -m probes.probe_spec_decoding --n 8 --max-tokens 512 \
    --output out/spec_decoding.json

echo
echo ">>> [3] Roofline / HBM-bandwidth ceiling"
python -m probes.probe_roofline \
    --hardware "$HARDWARE" \
    --chip-count "$CHIP_COUNT" \
    --precision "$PRECISION" \
    --input-tokens 8000 --output-tokens 1024 \
    --n 10 \
    --output out/roofline.json

if [[ "$HAS_REFERENCE" -eq 1 && "$SKIP_BF16_COMPARE" -eq 0 ]]; then
  echo
  echo ">>> [4] Precision sleuthing (vs BF16 reference)"
  python -m probes.probe_precision_sleuth \
      --partner-url "$PARTNER_BASE_URL" \
      --partner-key "$PARTNER_API_KEY" \
      --partner-model "$PARTNER_MODEL" \
      --reference-url "$REFERENCE_BASE_URL" \
      --reference-key "$REFERENCE_API_KEY" \
      --reference-model "$REFERENCE_MODEL" \
      --n 20 --max-tokens 256 \
      --output out/precision_sleuth.json
else
  echo
  echo ">>> [4] Precision sleuthing SKIPPED (no REFERENCE_* env vars or --skip-bf16-compare)"
fi

if [[ "$SKIP_PARETO" -eq 0 ]]; then
  echo
  echo ">>> [5] Throughput-latency Pareto sweep (long-running, ~15 min)"
  python -m benchmarks.bench_pareto \
      --concurrencies 1 4 8 16 32 64 128 \
      --duration 90 \
      --input-tokens 8000 --output-tokens 1024 \
      --output out/pareto.json
else
  echo
  echo ">>> [5] Pareto sweep SKIPPED"
fi

if [[ "$SKIP_ACCURACY" -eq 0 ]]; then
  echo
  echo ">>> [6] Accuracy harness (custom + standard-bench commands)"
  PROMPTS=()
  for f in prompt_sets/*.jsonl; do
    [[ -e "$f" ]] && PROMPTS+=("$f")
  done
  if [[ ${#PROMPTS[@]} -gt 0 ]]; then
    python -m benchmarks.accuracy_harness --prompts "${PROMPTS[@]}" \
        --output-dir out/accuracy
  else
    echo "    (no custom prompt sets in prompt_sets/; run perturb_benchmarks.py first)"
    python -m benchmarks.accuracy_harness --output-dir out/accuracy
  fi

  if [[ "$HAS_REFERENCE" -eq 1 && "$SKIP_BF16_COMPARE" -eq 0 && ${#PROMPTS[@]} -gt 0 ]]; then
    echo
    echo ">>> [6b] BF16 vs FP4 accuracy comparison"
    python -m benchmarks.compare_bf16_fp4 \
        --partner-url "$PARTNER_BASE_URL" \
        --partner-key "$PARTNER_API_KEY" \
        --partner-model "$PARTNER_MODEL" \
        --reference-url "$REFERENCE_BASE_URL" \
        --reference-key "$REFERENCE_API_KEY" \
        --reference-model "$REFERENCE_MODEL" \
        --prompts "${PROMPTS[@]}" \
        --output-dir out/compare
  fi
else
  echo
  echo ">>> [6] Accuracy SKIPPED"
fi

echo
echo ">>> [7] Generating final report"
python -m reports.generate_report --out-dir out --output out/validation_report.md

echo
echo "=========================================================="
echo " Done."
echo
echo "   FINAL REPORT (send to boss): out/validation_report.md"
echo
echo "   Raw probe outputs:"
echo "     out/cache.json              -- response/prefix cache verdicts"
echo "     out/spec_decoding.json      -- SD/MTP verdicts"
echo "     out/roofline.json           -- bandwidth-ceiling verdicts"
if [[ "$HAS_REFERENCE" -eq 1 ]]; then
  echo "     out/precision_sleuth.json   -- output-divergence verdicts"
  echo "     out/compare/                -- BF16 vs FP4 accuracy comparison"
fi
echo "     out/pareto.json             -- throughput-latency frontier"
echo "     out/accuracy/               -- accuracy harness output"
echo "=========================================================="
