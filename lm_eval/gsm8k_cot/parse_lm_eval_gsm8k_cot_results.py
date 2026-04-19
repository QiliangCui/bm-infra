import argparse
import json
import os
import sys


def parse_gsm8k_cot_results(input_file):
    """
    Parses the raw results from an lm_eval gsm8k_cot run. Prints a
    machine-readable JSON object to stdout for automation and a human-readable
    summary to stderr. The aggregate score is exposed as 'gsm8k_cot_agg';
    individual metrics are prefixed with 'gsm8k_cot_'.
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_file}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}", file=sys.stderr)
        sys.exit(1)

    results = data.get("results", {})
    task_metrics = results.get("gsm8k_cot", {})

    summary = {}

    # lm_eval reports gsm8k_cot with keys like "exact_match,strict-match"
    # and "exact_match,flexible-extract". Prefer flexible-extract as the headline
    # since strict-match often fails on reasoning-style output (e.g. Qwen3 thinking).
    flexible_key = "exact_match,flexible-extract"
    strict_key = "exact_match,strict-match"

    if flexible_key in task_metrics:
        summary["gsm8k_cot_agg"] = task_metrics[flexible_key]
        summary["gsm8k_cot_flexible_extract"] = task_metrics[flexible_key]
    if strict_key in task_metrics:
        summary["gsm8k_cot_strict_match"] = task_metrics[strict_key]
        # Fall back to strict if flexible is missing
        if "gsm8k_cot_agg" not in summary:
            summary["gsm8k_cot_agg"] = task_metrics[strict_key]

    print(json.dumps(summary))

    print("\n--- GSM8K-CoT Results Summary ---", file=sys.stderr)
    print(f"File: {os.path.basename(input_file)}", file=sys.stderr)
    print("-" * 30, file=sys.stderr)
    if "gsm8k_cot_agg" in summary:
        print(f"Overall GSM8K-CoT Accuracy: {summary['gsm8k_cot_agg']:.4f}", file=sys.stderr)
        print("-" * 30, file=sys.stderr)
    for k, v in sorted(summary.items()):
        if k != "gsm8k_cot_agg":
            print(f"- {k}: {v:.4f}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse lm_eval gsm8k_cot results.")
    parser.add_argument("input_file", type=str, help="Path to the input JSON file from lm_eval.")
    args = parser.parse_args()

    parse_gsm8k_cot_results(args.input_file)
