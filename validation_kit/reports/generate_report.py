"""
Final report generator.

Reads every JSON file produced by the kit (under out/) and emits a single
markdown report with verdicts grouped by severity, ready to send to your
boss.

Severity grouping:
  CRITICAL  -- physically impossible claims, multi-token decoding detected,
               output divergence inconsistent with claimed precision
  WARNING   -- borderline numbers, accuracy gaps in 1-3% range, partial
               signals
  INFO      -- no red flags but the test ran, useful as context

Usage:
  python -m reports.generate_report --out-dir out/ --output validation_report.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

# Patterns in verdict text -> severity. Earlier matches win.
SEVERITY_PATTERNS = [
    # CRITICAL
    ("CRITICAL", [
        "EXCEEDS PHYSICAL CEILING",
        "physically impossible",
        "not physically achievable",
        "DETECTED",
        "physically impossible without",
    ]),
    # WARNING
    ("WARNING", [
        "LIKELY",
        "BIMODAL",
        "BORDERLINE",
        "heroic",
        "Exceeds 3 pp",
        "degrades MORE",
        "diverge rapidly",
        "differential",
        "TOKENS-PER-CHUNK",
        "implied MBU = 0.9",
        "implied MBU = 0.8",
    ]),
    # INFO (default)
]


def classify_severity(verdict: str) -> str:
    for severity, patterns in SEVERITY_PATTERNS:
        if any(p in verdict for p in patterns):
            return severity
    return "INFO"


# ---------------------------------------------------------------------------
# JSON loaders for each probe's output schema
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------

def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x*100:.1f}%"


def fmt_num(x: float | None, places: int = 2) -> str:
    if x is None or x != x:  # NaN check
        return "—"
    return f"{x:.{places}f}"


def section_header(level: int, text: str) -> str:
    return f"\n{'#' * level} {text}\n"


def render_cache(data: dict | None) -> tuple[list[str], list[str]]:
    """Returns (lines for body, list of verdicts)."""
    lines: list[str] = []
    if data is None:
        lines.append("_Cache probe did not run._")
        return lines, []

    summ = data.get("summary", {})
    rows = []
    for tag in ("identical", "shared_prefix", "random_prefix"):
        s = summ.get(tag, {})
        if not s:
            continue
        rows.append(
            f"| {tag} | {s.get('n','—')} | {fmt_num(s.get('mean'))} | "
            f"{fmt_num(s.get('p50'))} | {fmt_num(s.get('p95'))} | "
            f"{fmt_num(s.get('min'))} | {fmt_num(s.get('max'))} |"
        )

    lines.append("**TTFT distribution by sub-test (seconds):**\n")
    lines.append("| sub-test | n | mean | p50 | p95 | min | max |")
    lines.append("|---|---|---|---|---|---|---|")
    lines.extend(rows)
    lines.append("")

    # Interpretation
    s_id = summ.get("identical", {}).get("mean", 0)
    s_rand = summ.get("random_prefix", {}).get("mean", 0)
    if s_id and s_rand:
        ratio = s_rand / s_id
        lines.append(
            f"**Random-prefix / identical TTFT ratio:** {ratio:.2f}× "
            f"(>1.8× indicates response cache; >1.5× shared/random ratio indicates prefix cache)"
        )
        lines.append("")

    return lines, data.get("verdicts", [])


def render_spec(data: dict | None) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if data is None:
        lines.append("_Spec-decoding probe did not run._")
        return lines, []

    low_tps = data.get("low_decode_tps", [])
    high_tps = data.get("high_decode_tps", [])
    if low_tps and high_tps:
        low_mean = sum(low_tps) / len(low_tps)
        high_mean = sum(high_tps) / len(high_tps)
        ratio = low_mean / high_mean if high_mean > 0 else float("inf")
        lines.append("**Entropy probe (decode tok/s/user):**\n")
        lines.append(f"- Low-entropy mean: **{low_mean:.1f}** tok/s/user")
        lines.append(f"- High-entropy mean: **{high_mean:.1f}** tok/s/user")
        lines.append(f"- Ratio (low / high): **{ratio:.2f}×** (>1.5× = multi-token decoding active)")
        lines.append("")

    # Tokens per chunk
    low_analyses = data.get("low_analyses", [])
    if low_analyses:
        max_tpcs = [
            max(la.get("tokens_per_chunk", [1]) or [1])
            for la in low_analyses
        ]
        if max_tpcs:
            tpc_max = max(max_tpcs)
            lines.append(
                f"**Tokens-per-chunk (low-entropy runs):** max observed = "
                f"**{tpc_max}** "
                f"(>1 = direct evidence of multi-token decoding)"
            )
            lines.append("")

    return lines, data.get("verdicts", [])


def render_roofline(data: dict | None) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if data is None:
        lines.append("_Roofline probe did not run._")
        return lines, []

    lines.append(
        f"**Configuration:** {data.get('chip_count', '?')}× "
        f"{data.get('hardware', '?')}, "
        f"precision assumption: `{data.get('precision', '?')}`"
    )
    lines.append("")

    stats = data.get("stats", {})
    if stats:
        lines.append("**Measured batch=1 decode tok/s/user:**")
        lines.append("")
        lines.append(
            f"- n={stats.get('n','—')}  "
            f"mean=**{fmt_num(stats.get('mean'), 1)}**  "
            f"p50={fmt_num(stats.get('p50'), 1)}  "
            f"p95={fmt_num(stats.get('p95'), 1)}  "
            f"max={fmt_num(stats.get('max'), 1)}"
        )
        lines.append("")

    implied = data.get("implied_mbu", {})
    if implied:
        lines.append("**Implied MBU at each precision assumption:**\n")
        lines.append("| precision | implied MBU | verdict |")
        lines.append("|---|---|---|")
        for prec in ("default", "aggressive", "heroic"):
            mbu = implied.get(prec)
            if mbu is None:
                continue
            if mbu > 1.0:
                verdict = "❌ physically impossible"
            elif mbu > 0.8:
                verdict = "⚠️ heroic, demand evidence"
            else:
                verdict = "✓ plausible"
            lines.append(f"| {prec} | {mbu:.2f} | {verdict} |")
        lines.append("")

    return lines, data.get("verdicts", [])


def render_pareto(data: dict | None) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if data is None:
        lines.append("_Pareto sweep did not run._")
        return lines, []

    lines.append(
        f"**Shape:** {data.get('input_tokens','?')}-token input / "
        f"{data.get('output_tokens','?')}-token output. "
        f"Sustained {data.get('duration_s_per_point','?')}s per concurrency."
    )
    lines.append("")

    sweep = data.get("sweep", [])
    if sweep:
        lines.append("**Throughput / latency by concurrency:**\n")
        lines.append(
            "| concurrency | agg tok/s | per-user dec tok/s | p50 TPOT (ms) | "
            "p95 TPOT (ms) | p99 TPOT (ms) | p95 TTFT (ms) |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for p in sweep:
            tpot_p50 = p.get("tpot_stats", {}).get("p50", 0) * 1000
            tpot_p95 = p.get("tpot_stats", {}).get("p95", 0) * 1000
            tpot_p99 = p.get("tpot_stats", {}).get("p99", 0) * 1000
            ttft_p95 = p.get("ttft_stats", {}).get("p95", 0) * 1000
            lines.append(
                f"| {p.get('concurrency','—')} | "
                f"{p.get('aggregate_tok_per_s', 0):.0f} | "
                f"{p.get('mean_per_user_decode_tps', 0):.1f} | "
                f"{tpot_p50:.0f} | {tpot_p95:.0f} | {tpot_p99:.0f} | "
                f"{ttft_p95:.0f} |"
            )
        lines.append("")

    slas = data.get("sla_summaries", {})
    if slas:
        lines.append("**Useful throughput at SLA:**\n")
        for sla, summary in slas.items():
            if summary is None:
                lines.append(f"- {sla}: NO concurrency met the SLA")
            else:
                lines.append(
                    f"- {sla}: **{summary['max_aggregate_tok_per_s']:.0f} "
                    f"tok/s aggregate** at concurrency "
                    f"{summary['concurrency']} "
                    f"(per-user decode: {summary['mean_per_user_decode_tps']:.1f} tok/s)"
                )
        lines.append("")

    return lines, []  # Pareto doesn't emit verdicts itself


def render_precision_sleuth(data: dict | None) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if data is None:
        lines.append("_Precision-sleuthing probe did not run._")
        return lines, []

    summary = data.get("summary", {})
    lines.append("**Output divergence at temperature=0:**\n")
    lines.append(
        f"- Identical-output rate: **{fmt_pct(summary.get('identity_rate'))}**"
    )
    lcp = summary.get("lcp_ratio_stats", {})
    lines.append(
        f"- Mean longest common prefix (ratio): **{fmt_pct(lcp.get('mean'))}** "
        f"(p50: {fmt_pct(lcp.get('p50'))}, p95: {fmt_pct(lcp.get('p95'))})"
    )
    diverge = summary.get("first_diverge_char_stats", {})
    if diverge:
        lines.append(
            f"- Mean first-divergence position: char "
            f"{fmt_num(diverge.get('mean'), 0)}"
        )
    lines.append("")

    return lines, data.get("verdicts", [])


def render_compare(data: dict | None) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    if data is None:
        lines.append("_BF16-vs-FP4 comparison did not run._")
        return lines, []

    per_set = data.get("per_set", {})
    if per_set:
        lines.append("**Accuracy comparison (BF16 reference vs FP4 partner):**\n")
        lines.append(
            "| prompt set | n | BF16 acc | FP4 acc | gap | "
            "identical outputs | FP4 regressed |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for name, s in per_set.items():
            tab = s.get("cross_tab", {})
            ident_rate = tab.get("output_identity_rate", 0)
            lines.append(
                f"| {name} | {s.get('n','—')} | "
                f"{fmt_num(s.get('bf16_accuracy'), 4)} | "
                f"{fmt_num(s.get('fp4_accuracy'), 4)} | "
                f"{s.get('gap_bf16_minus_fp4', 0):+.4f} | "
                f"{fmt_pct(ident_rate)} | "
                f"{tab.get('fp4_regressed', '—')} |"
            )
        lines.append("")

    return lines, data.get("verdicts", [])


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def build_report(out_dir: Path) -> str:
    """Read all JSON files from out_dir and assemble report."""
    cache = load_json(out_dir / "cache.json")
    spec = load_json(out_dir / "spec_decoding.json")
    roofline = load_json(out_dir / "roofline.json")
    pareto = load_json(out_dir / "pareto.json")
    sleuth = load_json(out_dir / "precision_sleuth.json")
    compare = load_json(out_dir / "compare" / "compare_summary.json")

    all_sections: list[tuple[str, list[str], list[str]]] = []  # (title, body_lines, verdicts)

    body, v = render_cache(cache)
    all_sections.append(("Cache detection", body, v))
    body, v = render_spec(spec)
    all_sections.append(("Multi-token decoding (SD / MTP / Eagle / Medusa)", body, v))
    body, v = render_roofline(roofline)
    all_sections.append(("Roofline / HBM-bandwidth ceiling", body, v))
    body, v = render_pareto(pareto)
    all_sections.append(("Throughput / latency Pareto frontier", body, v))
    body, v = render_precision_sleuth(sleuth)
    all_sections.append(("Precision sleuthing (output divergence)", body, v))
    body, v = render_compare(compare)
    all_sections.append(("BF16 vs FP4 accuracy comparison", body, v))

    # Severity-classified verdicts
    by_severity: dict[str, list[tuple[str, str]]] = {
        "CRITICAL": [], "WARNING": [], "INFO": []
    }
    for title, _body, verdicts in all_sections:
        for v in verdicts:
            sev = classify_severity(v)
            by_severity[sev].append((title, v))

    # Compose the report
    out: list[str] = []
    out.append("# Partner endpoint validation report\n")
    out.append("Generated by the API-only validation kit. All numbers below "
               "are measured client-side; no server-side observability was "
               "available. Read CRITICAL findings first.\n")

    # Summary table
    out.append(section_header(2, "Summary"))
    out.append(
        f"- **CRITICAL findings:** {len(by_severity['CRITICAL'])}"
    )
    out.append(
        f"- **WARNING findings:** {len(by_severity['WARNING'])}"
    )
    out.append(
        f"- **INFO findings:** {len(by_severity['INFO'])}"
    )
    out.append("")
    probes_run = [
        ("Cache detection", cache is not None),
        ("Multi-token decoding", spec is not None),
        ("Roofline", roofline is not None),
        ("Pareto sweep", pareto is not None),
        ("Precision sleuthing", sleuth is not None),
        ("BF16-vs-FP4 accuracy", compare is not None),
    ]
    out.append("**Probes run:** " + ", ".join(
        f"{name}{'' if ran else ' (skipped)'}" for name, ran in probes_run
    ))
    out.append("")

    # Verdicts by severity
    for severity in ("CRITICAL", "WARNING", "INFO"):
        items = by_severity[severity]
        if not items:
            continue
        icon = {"CRITICAL": "🚨", "WARNING": "⚠️", "INFO": "ℹ️"}[severity]
        out.append(section_header(2, f"{icon} {severity} findings"))
        for source, verdict in items:
            out.append(f"**[{source}]** {verdict}\n")

    if all(len(by_severity[s]) == 0 for s in by_severity):
        out.append(section_header(2, "✅ No findings"))
        out.append("All probes ran without producing verdicts above the "
                   "threshold for flagging. This does NOT mean the partner's "
                   "claims are validated -- only that the probes did not "
                   "produce evidence to refute them. Re-read the per-probe "
                   "details below before drawing conclusions.\n")

    # Detail sections
    out.append(section_header(2, "Per-probe details"))
    for title, body, _verdicts in all_sections:
        out.append(section_header(3, title))
        out.extend(body)

    # Closing caveats
    out.append(section_header(2, "Caveats"))
    out.append("""\
These tests cannot, by their nature, observe the partner's server-side
configuration. The probes infer behavior from client-visible signals only:
output text, response timings, and token streaming patterns. The following
remain unverifiable without container or metric access:

- Actual chip count and hardware generation
- Actual quantization scheme (we can only infer from divergence and roofline math)
- TPU MFU, HBM utilization, duty cycle
- Disaggregated serving details
- Batch policy and scheduler internals

For procurement or integration decisions, treat these results as
falsifying evidence (probes that fire = problem) but not as confirming
evidence (probes that don't fire ≠ "validated"). Recommend requiring
Docker image access before final acceptance.
""")
    return "\n".join(out)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--out-dir", default="out",
                   help="Directory containing probe JSON outputs")
    p.add_argument("--output", default="validation_report.md",
                   help="Path to write the assembled markdown report")
    args = p.parse_args()
    report = build_report(Path(args.out_dir))
    Path(args.output).write_text(report)
    print(f"[report] wrote {args.output}")
    # Also print summary counts to stderr
    import sys
    print(f"[report] see {args.output} for full details", file=sys.stderr)


if __name__ == "__main__":
    main()
