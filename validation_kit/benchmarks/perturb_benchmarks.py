"""
Programmatic perturbation of standard benchmarks.

The goal is to produce variants of MMLU and GSM8K that should yield the
same answer but exercise different surface forms. A model that scores well
on the original and badly on the perturbed set is overfit to surface
patterns (or its quantization calibration is benchmark-tuned).

Inputs:
  - GSM8K test set in JSONL: {"question": ..., "answer": ...}
  - MMLU CSV or JSONL: {"question": ..., "A": ..., "B": ..., "C": ..., "D": ..., "answer": "A"}

Outputs: JSONL files ready for accuracy_harness.py.

Usage:
  python -m benchmarks.perturb_benchmarks \\
      --gsm8k gsm8k_test.jsonl \\
      --mmlu mmlu_test.jsonl \\
      --out-dir prompt_sets/
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# GSM8K: change number magnitudes, preserve problem structure (GSM-Symbolic-lite)
# ---------------------------------------------------------------------------

_INT_RE = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def perturb_gsm8k_numbers(
    question: str, answer: str, rng: random.Random
) -> tuple[str, str] | None:
    """
    Replace integers in a GSM8K question with random integers of similar
    magnitude, and recompute the answer if possible.

    The GSM8K answer field has format "... reasoning ... #### NUMERIC_ANSWER".
    We can't actually rerun the reasoning -- so the safe perturbation is:
    pick a SCALE factor and multiply ALL integers in the question by that
    scale, then multiply the final answer by the same scale. This preserves
    the arithmetic structure.
    """
    # Extract numeric answer
    if "####" not in answer:
        return None
    try:
        final_str = answer.split("####")[-1].strip().replace(",", "")
        final = float(final_str)
    except (ValueError, IndexError):
        return None

    scale = rng.choice([2, 3, 5, 7, 11])  # integer scales preserve int answers
    perturbed_question = _INT_RE.sub(
        lambda m: str(int(m.group(1)) * scale),
        question,
    )
    perturbed_answer = str(int(final * scale)) if final.is_integer() \
        else f"{final * scale:.4f}".rstrip("0").rstrip(".")
    return perturbed_question, perturbed_answer


def gsm8k_prompt(question: str) -> str:
    """Standard GSM8K chain-of-thought prompt."""
    return (
        "Solve the following math problem. Show your work, then on a new "
        "line write '#### ' followed by the final numeric answer.\n\n"
        f"Question: {question}\n\n"
        "Solution:"
    )


# ---------------------------------------------------------------------------
# MMLU: permute answer order, paraphrase question (basic)
# ---------------------------------------------------------------------------

def perturb_mmlu_permute(
    item: dict, rng: random.Random
) -> dict:
    """
    Shuffle MMLU answer order. The correct content is preserved; only the
    letter (A/B/C/D) attached to it changes. A model relying on positional
    biases will score worse.
    """
    letters = ["A", "B", "C", "D"]
    answer_letter = item["answer"].strip().upper()
    correct_content = item[answer_letter]

    # Get all 4 options, shuffle them, assign new letters
    options = [item[L] for L in letters]
    permutation = list(range(4))
    rng.shuffle(permutation)
    new_options = [options[i] for i in permutation]
    new_item = dict(item)
    for i, L in enumerate(letters):
        new_item[L] = new_options[i]
    # Find new letter for the correct answer
    new_answer = letters[new_options.index(correct_content)]
    new_item["answer"] = new_answer
    return new_item


def mmlu_prompt(item: dict) -> str:
    """Standard MMLU prompt."""
    return (
        f"Answer the following multiple choice question. Respond with just "
        f"the letter (A, B, C, or D) of the correct answer.\n\n"
        f"Question: {item['question']}\n"
        f"A. {item['A']}\n"
        f"B. {item['B']}\n"
        f"C. {item['C']}\n"
        f"D. {item['D']}\n\n"
        f"Answer:"
    )


# ---------------------------------------------------------------------------
# Filler-injection perturbation: distract with irrelevant prefix
# ---------------------------------------------------------------------------

_FILLER = (
    "Before answering, please note that the following information is "
    "completely irrelevant and should be ignored: " + " ".join(
        [
            "The capital of an imaginary country is Zephyria.",
            "Octopuses have three hearts and blue blood.",
            "A standard deck of cards has 52 cards in four suits.",
            "Mount Everest grows by approximately 4mm each year.",
            "Honey never spoils due to its low water content.",
        ]
    ) + " Now, the actual question: "
)


def with_filler(prompt: str) -> str:
    return _FILLER + prompt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--gsm8k", type=str, help="Path to GSM8K test JSONL")
    p.add_argument("--mmlu", type=str, help="Path to MMLU test JSONL")
    p.add_argument("--limit", type=int, default=200,
                   help="Cap items per benchmark (for fast turnaround)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default="prompt_sets")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.out_dir)

    if args.gsm8k:
        src = Path(args.gsm8k)
        items = [json.loads(line) for line in src.open() if line.strip()]
        items = items[: args.limit]

        # Standard
        standard = []
        for i, item in enumerate(items):
            ans_num = re.search(r"####\s*(-?[\d,.]+)", item["answer"])
            if not ans_num:
                continue
            standard.append({
                "id": f"gsm8k_std_{i}",
                "prompt": gsm8k_prompt(item["question"]),
                "expected": ans_num.group(1).replace(",", ""),
                "type": "numeric",
            })
        write_jsonl(out_dir / "gsm8k_standard.jsonl", standard)
        print(f"  wrote gsm8k_standard.jsonl: {len(standard)} prompts")

        # Number-perturbed
        perturbed = []
        for i, item in enumerate(items):
            res = perturb_gsm8k_numbers(item["question"], item["answer"], rng)
            if res is None:
                continue
            q, a = res
            perturbed.append({
                "id": f"gsm8k_pert_{i}",
                "perturbed_from": f"gsm8k_std_{i}",
                "prompt": gsm8k_prompt(q),
                "expected": a,
                "type": "numeric",
            })
        write_jsonl(out_dir / "gsm8k_perturbed.jsonl", perturbed)
        print(f"  wrote gsm8k_perturbed.jsonl: {len(perturbed)} prompts")

        # Filler-injected (uses original questions)
        filler = [{**s, "id": s["id"].replace("std", "filler"),
                   "perturbed_from": s["id"],
                   "prompt": with_filler(s["prompt"])} for s in standard]
        write_jsonl(out_dir / "gsm8k_filler.jsonl", filler)
        print(f"  wrote gsm8k_filler.jsonl: {len(filler)} prompts")

    if args.mmlu:
        src = Path(args.mmlu)
        items = [json.loads(line) for line in src.open() if line.strip()]
        items = items[: args.limit]

        # Standard
        standard = [
            {
                "id": f"mmlu_std_{i}",
                "prompt": mmlu_prompt(item),
                "expected": item["answer"],
                "type": "letter",
            }
            for i, item in enumerate(items)
        ]
        write_jsonl(out_dir / "mmlu_standard.jsonl", standard)
        print(f"  wrote mmlu_standard.jsonl: {len(standard)} prompts")

        # Permuted answer order
        permuted = []
        for i, item in enumerate(items):
            p_item = perturb_mmlu_permute(item, rng)
            permuted.append({
                "id": f"mmlu_perm_{i}",
                "perturbed_from": f"mmlu_std_{i}",
                "prompt": mmlu_prompt(p_item),
                "expected": p_item["answer"],
                "type": "letter",
            })
        write_jsonl(out_dir / "mmlu_permuted.jsonl", permuted)
        print(f"  wrote mmlu_permuted.jsonl: {len(permuted)} prompts")


if __name__ == "__main__":
    main()
