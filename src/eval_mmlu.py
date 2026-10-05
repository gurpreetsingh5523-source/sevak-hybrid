"""Evaluate on MMLU with multi-path voting."""

import argparse
import json
import os
import re

from datasets import load_dataset

from src.config import load_config
from src.llm import LLM
from src.verifier import majority_vote

LETTERS = ["A", "B", "C", "D"]

SYSTEM = """You are a multiple-choice exam solver.
Answer only with A, B, C, or D.
No explanation.
"""


def format_question(row) -> str:
    """Format an MMLU question for the model."""
    choices = row["choices"]
    lines = [row["question"]]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer with only the letter.")
    return "\n".join(lines)


def extract_letter(text: str) -> str:
    """Extract A/B/C/D from model output."""
    if not text:
        return ""

    t = text.strip().upper()
    m = re.search(r"\b([A-D])\b", t)
    if m:
        return m.group(1)

    if t and t[0] in "ABCD":
        return t[0]

    return ""


def main():
    ap = argparse.ArgumentParser(description="Evaluate on MMLU")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="runs/mmlu_results.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading MMLU dataset...")
    ds = load_dataset("cais/mmlu", "all", split="test")
    # "all" is grouped by subject; shuffle so --limit samples across subjects
    ds = ds.shuffle(seed=0)

    print(f"Loading model: {cfg['model_name']}...")
    llm = LLM(
        model_name=cfg["model_name"],
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        cache_dir=cfg.get("cache_dir"),
    )

    correct = 0
    total = 0

    n_paths = int(cfg.get("n_paths", 1))

    print(f"Starting evaluation (limit={args.limit}, n_paths={n_paths})...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= args.limit:
                break

            user_content = format_question(row)
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ]

            letters = []
            for _ in range(n_paths):
                raw = llm.chat(
                    messages,
                    temperature=0.7 if n_paths > 1 else 0.0,
                    top_p=cfg.get("top_p", 0.95),
                    max_new_tokens=64,
                )
                letters.append(extract_letter(raw))

            # majority_vote lowercases; gold is uppercase
            pred = (majority_vote(letters) or "").upper()
            gold = LETTERS[int(row["answer"])]

            ok = pred == gold
            correct += int(ok)
            total += 1

            f.write(
                json.dumps(
                    {
                        "id": i,
                        "pred": pred,
                        "gold": gold,
                        "ok": ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if total % 20 == 0:
                print(f"  [{total}/{args.limit}] Accuracy: {correct / total:.4f} ({correct}/{total})")

    if total:
        print(f"\nFinal MMLU accuracy: {correct / total:.4f} on {total}")
    else:
        print("No samples evaluated.")


if __name__ == "__main__":
    main()
