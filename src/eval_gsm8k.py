"""Evaluate on GSM8K using the hybrid agent."""

import argparse
import json
import os

from datasets import load_dataset

from src.agent import solve
from src.config import load_config
from src.llm import LLM
from src.verifier import (
    extract_gsm8k_answer,
    extract_number,
    majority_vote,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate on GSM8K")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--out", default="runs/gsm8k_results.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading GSM8K dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    print(f"Loading model: {cfg['model_name']}...")
    llm = LLM(
        model_name=cfg["model_name"],
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        cache_dir=cfg.get("cache_dir"),
    )

    correct = 0
    total = 0

    print(f"Starting evaluation (limit={args.limit})...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break

            candidates = solve(ex["question"], llm, cfg)
            pred = majority_vote([extract_number(c) for c in candidates])

            gold = extract_number(extract_gsm8k_answer(ex["answer"]))
            pred_norm = pred or ""

            ok = pred_norm == gold
            correct += int(ok)
            total += 1

            f.write(
                json.dumps(
                    {
                        "id": i,
                        "question": ex["question"],
                        "pred": pred,
                        "pred_norm": pred_norm,
                        "gold": gold,
                        "ok": ok,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if total % 10 == 0:
                print(f"  [{total}/{args.limit}] Accuracy: {correct / total:.4f} ({correct}/{total})")

    if total:
        print(f"\nFinal GSM8K accuracy: {correct / total:.4f} on {total} samples")
    else:
        print("No samples evaluated.")


if __name__ == "__main__":
    main()
