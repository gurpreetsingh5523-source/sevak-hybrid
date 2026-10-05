"""Bootstrap tool-use training data from GSM8K train set.

Run the agent on GSM8K train problems, keep only successful trajectories,
and save them as SFT training data.
"""

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
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--out", default="data/tool_gsm8k.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["n_paths"] = args.n

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading GSM8K train dataset...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    print(f"Loading model: {cfg['model_name']}...")
    llm = LLM(
        model_name=cfg["model_name"],
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        cache_dir=cfg.get("cache_dir"),
    )

    saved = 0

    print(f"Bootstrapping {args.limit} examples (n_paths={args.n})...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break

            candidates, trajectories = solve(
                ex["question"],
                llm,
                cfg,
                return_trajectory=True,
            )

            gold = extract_number(extract_gsm8k_answer(ex["answer"]))
            # Save the first trajectory that itself reached the gold answer
            good = [t for c, t in zip(candidates, trajectories) if extract_number(c) == gold]

            if good:
                f.write(
                    json.dumps(
                        {"messages": good[0]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                saved += 1

            if (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}, saved {saved}")

    print(f"\nDone. Saved {saved} trajectories to {args.out}")


if __name__ == "__main__":
    main()
