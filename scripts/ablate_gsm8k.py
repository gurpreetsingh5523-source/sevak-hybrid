"""Compare model x protocol x n_paths on a FIXED GSM8K dev set.

Every config runs on the same questions (data/gsm8k_dev.jsonl), so numbers
are directly comparable. Results append to runs/ablation.jsonl.

  python -m scripts.ablate_gsm8k --model mlx-community/Qwen3-1.7B-4bit \
      --protocol hybrid --n 4 --temp 0.7
"""

import argparse
import json
import os
import time

from datasets import load_dataset

from src.mlx_llm import MLXLLM
from src.solvers import PROTOCOLS
from src.verifier import extract_gsm8k_answer, extract_number, majority_vote

DEV = "data/gsm8k_dev.jsonl"


def make_dev(size: int, seed: int = 0):
    ds = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=seed)
    os.makedirs("data", exist_ok=True)
    with open(DEV, "w", encoding="utf-8") as f:
        for i in range(size):
            ex = ds[i]
            gold = extract_number(extract_gsm8k_answer(ex["answer"]))
            f.write(json.dumps({"question": ex["question"], "gold": gold}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=list(PROTOCOLS), default="cot")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--offset", type=int, default=0, help="rows 0-99 = tuning set; use 100+ for held-out")
    ap.add_argument("--dev_size", type=int, default=400)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if not os.path.exists(DEV):
        make_dev(args.dev_size)
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")][args.offset : args.offset + args.limit]

    llm = MLXLLM(args.model, thinking=args.thinking)
    solver = PROTOCOLS[args.protocol]
    gen = {"temperature": args.temp, "max_new_tokens": args.max_tokens}

    name = f"{args.model.split('/')[-1]}|{args.protocol}|n{args.n}|t{args.temp}" + (
        "|think" if args.thinking else ""
    ) + (f"|{args.tag}" if args.tag else "") + (f"|off{args.offset}" if args.offset else "")
    os.makedirs("runs", exist_ok=True)
    detail = open(f"runs/ablate_{name.replace('|', '_')}.jsonl", "w", encoding="utf-8")

    correct = 0
    oracle = 0  # any path right: upper bound a perfect verifier could reach
    t0 = time.time()
    for i, r in enumerate(rows):
        cands = solver(r["question"], llm, args.n, **gen)
        pred = majority_vote(cands) or ""
        ok = pred == r["gold"]
        correct += ok
        oracle += r["gold"] in cands
        detail.write(json.dumps({"i": i, "gold": r["gold"], "pred": pred, "cands": cands, "ok": ok}) + "\n")
        print(f"[{i + 1}/{len(rows)}] acc={correct / (i + 1):.3f}", flush=True)

    secs = time.time() - t0
    out = {
        "config": name,
        "n_q": len(rows),
        "acc": round(correct / len(rows), 4),
        "oracle": round(oracle / len(rows), 4),
        "sec_per_q": round(secs / len(rows), 2),
    }
    print(json.dumps(out))
    with open("runs/ablation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
