"""MMLU ablation on a fixed shuffled subset (rows 0-99 tuning, 100+ held-out).

  python -m scripts.ablate_mmlu --model mlx-community/Qwen3-1.7B-bf16 --protocol cot --n 1
"""

import argparse
import json
import os
import re
import time
from collections import Counter

from datasets import load_dataset

from src.mlx_llm import MLXLLM

DEV = "data/mmlu_dev.jsonl"
LETTERS = "ABCD"

DIRECT_SYSTEM = "Answer the multiple-choice question with only the letter A, B, C, or D."
COT_SYSTEM = (
    "Answer the multiple-choice question. Think step by step briefly, "
    "then finish with 'Answer: X' where X is A, B, C, or D."
)


def make_dev(size=400, seed=0):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=seed)
    os.makedirs("data", exist_ok=True)
    with open(DEV, "w", encoding="utf-8") as f:
        for i in range(size):
            r = ds[i]
            f.write(json.dumps({
                "subject": r["subject"], "question": r["question"],
                "choices": r["choices"], "gold": LETTERS[int(r["answer"])],
            }) + "\n")


def format_q(r):
    lines = [r["question"]] + [f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"])]
    return "\n".join(lines)


def extract_letter(text: str) -> str:
    """Prefer an explicit 'Answer: X'; else a lone leading letter; else last standalone A-D."""
    t = text or ""
    m = re.findall(r"[Aa]nswer\s*(?:is)?\s*[:：]?\s*\(?([ABCD])\b", t)
    if m:
        return m[-1]
    m = re.match(r"\s*\(?([ABCD])\b", t)
    if m:
        return m.group(1)
    m = re.findall(r"\b([ABCD])\b", t)
    return m[-1] if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=["direct", "cot", "logit"], default="direct")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    if not os.path.exists(DEV):
        make_dev()
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")][args.offset: args.offset + args.limit]
    llm = MLXLLM(args.model)
    system = COT_SYSTEM if args.protocol == "cot" else DIRECT_SYSTEM
    max_tokens = 768 if args.protocol == "cot" else 8

    name = f"mmlu|{args.model.split('/')[-1]}|{args.protocol}|n{args.n}|t{args.temp}" + (
        f"|off{args.offset}" if args.offset else "")
    detail = open(f"runs/{name.replace('|', '_')}.jsonl", "w", encoding="utf-8")
    correct = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": format_q(r)}]
        if args.protocol == "logit":
            # System-1 decision: one forward pass, calibrated A-D probabilities
            probs = llm.choice_probs(msgs, list(LETTERS))
            k = max(range(4), key=lambda j: probs[j])
            pred, conf, cands = LETTERS[k], probs[k], [round(p, 4) for p in probs]
        else:
            outs = llm.chat_n(msgs, n=args.n, temperature=args.temp, max_new_tokens=max_tokens)
            cands = [extract_letter(o) for o in outs]
            c = Counter(x for x in cands if x)
            pred = c.most_common(1)[0][0] if c else ""
            conf = (c.most_common(1)[0][1] / len(cands)) if c else 0.0
        ok = pred == r["gold"]
        correct += ok
        detail.write(json.dumps({"i": args.offset + i, "subject": r["subject"], "gold": r["gold"],
                                 "pred": pred, "cands": cands, "conf": conf, "ok": ok}) + "\n")
        print(f"[{i + 1}/{len(rows)}] acc={correct / (i + 1):.3f}", flush=True)

    out = {"config": name, "n_q": len(rows), "acc": round(correct / len(rows), 4),
           "sec_per_q": round((time.time() - t0) / len(rows), 2)}
    print(json.dumps(out))
    with open("runs/ablation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
