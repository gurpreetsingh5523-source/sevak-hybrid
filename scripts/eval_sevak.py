"""Evaluate the adaptive Sevak ensemble on the GSM8K dev file.

Rows 0-99 were used to choose models and thresholds; report rows 100+ as the
held-out number:  python -m scripts.eval_sevak --offset 100 --limit 200
"""

import argparse
import json
import time

from src.sevak import Sevak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--stop_at", type=float, default=0.75)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data/gsm8k_dev.jsonl", encoding="utf-8")]
    rows = rows[args.offset : args.offset + args.limit]
    sevak = Sevak(stop_at=args.stop_at)

    tag = f"sevak_off{args.offset}_n{len(rows)}"
    detail = open(f"runs/{tag}.jsonl", "w", encoding="utf-8")
    correct = confident = confident_right = rounds = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        out = sevak.solve(r["question"])
        ok = out["answer"] == r["gold"]
        hi = out["confidence"] >= args.stop_at
        correct += ok
        confident += hi
        confident_right += ok and hi
        rounds += out["rounds"]
        detail.write(json.dumps({"i": args.offset + i, "gold": r["gold"], "ok": ok, **out}) + "\n")
        detail.flush()
        print(f"[{i + 1}/{len(rows)}] acc={correct / (i + 1):.3f}", flush=True)

    n = len(rows)
    summary = {
        "config": tag,
        "acc": round(correct / n, 4),
        "high_conf_share": round(confident / n, 4),
        "high_conf_acc": round(confident_right / max(confident, 1), 4),
        "avg_rounds": round(rounds / n, 2),
        "sec_per_q": round((time.time() - t0) / n, 2),
    }
    print(json.dumps(summary))
    with open("runs/ablation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")


if __name__ == "__main__":
    main()
