"""MMLU System-1 -> System-2 gate (the JEV idea, built from small models).

System 1: average the A-D logit probabilities of Qwen2.5-1.5B and Qwen3.5-2B
          (read from their saved `--protocol logit` runs; one forward pass each).
System 2: if the pooled top probability < tau, ask Qwen3-1.7B to reason (CoT).

tau=0.7 was picked on rows 0-99. Report rows 100-399.

  python -m scripts.eval_mmlu_gate --offset 100 --limit 300 --tau 0.7
"""

import argparse
import json
import time

from scripts.ablate_mmlu import COT_SYSTEM, DEV, LETTERS, extract_letter, format_q
from src.mlx_llm import MLXLLM

S1 = ["Qwen2.5-1.5B-Instruct", "Qwen3.5-2B-MLX-bf16"]
S2 = "mlx-community/Qwen3-1.7B-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offset", type=int, default=100)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")]
    s1 = [[json.loads(l) for l in open(f"runs/mmlu_{m}_logit_n1_t0.0.jsonl")] for m in S1]
    idx = range(args.offset, args.offset + args.limit)

    llm = MLXLLM(S2)
    correct = escalated = 0
    t0 = time.time()
    detail = open(f"runs/mmlu_gate_tau{args.tau}_off{args.offset}.jsonl", "w", encoding="utf-8")
    for n, i in enumerate(idx):
        p = [sum(run[i]["cands"][j] for run in s1) / len(s1) for j in range(4)]
        k = max(range(4), key=lambda j: p[j])
        pred, system = LETTERS[k], 1
        if p[k] < args.tau:
            msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": format_q(rows[i])}]
            pred = extract_letter(llm.chat(msgs, temperature=0.0, max_new_tokens=768)) or pred
            system = 2
            escalated += 1
        ok = pred == rows[i]["gold"]
        correct += ok
        detail.write(json.dumps({"i": i, "gold": rows[i]["gold"], "pred": pred, "s1_conf": round(p[k], 4),
                                 "system": system, "ok": ok}) + "\n")
        print(f"[{n + 1}/{len(idx)}] acc={correct / (n + 1):.3f}", flush=True)

    out = {"config": f"mmlu|gate|tau{args.tau}|off{args.offset}", "n_q": len(idx),
           "acc": round(correct / len(idx), 4), "escalated": round(escalated / len(idx), 3),
           "sec_per_q_s2_only": round((time.time() - t0) / len(idx), 2)}
    print(json.dumps(out))
    with open("runs/ablation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
