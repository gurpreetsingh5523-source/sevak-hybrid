"""Probe thinking+budget-forcing on the low-confidence questions only."""
import json, sys, time
from collections import Counter
from src.mlx_llm import MLXLLM
from src.solvers import COT_SYSTEM, extract_boxed

budget = int(sys.argv[1]); n = int(sys.argv[2])
rows = [json.loads(l) for l in open("data/gsm8k_dev.jsonl")][:100]
q = [json.loads(l) for l in open("runs/ablate_Qwen3-1.7B-4bit_cot_n8_t0.7.jsonl")]
low = [i for i, x in enumerate(q) if (Counter([v for v in x["cands"] if v]).most_common(1) or [(0, 0)])[0][1] < 6]
llm = MLXLLM("mlx-community/Qwen3-1.7B-4bit", thinking=True)
right = 0; t0 = time.time()
for i in low:
    msgs = [{"role": "system", "content": COT_SYSTEM}, {"role": "user", "content": rows[i]["question"]}]
    c = [extract_boxed(t) for t in llm.think_n(msgs, n=n, budget=budget)]
    pred = Counter([v for v in c if v]).most_common(1)[0][0] if any(c) else ""
    right += pred == rows[i]["gold"]
    print(i, rows[i]["gold"], c, flush=True)
print(json.dumps({"budget": budget, "n": n, "low_q": len(low), "right": right, "sec_per_q": round((time.time() - t0) / len(low), 1)}))
