"""HumanEval ablation. Rows 0-63 tuning, 64-163 held-out.

Protocols:
  greedy  - one completion, scored on the hidden tests (= pass@1)
  select  - sample n completions, pick one using ONLY the public docstring
            examples (>>> lines) as a self-test, then score that one pick on
            the hidden tests. Hidden tests are never used for selection.

  python -m scripts.ablate_humaneval --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol select --n 8
"""

import argparse
import json
import re
import time

from datasets import load_dataset

from src.mlx_llm import MLXLLM
from src.tools import run_python

SYSTEM = (
    "You are an expert Python programmer. Complete the function. Reply with one "
    "```python code block containing the full function (and any imports it needs)."
)


def extract_code(text: str, prompt: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    code = blocks[0] if blocks else (text or "")
    # If the model returned only a body, glue it onto the given signature
    if "def " not in code:
        code = prompt + code
    # Keep the prompt's imports/helpers available
    header = "\n".join(l for l in prompt.splitlines() if l.startswith(("import ", "from ")))
    return header + "\n" + code


def doc_examples(prompt: str, entry: str) -> list[str]:
    """Public examples from the docstring only (hidden tests are never used).

    Handles the styles HumanEval uses:  '>>> f(x)' + next-line result,
    'f(x) == y', 'f(x) ➞ y', 'f(x) => y', 'f(x) -> y', 'f(x) should return y'.
    """
    lines = prompt.splitlines()
    tests = []
    for k, l in enumerate(lines):
        s = l.strip()
        if s.startswith(">>>") and k + 1 < len(lines):
            call = s[3:].strip()
            expect = lines[k + 1].strip()
            if call and expect and not expect.startswith(">>>"):
                tests.append(f"assert ({call}) == ({expect})")
            continue
        m = re.match(rf"^.*?({re.escape(entry)}\(.*\))\s*(?:==|➞|=>|->|should return|returns|=)\s*(.+?)\.?$", s)
        if m:
            tests.append(f"assert ({m.group(1)}) == ({m.group(2).strip()})")
    # keep only examples that are valid Python
    ok = []
    for t in tests:
        try:
            compile(t, "<ex>", "exec")
            ok.append(t)
        except SyntaxError:
            pass
    return ok


def self_test_score(code: str, tests: list[str]) -> int:
    if not tests:
        r = run_python(code, timeout=5)
        return int(r["ok"])
    body = code + "\n" + "\n".join(f"try:\n    {t}\n    print('PASS')\nexcept Exception:\n    pass" for t in tests)
    r = run_python(body, timeout=5)
    return r["stdout"].count("PASS") if r["ok"] or r["stdout"] else 0


def hidden_pass(code: str, ex) -> bool:
    full = code + "\n\n" + ex["test"] + f"\n\ncheck({ex['entry_point']})\n"
    return run_python(full, timeout=10)["ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--protocol", choices=["greedy", "select"], default="greedy")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--limit", type=int, default=64)
    args = ap.parse_args()

    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = [ds[i] for i in range(args.offset, min(len(ds), args.offset + args.limit))]
    llm = MLXLLM(args.model)

    name = f"he|{args.model.split('/')[-1]}|{args.protocol}|n{args.n}" + (f"|off{args.offset}" if args.offset else "")
    detail = open(f"runs/{name.replace('|', '_')}.jsonl", "w", encoding="utf-8")
    passed = oracle = 0
    t0 = time.time()
    for i, ex in enumerate(rows):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ex["prompt"]}]
        if args.protocol == "greedy":
            outs = llm.chat_n(msgs, n=1, temperature=0.0, max_new_tokens=768)
        else:
            # sample 0 is greedy so ties / no public examples fall back to pass@1 greedy
            outs = llm.chat_n(msgs, n=1, temperature=0.0, max_new_tokens=768)
            outs += llm.chat_n(msgs, n=args.n - 1, temperature=args.temp, max_new_tokens=768)
        codes = [extract_code(o, ex["prompt"]) for o in outs]
        tests = doc_examples(ex["prompt"], ex["entry_point"])
        scores = [self_test_score(c, tests) for c in codes]
        pick = max(range(len(codes)), key=lambda k: scores[k])  # ties -> first sample
        ok = hidden_pass(codes[pick], ex)
        passed += ok
        if args.protocol == "select":
            oracle += any(hidden_pass(c, ex) for c in codes)
        detail.write(json.dumps({"task": ex["task_id"], "ok": ok, "scores": scores, "n_public": len(tests)}) + "\n")
        print(f"[{i + 1}/{len(rows)}] pass={passed / (i + 1):.3f}", flush=True)

    out = {"config": name, "n_q": len(rows), "pass@1": round(passed / len(rows), 4),
           "oracle": round(oracle / len(rows), 4) if args.protocol == "select" else None,
           "sec_per_q": round((time.time() - t0) / len(rows), 2)}
    print(json.dumps(out))
    with open("runs/ablation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()
