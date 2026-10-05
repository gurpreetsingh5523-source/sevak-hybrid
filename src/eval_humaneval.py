"""Evaluate on HumanEval-style coding tasks."""

import argparse
import json
import os
import re

from datasets import load_dataset

from src.config import load_config
from src.llm import LLM
from src.verifier import verify_code


SYSTEM = """You are a Python expert.
Output only complete Python code.
No markdown.
No explanation.
"""


def extract_code(text: str) -> str:
    """Extract code from model output."""
    if not text:
        return ""

    fenced = re.search(r"```(?:python)?\s*(.*?)\s*```", text, re.S)
    if fenced:
        return fenced.group(1).strip()

    return text.strip()


def main():
    ap = argparse.ArgumentParser(description="Evaluate on HumanEval")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", default="runs/humaneval_results.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("Loading HumanEval dataset...")
    ds = load_dataset("openai/openai_humaneval", split="test")

    print(f"Loading model: {cfg['model_name']}...")
    llm = LLM(
        model_name=cfg["model_name"],
        device_map=cfg.get("device_map", "auto"),
        torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        cache_dir=cfg.get("cache_dir"),
    )

    passed = 0
    total = 0

    print(f"Starting evaluation (limit={args.limit})...")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, ex in enumerate(ds):
            if i >= args.limit:
                break

            user_content = (
                "Complete the following Python function. "
                "Output only the full code.\n\n"
                f"{ex['prompt']}"
            )

            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_content},
            ]

            raw = llm.chat(
                messages,
                temperature=cfg.get("temperature", 0.2),
                top_p=cfg.get("top_p", 0.95),
                max_new_tokens=cfg.get("max_new_tokens", 1024),
            )

            code = extract_code(raw)

            ok, result = verify_code(
                code=code,
                tests=ex["test"] + f"\n\ncheck({ex['entry_point']})\n",
                timeout=cfg.get("timeout", 8),
            )

            passed += int(ok)
            total += 1

            f.write(
                json.dumps(
                    {
                        "task_id": ex.get("task_id", i),
                        "ok": ok,
                        "stderr": result["stderr"][-1000:],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if total % 5 == 0:
                print(f"  [{total}/{args.limit}] Pass rate: {passed / total:.4f} ({passed}/{total})")

    if total:
        print(f"\nFinal HumanEval pass rate: {passed / total:.4f} on {total}")
    else:
        print("No samples evaluated.")


if __name__ == "__main__":
    main()
