"""Sevak router: send each question to the small specialist that measured best.

  math   -> Qwen2.5-Math-1.5B, CoT, adaptive 4->8 self-consistency (src.sevak)
  code   -> Qwen2.5-Coder-1.5B
  choice -> System 1: pooled A-D logits of Qwen2.5-1.5B + Qwen3.5-2B;
            System 2 (if pooled confidence < tau): Qwen3-1.7B chain-of-thought

Every answer comes back with a confidence and which system produced it.
Models load lazily and only one specialist group is kept in memory, so the
whole thing fits in 16 GB.
"""

import gc
import re

from .mlx_llm import MLXLLM
from .solvers import extract_boxed, COT_SYSTEM as MATH_SYSTEM

MODELS = {
    "math": "mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16",
    "code": "mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16",
    "s1": ("Qwen/Qwen2.5-1.5B-Instruct", "mlx-community/Qwen3.5-2B-MLX-bf16"),
    "s2": "mlx-community/Qwen3-1.7B-bf16",
}

LETTERS = "ABCDEFGHIJ"
CHOICE_SYSTEM = "Answer the multiple-choice question with only the letter of the correct option."
CHOICE_COT = (
    "Answer the multiple-choice question. Think step by step briefly, "
    "then finish with 'Answer: X' where X is the option letter."
)
CODE_SYSTEM = (
    "You are an expert Python programmer. Reply with one ```python code block "
    "containing the complete solution."
)


def detect_task(question: str, choices: list[str] | None = None) -> str:
    """Cheap, transparent routing. choices given -> choice; a Python signature
    or explicit coding request -> code; otherwise math/general reasoning."""
    if choices:
        return "choice"
    if re.search(r"^\s*def \w+\(.*\).*:", question, re.M) or re.search(
        r"\b(write|implement|code)\b.*\b(function|python|program|class)\b", question, re.I
    ):
        return "code"
    return "math"


class Router:
    def __init__(self, tau: float = 0.7):
        self.tau = tau
        self._loaded: dict[str, object] = {}

    def _get(self, key: str):
        """Keep only the models needed for the current task resident."""
        group = {"math": {"math"}, "code": {"code"}, "s1": {"s1", "s2"}, "s2": {"s1", "s2"}}[key]
        for k in list(self._loaded):
            if k not in group:
                del self._loaded[k]
        gc.collect()
        if key not in self._loaded:
            spec = MODELS[key]
            self._loaded[key] = [MLXLLM(m) for m in spec] if isinstance(spec, tuple) else MLXLLM(spec)
        return self._loaded[key]

    def answer(self, question: str, choices: list[str] | None = None) -> dict:
        task = detect_task(question, choices)
        if task == "math":
            return {"task": task, **self._math(question)}
        if task == "code":
            return {"task": task, **self._code(question)}
        return {"task": task, **self._choice(question, choices)}

    def _math(self, q, round_size=4, stop_at=0.75):
        from collections import Counter

        llm = self._get("math")
        msgs = [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": q}]
        votes = []
        for _ in range(2):
            votes += [extract_boxed(t) for t in llm.chat_n(msgs, n=round_size, temperature=0.7)]
            c = Counter(v for v in votes if v)
            ans, k = c.most_common(1)[0] if c else ("", 0)
            if k / len(votes) >= stop_at:
                break
        return {"answer": ans, "confidence": round(k / len(votes), 3), "system": 2}

    def _code(self, q):
        llm = self._get("code")
        out = llm.chat([{"role": "system", "content": CODE_SYSTEM}, {"role": "user", "content": q}],
                       temperature=0.0, max_new_tokens=768)
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", out, re.S)
        return {"answer": blocks[0] if blocks else out, "confidence": None, "system": 2}

    def _choice(self, q, choices):
        letters = list(LETTERS[: len(choices)])
        prompt = "\n".join([q] + [f"{l}. {c}" for l, c in zip(letters, choices)])
        msgs = [{"role": "system", "content": CHOICE_SYSTEM}, {"role": "user", "content": prompt}]
        s1 = self._get("s1")
        probs = [m.choice_probs(msgs, letters) for m in s1]
        p = [sum(x[j] for x in probs) / len(probs) for j in range(len(letters))]
        k = max(range(len(p)), key=lambda j: p[j])
        if p[k] >= self.tau:
            return {"answer": letters[k], "confidence": round(p[k], 3), "system": 1}
        s2 = self._get("s2")
        out = s2.chat([{"role": "system", "content": CHOICE_COT}, {"role": "user", "content": prompt}],
                      temperature=0.0, max_new_tokens=768)
        m = re.findall(r"[Aa]nswer\s*[:：]?\s*\(?([A-J])\b", out)
        ans = m[-1] if m and m[-1] in letters else letters[k]
        return {"answer": ans, "confidence": round(p[k], 3), "system": 2}
