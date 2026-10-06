"""Sevak: adaptive self-consistency with a calibrated confidence.

Default is ONE model, Qwen2.5-Math-1.5B-Instruct: sample 4 chain-of-thought
answers, stop if >=75% agree, otherwise sample 4 more and vote over all 8.

Held-out GSM8K (rows 100-299 of data/gsm8k_dev.jsonl, never used for tuning):
  93.5% at ~4.5 samples/question  vs  Phi-4-mini-8bit 92.5% (8 votes), 90.5% (greedy)
High-confidence answers (90% of questions) were right 97% of the time, so
`confidence` is usable: trust high, flag/escalate low instead of guessing.

Multiple models are supported, but pooling Math-1.5B with Qwen3-1.7B scored
92% on the tuning split and only 90.5% held-out -- the weaker model drags the
vote down. Don't add a model without a held-out check.
"""

from collections import Counter

from .mlx_llm import MLXLLM
from .solvers import cot

DEFAULT_MODELS = ("mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16",)


class Sevak:
    def __init__(
        self,
        models=DEFAULT_MODELS,
        round_size: int = 4,
        max_rounds: int = 2,
        stop_at: float = 0.75,
        temperature: float = 0.7,
        max_new_tokens: int = 1024,
    ):
        self.llms = [MLXLLM(m) for m in models]
        self.round_size = round_size
        self.max_rounds = max_rounds
        self.stop_at = stop_at
        self.gen = {"temperature": temperature, "max_new_tokens": max_new_tokens}

    def solve(self, problem: str) -> dict:
        votes = []
        for r in range(self.max_rounds):
            for llm in self.llms:
                votes += cot(problem, llm, self.round_size, **self.gen)
            answer, confidence = self._tally(votes)
            if confidence >= self.stop_at:
                break
        return {
            "answer": answer,
            "confidence": round(confidence, 3),
            "votes": votes,
            "rounds": r + 1,
        }

    @staticmethod
    def _tally(votes):
        counts = Counter(v for v in votes if v)
        if not counts:
            return "", 0.0
        answer, k = counts.most_common(1)[0]
        # share of ALL votes (abstentions count against confidence)
        return answer, k / len(votes)
