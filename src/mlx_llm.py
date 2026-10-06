"""MLX backend: same .chat() interface as src.llm.LLM, plus batched sampling.

Much faster than transformers+MPS on Apple Silicon, and chat_n() samples
all voting paths for a prompt in a single batch.
"""

import re

from mlx_lm import batch_generate, load
from mlx_lm.sample_utils import make_sampler

THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S)


def strip_think(text: str) -> str:
    """Drop Qwen3-style <think>...</think> blocks (unterminated ones too)."""
    return THINK_RE.sub("", text or "").strip()


class MLXLLM:
    def __init__(self, model_name: str, thinking: bool = False, **_):
        self.model, self.tokenizer = load(model_name)
        self.thinking = thinking

    def _encode(self, messages: list[dict]) -> list[int]:
        kwargs = {"add_generation_prompt": True, "tokenize": True}
        try:
            # Qwen3/3.5 templates take enable_thinking; others ignore/reject it
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def chat_batch(
        self,
        conversations: list[list[dict]],
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ) -> list[str]:
        """Generate one completion per conversation, all in one batch."""
        prompts = [self._encode(m) for m in conversations]
        sampler = make_sampler(temp=temperature, top_p=top_p if temperature > 0 else 0.0)
        resp = batch_generate(
            self.model,
            self.tokenizer,
            prompts,
            max_tokens=max_new_tokens,
            sampler=sampler,
        )
        return [strip_think(t) for t in resp.texts]

    def think_n(
        self,
        messages: list[dict],
        n: int = 1,
        budget: int = 2048,
        temperature: float = 0.6,
        force_suffix: str = "\n</think>\n\nThe final answer is \\boxed{",
    ) -> list[str]:
        """Thinking-mode samples with budget forcing.

        Small models' thinking often runs past any token cap (or loops), which
        leaves no answer at all. When a sample hits the budget without closing
        </think>, we close it for the model and make it commit to an answer.
        Greedy decoding makes Qwen3 thinking loop, so temperature defaults to
        the model card's recommended 0.6.
        """
        assert self.thinking, "construct with thinking=True"
        base = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=True
        )
        sampler = make_sampler(temp=temperature, top_p=0.95)
        prompt_ids = self.tokenizer.encode(base, add_special_tokens=False)
        raw = batch_generate(
            self.model, self.tokenizer, [prompt_ids] * n, max_tokens=budget, sampler=sampler
        ).texts

        unfinished = [i for i, t in enumerate(raw) if "</think>" not in t]
        if unfinished:
            cont = [
                self.tokenizer.encode(base + raw[i] + force_suffix, add_special_tokens=False)
                for i in unfinished
            ]
            tails = batch_generate(
                self.model, self.tokenizer, cont, max_tokens=24, sampler=make_sampler(temp=0.0)
            ).texts
            for i, tail in zip(unfinished, tails):
                raw[i] = "\\boxed{" + tail
        return [strip_think(t) for t in raw]

    def choice_probs(self, messages: list[dict], options: list[str]) -> list[float]:
        """'System 1' decision (the JEV / Jev idea): no text generation, one
        forward pass, read the next-token logits of each option label and
        softmax over just those. Options must be single-token labels like
        "A".."D"; returns one probability per option."""
        import mlx.core as mx

        ids = self._encode(messages)
        logits = self.model(mx.array([ids]))[0, -1].astype(mx.float32)
        opt_ids = [self.tokenizer.encode(o, add_special_tokens=False)[0] for o in options]
        sel = logits[mx.array(opt_ids)]
        probs = mx.softmax(sel, axis=-1)
        return probs.tolist()

    def chat_n(self, messages: list[dict], n: int = 1, **kw) -> list[str]:
        """Sample n completions of the same conversation (for voting)."""
        return self.chat_batch([messages] * n, **kw)

    def chat(self, messages: list[dict], **kw) -> str:
        return self.chat_batch([messages], **kw)[0]
