"""LLM wrapper with chat template support."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLM:
    """Wrapper around a causal LM with chat template support."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        cache_dir: str | None = None,
    ):
        dtype = getattr(torch, torch_dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            dtype=dtype,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_new_tokens: int = 1024,
    ) -> str:
        """Chat with the model using chat templates."""
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)
