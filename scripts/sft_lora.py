"""Fine-tune the base model with LoRA on tool-use trajectories."""

import argparse

import torch

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="data/tool_gsm8k.jsonl")
    ap.add_argument("--output_dir", default="runs/qwen15b_tool_lora")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--max_steps", type=int, default=-1)
    args = ap.parse_args()

    print(f"Loading tokenizer: {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset: {args.data}...")
    dataset = load_dataset("json", data_files=args.data, split="train")

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        max_steps=args.max_steps,
        # bf16 only where supported (CUDA); MPS/CPU train in fp32
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        max_length=args.max_seq_length,
        report_to="none",
        packing=False,
    )

    # Load the model ourselves: letting SFTTrainer load it from a name string
    # segfaults in transformers 5.x's threaded weight loader on this Mac.
    print(f"Loading model: {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32)

    print("Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_func,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"\nSaved LoRA model to {args.output_dir}")


if __name__ == "__main__":
    main()
