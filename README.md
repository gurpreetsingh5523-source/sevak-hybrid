# Sevak-Hybrid: Beat Phi-4 Mini

A 1.5B parameter hybrid agent that combines test-time compute scaling, tool use, and multi-path voting to outperform 3.8B dense models.

## Quick Start

```bash
# 1. Setup environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Test all components (no model download)
python test_all.py

# 3. Run GSM8K evaluation (20 samples)
python -m src.eval_gsm8k --limit 20

# 4. Run HumanEval evaluation (5 samples)
python -m src.eval_humaneval --limit 5

# 5. Run MMLU evaluation (50 samples)
python -m src.eval_mmlu --limit 50
```

## Project Structure

```
beat-phi4-mini/
├── config.yaml              # Main configuration
├── requirements.txt         # Dependencies
├── test_all.py             # Component test suite
├── src/
│   ├── config.py           # Config loader
│   ├── llm.py              # LLM wrapper
│   ├── prompts.py          # System/user prompts
│   ├── parser.py           # JSON extraction
│   ├── tools.py            # Python sandbox
│   ├── agent.py            # Hybrid agent loop
│   ├── verifier.py         # Answer normalization/voting
│   ├── eval_gsm8k.py       # GSM8K benchmark
│   ├── eval_humaneval.py   # HumanEval benchmark
│   └── eval_mmlu.py        # MMLU benchmark
├── scripts/
│   ├── bootstrap_tool_data.py  # Generate training data
│   └── sft_lora.py            # LoRA fine-tuning
├── data/                   # Training data
└── runs/                   # Evaluation results
```

## How It Works

1. **Base Model**: Qwen2.5-1.5B-Instruct (already 50% MMLU)
2. **Tool Use**: For math/code, generate Python → execute → get exact answer
3. **Multi-Path Voting**: Generate N=4-16 reasoning paths, vote on best answer
4. **Verifier**: Small classifier scores each path quality
5. **LoRA Fine-Tuning**: Train on successful tool-use trajectories

## Targets vs. Reality

Phi-4-mini-instruct numbers are from its official HF model card
(microsoft/Phi-4-mini-instruct). Note the card's GSM8K is **88.6%**, not 70%:
the bar is higher than this README originally claimed.

| Benchmark | Phi-4-mini (3.8B, model card) | Sevak-Hybrid (1.5B) measured |
|-----------|-------------------------------|------------------------------|
| MMLU      | 67.3% (5-shot)                | not yet run                  |
| GSM8K     | 88.6% (8-shot CoT)            | 40% (10 samples, n_paths=1, smoke test only) |
| HumanEval | (check card)                  | not yet run                  |

Only fill the right column with numbers produced by `src/eval_*.py`.

## Training Pipeline

```bash
# Step 1: Bootstrap tool-use data
python -m scripts.bootstrap_tool_data --limit 200 --n 2

# Step 2: Fine-tune with LoRA
python -m scripts.sft_lora --data data/tool_gsm8k.jsonl --epochs 1

# Step 3: Evaluate the fine-tuned model
# (update config.yaml model_name to the LoRA output path)
python -m src.eval_gsm8k --limit 100
```

## Safety Note

The Python sandbox uses `subprocess` with resource limits. For production, use Docker or Firejail for stronger isolation.
