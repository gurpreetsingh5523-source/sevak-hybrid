# Sevak-Hybrid: Beat Phi-4 Mini

A router over 1.5-2B specialist models (math, code, knowledge) that matches Phi-4-mini (3.8B) on held-out GSM8K, MMLU and HumanEval, with a calibrated confidence on every answer. See the scoreboard.

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

## Final held-out scoreboard (Sevak router vs Phi-4-mini, same harness, same questions)

| Task (held-out) | Sevak specialist | Sevak | Phi-4-mini 3.8B (8bit) |
|---|---|---|---|
| Math: GSM8K, 200 q | Qwen2.5-Math-1.5B, CoT, adaptive 4->8 votes | **93.5%** | 92.5% (8 votes) / 90.5% (greedy) |
| Knowledge: MMLU, 300 q | System 1: pooled A-D logits of Qwen2.5-1.5B + Qwen3.5-2B; System 2 (conf < 0.7, 59% of q): Qwen3-1.7B CoT | **71.3%** | 70.7% (CoT) / 61.7% (logit) |
| Code: HumanEval tasks 64-163 | Qwen2.5-Coder-1.5B greedy | **59%** | 56% (greedy) |
| Code, same, 8 samples + docstring self-test* | Qwen2.5-Coder-1.5B | **70%** | 66% |

Every gap is inside noise (+/-3.5 to +/-5 points), so the honest summary is: **1.5-2B specialists
behind a router MATCH Phi-4-mini on math, knowledge and code**, with a calibrated confidence on every answer.

\* The docstring-example parser was broadened (`==`, `➞`, `=>` styles) AFTER seeing the first held-out code
run, so that row is post-hoc. It only reads public examples in the prompt; hidden tests never pick a sample.
The greedy code row is clean. Harness sanity: Coder-1.5B over all 164 tasks = 71.3% vs Qwen's reported 70.7%.

The "System 1 / System 2" MMLU gate borrows the idea of the JEV / Jev decision models: answer from one
forward pass with option probabilities when confident, think only when not.

Use it: `from src.router import Router; Router().answer(question, choices=None)` returns
`{task, answer, confidence, system}`. Models load lazily, one specialist group at a time (fits 16 GB).

## GSM8K details (measured on this Mac, same harness for every model)

Held-out = rows 100-299 of `data/gsm8k_dev.jsonl` (seed-0 shuffle of the GSM8K
test split). Rows 0-99 were used to pick models/prompts/thresholds; the
held-out rows were run once, after all choices were frozen.

| System | Params | Held-out acc | Samples/q |
|--------|--------|--------------|-----------|
| **Sevak: Qwen2.5-Math-1.5B, CoT, adaptive 4->8 votes** | 1.5B | **93.5%** | ~4.5 |
| Phi-4-mini-instruct (8bit), CoT, 8 votes | 3.8B | 92.5% | 8 |
| Phi-4-mini-instruct (8bit), CoT, greedy | 3.8B | 90.5% | 1 |
| Qwen3-1.7B (bf16), CoT, 8 votes | 1.7B | 86.5% | 8 |
| Math-1.5B + Qwen3-1.7B pooled ensemble | 3.2B | 90.5% | ~9.4 |

Honest reading: on 200 questions the 95% interval is about +/-3.5 points, so
Sevak **matches** Phi-4-mini with 8 votes (at 40% of the size and ~half the
samples) and is ahead of Phi-4-mini greedy; it does not clearly beat it. The
4->8 adaptive number is computed from the first 4 / all 8 samples of the
held-out n=8 run. Phi-4-mini's model card reports 88.6% (8-shot, Microsoft's
harness); not comparable to our harness, which is why we ran it ourselves.

Tuning-split findings (rows 0-99, Qwen2.5-1.5B-Instruct unless noted):

| Change | Acc |
|--------|-----|
| Original JSON tool-agent protocol | 35% |
| Plain chain-of-thought, `\boxed{}` answer | 75% |
| Program-of-thought (write Python, run it) | 66% |
| Qwen3-1.7B 4bit -> bf16 (8 votes) | 83% -> 88% |
| Qwen3-1.7B thinking mode, budget-forced | no gain on hard questions, 4x slower |

`python -m scripts.eval_sevak --offset 100 --limit 200` reproduces the top row
(`--stop_at` sets the confidence threshold). `scripts/ablate_gsm8k.py` runs any
model x protocol x votes config. MMLU/HumanEval: see the scoreboard above (`scripts/ablate_mmlu.py`, `scripts/eval_mmlu_gate.py`,
`scripts/ablate_humaneval.py`).

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
