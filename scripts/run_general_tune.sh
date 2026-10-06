#!/bin/bash
# MMLU + HumanEval tuning sweep (MMLU rows 0-99, HumanEval tasks 0-63).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mm() { $PY -m scripts.ablate_mmlu --limit 100 "$@" 2>&1 | grep -E '^\{|Error' ; }
he() { $PY -m scripts.ablate_humaneval --limit 64 "$@" 2>&1 | grep -E '^\{|Error' ; }
for M in Qwen/Qwen2.5-1.5B-Instruct mlx-community/Qwen3-1.7B-bf16 mlx-community/Qwen3.5-2B-MLX-bf16 \
         mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 \
         mlx-community/Phi-4-mini-instruct-8bit; do
  mm --model $M --protocol direct
  mm --model $M --protocol cot
  he --model $M --protocol greedy
done
