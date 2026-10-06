#!/bin/bash
# JEV-style System-1 MMLU scoring, after the general sweep finishes.
cd "$(dirname "$0")/.."
while pgrep -f run_general_tune.sh >/dev/null; do sleep 30; done
PY=.venv/bin/python
for M in Qwen/Qwen2.5-1.5B-Instruct mlx-community/Qwen3-1.7B-bf16 mlx-community/Qwen3.5-2B-MLX-bf16 \
         mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 \
         mlx-community/Phi-4-mini-instruct-8bit; do
  $PY -m scripts.ablate_mmlu --model $M --protocol logit --limit 400 2>&1 | grep -E '^\{|Error'
done
