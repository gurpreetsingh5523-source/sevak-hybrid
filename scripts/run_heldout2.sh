#!/bin/bash
# Held-out for MMLU (rows 100-399) and HumanEval (tasks 64-163).
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -m scripts.eval_mmlu_gate --offset 100 --limit 300 --tau 0.7 2>&1 | grep -E '^\{|Error'
$PY -m scripts.ablate_mmlu --model mlx-community/Phi-4-mini-instruct-8bit --protocol cot --offset 100 --limit 300 2>&1 | grep -E '^\{|Error'
he() { $PY -m scripts.ablate_humaneval "$@" 2>&1 | grep -E '^\{|Error' ; }
he --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol select --n 8 --limit 64
he --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol greedy --offset 64 --limit 100
he --model mlx-community/Phi-4-mini-instruct-8bit --protocol greedy --offset 64 --limit 100
he --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol select --n 8 --offset 64 --limit 100
