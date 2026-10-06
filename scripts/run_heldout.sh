#!/bin/bash
# Held-out: rows 100-299 of data/gsm8k_dev.jsonl, never used for tuning.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
H="--offset 100 --limit 200"
$PY -m scripts.eval_sevak $H 2>&1 | grep -E '^\{|Error'
run() { $PY -m scripts.ablate_gsm8k $H "$@" 2>&1 | grep -E '^\{|Error' ; }
run --model mlx-community/Phi-4-mini-instruct-8bit --protocol cot
run --model mlx-community/Phi-4-mini-instruct-8bit --protocol cot --n 8 --temp 0.7
run --model mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 --protocol cot --n 8 --temp 0.7
run --model mlx-community/Qwen3-1.7B-bf16 --protocol cot --n 8 --temp 0.7
