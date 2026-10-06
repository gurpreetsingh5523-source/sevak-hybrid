#!/bin/bash
# Phase 3: math-specialised 1.5B + full-precision 1.7B, same 100 tuning questions.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { $PY -m scripts.ablate_gsm8k --limit 100 "$@" 2>&1 | grep -E '^\{|Error' ; }
run --model mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 --protocol cot
run --model mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 --protocol cot --n 8 --temp 0.7
run --model mlx-community/Qwen3-1.7B-bf16 --protocol cot --n 8 --temp 0.7
