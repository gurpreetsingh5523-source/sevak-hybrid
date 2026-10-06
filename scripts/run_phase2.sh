#!/bin/bash
# Phase 2: test-time compute levers on the same 100 dev questions.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { $PY -m scripts.ablate_gsm8k --limit 100 "$@" 2>&1 | grep -E '^\{' ; }
run --model mlx-community/Qwen3-1.7B-4bit --protocol cot --thinking --max_tokens 3072
run --model mlx-community/Qwen2.5-Math-1.5B-Instruct-bf16 --protocol cot
run --model Qwen/Qwen2.5-1.5B-Instruct --protocol cot --n 8 --temp 0.7
run --model mlx-community/Qwen3-1.7B-4bit --protocol cot --n 8 --temp 0.7
run --model mlx-community/Phi-4-mini-instruct-4bit --protocol cot --n 8 --temp 0.7
run --model Qwen/Qwen2.5-1.5B-Instruct --protocol pot
