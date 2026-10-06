#!/bin/bash
# Phase 1: protocol x model sweep on the same 100 dev questions, single path greedy.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { $PY -m scripts.ablate_gsm8k --limit 100 "$@" 2>&1 | grep -E '^\{' ; }
run --model Qwen/Qwen2.5-1.5B-Instruct --protocol json
run --model Qwen/Qwen2.5-1.5B-Instruct --protocol cot
run --model Qwen/Qwen2.5-1.5B-Instruct --protocol pot
run --model mlx-community/Qwen3-1.7B-4bit --protocol cot
run --model mlx-community/Qwen3-1.7B-4bit --protocol pot
run --model mlx-community/Qwen3.5-2B-4bit --protocol cot
run --model mlx-community/Qwen3.5-2B-4bit --protocol pot
run --model mlx-community/Phi-4-mini-instruct-4bit --protocol cot
