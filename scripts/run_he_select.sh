#!/bin/bash
cd "$(dirname "$0")/.."
PY=.venv/bin/python
he() { $PY -m scripts.ablate_humaneval "$@" 2>&1 | grep -E '^\{|Error' ; }
he --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol select --n 8 --limit 64
he --model mlx-community/Qwen2.5-Coder-1.5B-Instruct-bf16 --protocol select --n 8 --offset 64 --limit 100
he --model mlx-community/Phi-4-mini-instruct-8bit --protocol select --n 8 --offset 64 --limit 100
