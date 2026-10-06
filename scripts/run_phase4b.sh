#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f run_phase4.sh >/dev/null; do sleep 20; done
PY=.venv/bin/python
run() { $PY -m scripts.ablate_gsm8k --limit 100 "$@" 2>&1 | grep -E '^\{|Error' ; }
run --model mlx-community/Phi-4-mini-instruct-8bit --protocol cot
run --model mlx-community/Phi-4-mini-instruct-8bit --protocol cot --n 8 --temp 0.7
