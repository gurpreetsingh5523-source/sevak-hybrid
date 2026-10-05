#!/bin/bash
# Sevak-Hybrid: One-command setup and test
# Run: bash setup.sh

set -e

echo "========================================"
echo "Sevak-Hybrid: Beat Phi-4 Mini Setup"
echo "========================================"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>/dev/null || python --version 2>/dev/null)
echo "Python: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "[1/5] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv 2>/dev/null || python -m venv .venv
fi

# Activate
echo "[2/5] Activating virtual environment..."
source .venv/bin/activate 2>/dev/null || .venv/Scripts/activate

# Install dependencies
echo "[3/5] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Run tests
echo ""
echo "[4/5] Running component tests..."
python test_all.py

# Show next steps
echo ""
echo "[5/5] Setup complete!"
echo ""
echo "========================================"
echo "NEXT STEPS:"
echo "========================================"
echo ""
echo "1. Run GSM8K benchmark (20 samples):"
echo "   python -m src.eval_gsm8k --limit 20"
echo ""
echo "2. Run HumanEval benchmark (5 samples):"
echo "   python -m src.eval_humaneval --limit 5"
echo ""
echo "3. Run MMLU benchmark (50 samples):"
echo "   python -m src.eval_mmlu --limit 50"
echo ""
echo "4. Bootstrap training data:"
echo "   python -m scripts.bootstrap_tool_data --limit 200 --n 2"
echo ""
echo "5. Fine-tune with LoRA:"
echo "   python -m scripts.sft_lora --data data/tool_gsm8k.jsonl --epochs 1"
echo ""
echo "========================================"
