"""Test all core components without downloading a large model.

This script verifies that:
1. Config loads correctly
2. JSON parser works on various inputs
3. Python sandbox executes safely
4. Verifier functions work
5. Agent logic is sound (mocked)

Run this BEFORE running real benchmarks.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.parser import extract_json
from src.tools import run_python
from src.verifier import normalize_answer, extract_gsm8k_answer, majority_vote


def test_config():
    print("=" * 60)
    print("TEST 1: Config Loading")
    print("=" * 60)
    cfg = load_config("config.yaml")
    assert "model_name" in cfg, "model_name missing"
    assert "n_paths" in cfg, "n_paths missing"
    print(f"  model_name: {cfg['model_name']}")
    print(f"  n_paths: {cfg['n_paths']}")
    print("  PASS\n")


def test_parser():
    print("=" * 60)
    print("TEST 2: JSON Parser")
    print("=" * 60)

    # Test 1: Direct JSON
    result = extract_json('{"tool": "final", "answer": "42"}')
    assert result == {"tool": "final", "answer": "42"}, f"Failed direct: {result}"
    print("  Direct JSON: PASS")

    # Test 2: Fenced JSON
    result = extract_json('```json\n{"tool": "python", "code": "print(1)"}\n```')
    assert result == {"tool": "python", "code": "print(1)"}, f"Failed fenced: {result}"
    print("  Fenced JSON: PASS")

    # Test 3: Embedded JSON
    result = extract_json('Some text before {\n  "tool": "final",\n  "answer": "hello"\n} and after')
    assert result == {"tool": "final", "answer": "hello"}, f"Failed embedded: {result}"
    print("  Embedded JSON: PASS")

    # Test 4: Invalid input
    result = extract_json("This is not JSON")
    assert result is None, f"Failed invalid: {result}"
    print("  Invalid input: PASS")

    # Test 5: Empty input
    result = extract_json("")
    assert result is None, f"Failed empty: {result}"
    print("  Empty input: PASS")

    # Test 6: Python triple-quoted code inside JSON (what Qwen-1.5B emits)
    result = extract_json('{"tool": "python", "code": """\nprint(1)\n"""}')
    assert result == {"tool": "python", "code": "\nprint(1)\n"}, f"Failed triple-quote: {result}"
    print("  Triple-quoted code: PASS")

    print("  ALL PARSER TESTS PASS\n")


def test_sandbox():
    print("=" * 60)
    print("TEST 3: Python Sandbox")
    print("=" * 60)

    # Test 1: Simple execution
    result = run_python("print(2 + 2)")
    assert result["ok"] is True, f"Failed simple exec: {result}"
    assert "4" in result["stdout"], f"Wrong output: {result['stdout']}"
    print("  Simple execution: PASS")

    # Test 2: Math problem
    result = run_python("""
import math
x = math.sqrt(16)
print(x)
""")
    assert result["ok"] is True, f"Failed math: {result}"
    assert "4.0" in result["stdout"], f"Wrong output: {result['stdout']}"
    print("  Math execution: PASS")

    # Test 3: Error handling
    result = run_python("print(undefined_var)")
    assert result["ok"] is False, f"Should fail: {result}"
    print("  Error handling: PASS")

    # Test 4: Timeout
    result = run_python("while True: pass", timeout=1)
    assert result["ok"] is False, f"Should timeout: {result}"
    assert "Timeout" in result["stderr"] or result["returncode"] == -1, f"Wrong timeout: {result}"
    print("  Timeout handling: PASS")

    # Test 5: Memory limit (try to allocate 1GB)
    result = run_python("x = 'x' * (1024 * 1024 * 1024)")
    # Should fail due to memory limit
    print(f"  Memory limit: {'PASS' if not result['ok'] else 'WARN (may need stricter limits)'}\n")


def test_verifier():
    print("=" * 60)
    print("TEST 4: Verifier Functions")
    print("=" * 60)

    # Test normalize_answer
    assert normalize_answer("  $1,234.56  ") == "1234.56"
    assert normalize_answer("Hello World") == "hello world"
    print("  normalize_answer: PASS")

    # Test extract_gsm8k_answer
    assert extract_gsm8k_answer("Some reasoning #### 42") == "42"
    assert extract_gsm8k_answer("The answer is 100") == "100"
    print("  extract_gsm8k_answer: PASS")

    # Test majority_vote
    assert majority_vote(["A", "B", "A", "C"]) == "a"
    assert majority_vote(["42", "42", "43"]) == "42"
    assert majority_vote([]) is None
    print("  majority_vote: PASS")

    print("  ALL VERIFIER TESTS PASS\n")


def test_agent_mock():
    print("=" * 60)
    print("TEST 5: Agent Logic (Mocked)")
    print("=" * 60)

    # Mock LLM that always returns correct JSON
    class MockLLM:
        def __init__(self, responses):
            self.responses = responses
            self.idx = 0

        def chat(self, messages, **kwargs):
            resp = self.responses[self.idx % len(self.responses)]
            self.idx += 1
            return resp

    from src.agent import solve

    # Test: Direct final answer
    mock = MockLLM(['{"tool": "final", "answer": "42"}'])
    cfg = {"n_paths": 1, "max_steps": 3, "temperature": 0, "top_p": 0.95, "max_new_tokens": 64, "timeout": 8}
    result = solve("What is 2+2?", mock, cfg)
    assert result == ["42"], f"Failed direct answer: {result}"
    print("  Direct final answer: PASS")

    # Test: Tool use then final
    mock = MockLLM([
        '{"tool": "python", "code": "print(2+2)"}',
        '{"tool": "final", "answer": "4"}',
    ])
    result = solve("What is 2+2?", mock, cfg)
    assert result == ["4"], f"Failed tool use: {result}"
    print("  Tool use + final: PASS")

    print("  ALL AGENT TESTS PASS\n")


def test_regressions():
    print("=" * 60)
    print("TEST 6: Scoring regressions")
    print("=" * 60)
    from src.verifier import extract_number, verify_code

    # GSM8K: equivalent numeric forms must match gold
    assert extract_number("18.0") == "18"
    assert extract_number("$1,234 dollars") == "1234"
    assert extract_number("no number") == ""
    print("  extract_number: PASS")

    # MMLU: vote is lowercased, scoring must uppercase it
    assert (majority_vote(["B", "B", "A"]) or "").upper() == "B"
    print("  MMLU vote case: PASS")

    # HumanEval: wrong code must FAIL once check() is actually invoked
    tests = "def check(candidate):\n    assert candidate(2) == 4\n"
    ok, _ = verify_code("def f(x):\n    return x + 1\n", tests + "\ncheck(f)\n")
    assert ok is False, "wrong solution passed"
    ok, _ = verify_code("def f(x):\n    return x * 2\n", tests + "\ncheck(f)\n")
    assert ok is True, "right solution failed"
    print("  HumanEval check() invoked: PASS")

    # Agent: unparseable output abstains instead of voting raw text
    from src.agent import solve

    class Garbage:
        def chat(self, messages, **kw):
            return "I think it's 7"

    cfg = {"n_paths": 1, "max_steps": 2, "timeout": 8}
    assert solve("q", Garbage(), cfg) == [""]

    # Agent: out of steps after a successful tool run -> use its stdout
    class ToolOnly:
        def chat(self, messages, **kw):
            return '{"tool": "python", "code": "print(16-3-4)"}'

    assert solve("q", ToolOnly(), cfg) == ["9"]
    print("  Agent abstain on bad JSON: PASS\n")


def main():
    print("\n" + "=" * 60)
    print("SEVAK-HYBRID: COMPONENT TEST SUITE")
    print("=" * 60 + "\n")

    try:
        test_config()
        test_parser()
        test_sandbox()
        test_verifier()
        test_agent_mock()
        test_regressions()

        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run real benchmarks:")
        print("  python -m src.eval_gsm8k --limit 20")
        print("  python -m src.eval_humaneval --limit 5")
        print("  python -m src.eval_mmlu --limit 50")

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
