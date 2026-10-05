"""Answer normalization, extraction, and voting utilities."""

import re
from collections import Counter

from .tools import run_python


def normalize_answer(x) -> str:
    """Normalize an answer string for comparison."""
    if x is None:
        return ""

    s = str(x).strip().lower()
    s = s.replace("$", "")
    s = s.replace("%", "")
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s)
    return s


def extract_gsm8k_answer(text: str) -> str:
    """Extract the final answer from a GSM8K answer string.

    GSM8K answers end with #### <number>.
    """
    if "####" in text:
        return text.split("####")[-1].strip()

    # Fallback: extract last number
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return nums[-1]

    return text.strip()


def extract_number(x) -> str:
    """Reduce an answer to its last number, canonicalised ("18.0" -> "18")."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", normalize_answer(x))
    if not nums:
        return ""
    n = float(nums[-1])
    return str(int(n)) if n == int(n) else str(n)


def majority_vote(answers: list) -> str | None:
    """Return the most common normalized answer."""
    normalized = [
        normalize_answer(a)
        for a in answers
        if a is not None and str(a).strip() != ""
    ]

    if not normalized:
        return None

    return Counter(normalized).most_common(1)[0][0]


def verify_code(code: str, tests: str, timeout: int = 8):
    """Verify generated code by running it with test cases.

    Returns:
        ok: bool, whether tests passed.
        result: dict with stdout, stderr, returncode.
    """
    full_code = code + "\n\n" + tests
    result = run_python(full_code, timeout=timeout)
    return result["ok"], result
