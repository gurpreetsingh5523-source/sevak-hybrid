"""Robust JSON extraction from model outputs."""

import json
import re


def _loads(s: str):
    """json.loads that also accepts what small models actually emit:
    raw newlines inside strings, and Python triple-quoted code blocks."""
    try:
        return json.loads(s, strict=False)
    except Exception:
        pass
    if '"""' in s or "'''" in s:
        fixed = re.sub(
            r'("""|\'\'\')(.*?)\1',
            lambda m: json.dumps(m.group(2)),
            s,
            flags=re.S,
        )
        return json.loads(fixed, strict=False)
    raise ValueError("not json")


def extract_json(text: str) -> dict | None:
    """Extract a JSON object from model output text.

    Tries multiple strategies:
    1. Direct JSON parse
    2. Fenced JSON block (```json {...} ```)
    3. First {...} block
    """
    if not text:
        return None

    text = text.strip()

    # Strategy 1: Direct JSON
    try:
        return _loads(text)
    except Exception:
        pass

    # Strategy 2: Fenced JSON block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        try:
            return _loads(fenced.group(1))
        except Exception:
            pass

    # Strategy 3: First {...} block (handle nested braces)
    start = text.find("{")
    if start == -1:
        return None

    # Find matching closing brace
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end != -1:
        candidate = text[start:end + 1]
        try:
            return _loads(candidate)
        except Exception:
            pass

    return None
