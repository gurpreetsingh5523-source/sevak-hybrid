"""System and user prompts for the hybrid agent."""

SYSTEM_PROMPT = """You are a precise problem solver.
Use the Python tool whenever math, calculation, or code execution is needed.

You must respond ONLY with a valid JSON object.
Do not write markdown.
Do not write explanations outside JSON.

JSON schema:
{
  "thought": "short reasoning",
  "tool": "python" | "final",
  "code": "optional full python code",
  "answer": "optional final answer"
}

Rules:
1. If you need calculation or code execution, set tool="python".
2. If tool="python", provide complete runnable Python code.
3. The Python code should print the useful final output when possible.
4. If you already know the final answer, set tool="final".
5. If tool="final", provide the answer in the "answer" field.
"""


def user_prompt(
    problem: str,
    tool_result: str | None = None,
    error: str | None = None,
) -> str:
    """Build user prompt for the agent loop."""
    if tool_result is None and error is None:
        return f"Problem:\n{problem}\n\nRespond only with JSON."

    parts = [f"Problem:\n{problem}"]

    if tool_result:
        parts.append(f"Tool result:\n{tool_result}")

    if error:
        parts.append(f"Previous output invalid:\n{error}")

    parts.append("Continue and respond only with JSON.")

    return "\n\n".join(parts)
