"""Hybrid agent: model + tool loop + multi-path voting."""

from .parser import extract_json
from .prompts import SYSTEM_PROMPT, user_prompt
from .tools import run_python


def solve(
    problem: str,
    llm,
    cfg: dict,
    return_trajectory: bool = False,
):
    """Solve a problem using the hybrid agent with multi-path voting.

    Args:
        problem: The problem text.
        llm: An LLM instance with a .chat() method.
        cfg: Config dict with keys: n_paths, max_steps, temperature, top_p,
             max_new_tokens, timeout.
        return_trajectory: If True, also return the conversation histories.

    Returns:
        candidates: List of final answer strings (one per path).
        trajectories: List of message histories (only if return_trajectory=True).
    """
    candidates = []
    trajectories = []

    n_paths = int(cfg.get("n_paths", 4))
    max_steps = int(cfg.get("max_steps", 3))
    temperature = float(cfg.get("temperature", 0.2))
    top_p = float(cfg.get("top_p", 0.95))
    max_new_tokens = int(cfg.get("max_new_tokens", 1024))
    timeout = int(cfg.get("timeout", 8))

    for _ in range(n_paths):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        tool_result = None
        error = None
        final_answer = None
        last_stdout = ""

        for step in range(max_steps):
            if step == 0:
                user_content = user_prompt(problem)
            else:
                user_content = user_prompt(
                    problem,
                    tool_result=tool_result,
                    error=error,
                )

            messages.append({"role": "user", "content": user_content})

            raw = llm.chat(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
            )

            messages.append({"role": "assistant", "content": raw})

            parsed = extract_json(raw)

            if not parsed:
                tool_result = None
                error = "Output is not valid JSON."
                continue

            tool = parsed.get("tool")

            if tool == "final":
                final_answer = str(parsed.get("answer", "")).strip()
                break

            if tool == "python":
                code = parsed.get("code", "")
                result = run_python(code, timeout=timeout)
                if result["ok"] and result["stdout"].strip():
                    last_stdout = result["stdout"].strip().splitlines()[-1]

                tool_result = (
                    f"ok={result['ok']}\n"
                    f"stdout={result['stdout']}\n"
                    f"stderr={result['stderr']}"
                )
                error = None
                continue

            tool_result = None
            error = f"Unknown tool: {tool}. Use python or final."

        if final_answer is None:
            # Out of steps: trust the last successful tool output, else abstain
            # so raw model text can't win the vote
            final_answer = last_stdout

        candidates.append(final_answer)
        trajectories.append(messages)

    if return_trajectory:
        return candidates, trajectories

    return candidates
