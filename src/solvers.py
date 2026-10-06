"""Math solving protocols, each returning one candidate answer per sampled path.

- cot:    chain-of-thought, final answer in \\boxed{}
- pot:    program-of-thought, model writes Python, we run it, stdout is the answer
- hybrid: cot paths + pot paths voted together
- json:   the original JSON tool-agent loop (src.agent.solve)
"""

import ast
import re

from .tools import run_python
from .verifier import extract_number

COT_SYSTEM = (
    "You are a careful math tutor. Think step by step, check your arithmetic, "
    "then give the final numeric answer as \\boxed{N}."
)

POT_SYSTEM = (
    "You solve math word problems by writing a short Python program. "
    "Use clear variable names, compute the answer step by step, and print only "
    "the final numeric answer. Reply with a single ```python code block."
)


def extract_boxed(text: str) -> str:
    """Answer from the last \\boxed{...}; falls back to the last number."""
    boxed = re.findall(r"\\boxed\{([^{}]*)\}", text or "")
    if boxed:
        return extract_number(boxed[-1])
    return extract_number(text)


def extract_code(text: str) -> str:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text or "", re.S)
    if blocks:
        return blocks[-1]
    return text or ""


def autoprint_last_expr(code: str) -> str:
    """Small models often end notebook-style with a bare `total` instead of
    print(total). Print a trailing expression, like a REPL would."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        call = last.value
        is_print = isinstance(call, ast.Call) and getattr(call.func, "id", "") == "print"
        if not is_print:
            tree.body[-1] = ast.Expr(
                ast.Call(ast.Name("print", ast.Load()), [last.value], [])
            )
            return ast.unparse(ast.fix_missing_locations(tree))
    return code


def run_program(text: str, timeout: int = 8) -> str:
    """Execute the model's program; the answer is the last number printed."""
    result = run_python(autoprint_last_expr(extract_code(text)), timeout=timeout)
    if not result["ok"]:
        return ""
    return extract_number(result["stdout"])


def cot(problem, llm, n, **gen):
    msgs = [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": problem},
    ]
    return [extract_boxed(t) for t in llm.chat_n(msgs, n=n, **gen)]


def pot(problem, llm, n, timeout=8, **gen):
    msgs = [
        {"role": "system", "content": POT_SYSTEM},
        {"role": "user", "content": problem},
    ]
    return [run_program(t, timeout) for t in llm.chat_n(msgs, n=n, **gen)]


def hybrid(problem, llm, n, timeout=8, **gen):
    """Half the budget on each protocol; their errors are weakly correlated."""
    k = max(1, n // 2)
    return cot(problem, llm, k, **gen) + pot(problem, llm, max(1, n - k), timeout=timeout, **gen)


def json_agent(problem, llm, n, timeout=8, **gen):
    from .agent import solve

    cfg = {"n_paths": n, "max_steps": 3, "timeout": timeout, **gen}
    return [extract_number(c) for c in solve(problem, llm, cfg)]


PROTOCOLS = {"cot": cot, "pot": pot, "hybrid": hybrid, "json": json_agent}
