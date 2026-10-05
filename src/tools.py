"""Sandboxed Python execution for tool use."""

import os
import resource
import subprocess
import sys
import tempfile


def _set_limits():
    """Set resource limits for the sandboxed process."""
    # Each limit in its own try: RLIMIT_AS is often rejected on macOS, and a
    # single shared try would then silently skip the CPU/FD limits too.
    mem = 512 * 1024 * 1024
    limits = [
        (resource.RLIMIT_AS, (mem, mem)),
        (resource.RLIMIT_CPU, (5, 5)),
        (resource.RLIMIT_NOFILE, (32, 32)),
    ]
    for kind, value in limits:
        try:
            resource.setrlimit(kind, value)
        except Exception:
            pass
    try:
        os.nice(10)
    except Exception:
        pass


def run_python(code: str, stdin: str = "", timeout: int = 8) -> dict:
    """Run Python code in a sandboxed subprocess.

    Returns dict with keys: ok, stdout, stderr, returncode
    """
    code = code or ""

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as f:
        f.write(code)
        path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_set_limits,
        )

        return {
            "ok": proc.returncode == 0,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-8000:],
            "returncode": proc.returncode,
        }

    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "stdout": "",
            "stderr": "Timeout",
            "returncode": -1,
        }

    except Exception as e:
        return {
            "ok": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }

    finally:
        try:
            os.unlink(path)
        except Exception:
            pass
