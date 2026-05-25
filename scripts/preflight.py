"""Pre-push sanity check.

Runs `classify.py` end-to-end and asserts the last stdout line parses as a
Python literal list of exactly 24 entries from {Positive, Negative, Neutral}.
Exits 0 on success, non-zero on any deviation. Run this before every push.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

VALID_LABELS = {"Positive", "Negative", "Neutral"}
REPO_ROOT = Path(__file__).resolve().parent.parent
CLASSIFY_PY = REPO_ROOT / "classify.py"


def main() -> int:
    print(f"[preflight] running {CLASSIFY_PY}")
    result = subprocess.run(
        [sys.executable, str(CLASSIFY_PY)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=25 * 60,
    )

    if result.returncode != 0:
        print(f"[preflight] non-zero exit: {result.returncode}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        return result.returncode

    stdout_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if not stdout_lines:
        print("[preflight] FAIL: no stdout produced", file=sys.stderr)
        return 1

    last = stdout_lines[-1]
    print(f"[preflight] last line: {last[:120]}{'...' if len(last) > 120 else ''}")

    try:
        parsed = ast.literal_eval(last)
    except (SyntaxError, ValueError) as e:
        print(f"[preflight] FAIL: last line not a Python literal: {e}", file=sys.stderr)
        return 1

    if not isinstance(parsed, list):
        print(f"[preflight] FAIL: last line is {type(parsed).__name__}, expected list", file=sys.stderr)
        return 1
    if len(parsed) != 24:
        print(f"[preflight] FAIL: list length {len(parsed)}, expected 24", file=sys.stderr)
        return 1
    bad = [p for p in parsed if p not in VALID_LABELS]
    if bad:
        print(f"[preflight] FAIL: invalid labels: {bad[:5]}", file=sys.stderr)
        return 1

    counts = {lbl: parsed.count(lbl) for lbl in VALID_LABELS}
    print(f"[preflight] OK: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
