"""Run all plain test_* functions without requiring pytest.

This is a convenience runner for environments where `pytest` has not been
installed yet. The same tests remain pytest-compatible.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def load_module(path: Path):
    """Import one test file from its path."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    """Discover, run, and summarize all no-argument test functions."""
    failures: list[tuple[str, BaseException, str]] = []
    total = 0

    for path in sorted(TEST_DIR.glob("test_*.py")):
        module = load_module(path)
        for name, func in sorted(inspect.getmembers(module, inspect.isfunction)):
            if not name.startswith("test_"):
                continue
            total += 1
            label = f"{path.name}::{name}"
            try:
                func()
                print(f"PASS {label}")
            except BaseException as exc:
                failures.append((label, exc, traceback.format_exc()))
                print(f"FAIL {label}: {exc}")

    print(f"\nRan {total} tests: {total - len(failures)} passed, {len(failures)} failed")
    if failures:
        print("\nFailure details:")
        for label, _, trace in failures:
            print(f"\n--- {label} ---")
            print(trace)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
