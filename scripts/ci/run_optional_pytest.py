"""Run an optional pytest slice without failing when it selects no tests.

Framework-specific CI slices can legitimately be empty while a backend is being
retired or reorganised.  Pytest reports that case with exit code 5.  Every other
exit code remains unchanged so collection errors, invalid arguments, interrupts,
and test failures still fail the CI job.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest


def normalize_exit_code(exit_code: int | pytest.ExitCode) -> int:
    """Return success only for pytest's explicit no-tests-collected outcome."""

    if int(exit_code) == int(pytest.ExitCode.NO_TESTS_COLLECTED):
        print("Optional pytest slice selected no tests; continuing.")
        return int(pytest.ExitCode.OK)
    return int(exit_code)


def main(args: Sequence[str] | None = None) -> int:
    """Run pytest with ``args`` and preserve every non-empty result."""

    pytest_args = list(args) if args is not None else None
    return normalize_exit_code(pytest.main(pytest_args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
