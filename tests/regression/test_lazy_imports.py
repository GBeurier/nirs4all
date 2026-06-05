"""Regression guard: ``import nirs4all`` must not eagerly load heavy optional deps.

matplotlib (charts) and shap (explainability, which drags numba/llvmlite) are
imported lazily — on first use, not at package import. This keeps cold-import cheap
for pure-inference users. The check runs in a subprocess because, within the test
process, another test may already have imported matplotlib/shap, which would make a
plain ``'matplotlib' in sys.modules`` assertion order-dependent and flaky.

optuna intentionally stays eager (it is light enough and broadly used), so it is not
asserted against here.
"""
import os
import subprocess
import sys

# Heavy modules that must NOT be pulled in by a bare ``import nirs4all``.
LAZY_HEAVY_MODULES = ("matplotlib", "matplotlib.pyplot", "shap", "numba", "llvmlite")


def test_import_nirs4all_does_not_eagerly_load_matplotlib_or_shap() -> None:
    code = (
        "import sys, nirs4all\n"
        f"heavy = [m for m in {LAZY_HEAVY_MODULES!r} if m in sys.modules]\n"
        "print(','.join(heavy))\n"
        "sys.exit(1 if heavy else 0)\n"
    )
    # PYTHONSAFEPATH keeps the cwd off sys.path so the package resolves from its
    # install and the nirs4all/sklearn subpackage cannot shadow real sklearn.
    env = {**os.environ, "PYTHONSAFEPATH": "1"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"import nirs4all eagerly loaded: {result.stdout.strip()}\n{result.stderr}"
    )
