"""Cold concurrent imports, runnable against an installed wheel without fixtures."""

import json
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("modules", [
    ["nirs4all", "nirs4all.data.config", "nirs4all.pipeline.storage.workspace_store"],
    ["nirs4all.pipeline.storage.workspace_store", "nirs4all.data.config", "nirs4all"],
    ["nirs4all.api", "nirs4all.data.config", "nirs4all.pipeline.storage.workspace_store", "nirs4all"],
    ["nirs4all.controllers", "nirs4all.pipeline.execution.result", "nirs4all.pipeline.steps.parser"],
])
def test_cold_concurrent_imports(modules, tmp_path):
    # A fresh interpreter is essential: pytest's other imports would hide the
    # startup race. In wheel CI this file can be extracted and run standalone;
    # it never discovers a checkout or changes the child interpreter's path.
    code = textwrap.dedent("""\
        import importlib
        import json
        import sys
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        modules = json.loads(sys.argv[1])
        barrier = Barrier(len(modules))
        def load(name):
            barrier.wait(timeout=15)
            module = importlib.import_module(name)
            return module.__name__
        with ThreadPoolExecutor(max_workers=len(modules)) as pool:
            assert list(pool.map(load, modules)) == modules
        from nirs4all.data.config import DatasetConfigs
        from nirs4all.pipeline.storage.workspace_store import WorkspaceStore
        from nirs4all import run, predict
        assert callable(run) and callable(predict)
        assert DatasetConfigs.__name__ == 'DatasetConfigs'
        assert WorkspaceStore.__name__ == 'WorkspaceStore'
        """)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(modules)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_lazy_root_preserves_public_exports(tmp_path):
    code = textwrap.dedent("""\
        import importlib
        import sys
        import nirs4all
        assert not any(name.startswith('nirs4all.') for name in sys.modules)
        assert set(nirs4all.__all__) <= set(dir(nirs4all))
        for name in nirs4all.__all__:
            module_name, attribute = nirs4all._LAZY_EXPORTS[name]
            expected = getattr(importlib.import_module(module_name, 'nirs4all'), attribute)
            assert getattr(nirs4all, name) is expected, name
        from nirs4all import run, predict, PipelineRunner, CONTROLLER_REGISTRY
        assert run is nirs4all.run and predict is nirs4all.predict
        assert PipelineRunner is nirs4all.PipelineRunner
        assert CONTROLLER_REGISTRY is nirs4all.CONTROLLER_REGISTRY
        import nirs4all.pipeline as pipeline
        assert set(pipeline.__all__) <= set(dir(pipeline))
        for name in pipeline.__all__:
            module_name, attribute = pipeline._LAZY_EXPORTS[name]
            expected = getattr(importlib.import_module(module_name, 'nirs4all.pipeline'), attribute)
            assert getattr(pipeline, name) is expected, name
        namespace = {}
        exec('from nirs4all import *', namespace)
        assert all(namespace[name] is getattr(nirs4all, name) for name in nirs4all.__all__)
        try:
            nirs4all.nonexistent_public_export
        except AttributeError:
            pass
        else:
            raise AssertionError('unknown export must raise AttributeError')
        """)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        capture_output=True, text=True, timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
