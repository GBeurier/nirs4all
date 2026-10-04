"""Fresh installed late-classifier replay with FIT/HPO and workspace removed."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import numpy as np
import pytest

import nirs4all
from nirs4all.pipeline.dagml.structural_tuning import _prepare_structure
from tests.integration.api.classification_oracle import topology_probabilities
from tests.integration.api.test_structural_hpo_classification import _run, _sequence, example
from tests.integration.api.test_structural_hpo_preprocessing_chains import _enqueue_recipes


@pytest.mark.parametrize("numeric", [False, True])
def test_fresh_installed_late_classification_replays_only_actual_payloads(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, numeric: bool) -> None:
    installed = os.environ.get("NIRS4ALL_CLASSIFICATION_INSTALLED_PYTHON")
    if not installed:
        if os.environ.get("NIRS4ALL_REQUIRE_CLASSIFICATION_INSTALLED") == "1":
            pytest.fail("mandatory installed proof requires NIRS4ALL_CLASSIFICATION_INSTALLED_PYTHON")
        pytest.skip("fresh installed child is supplied by the qualification gate")
    monkeypatch.setenv("N4A_DAGML_INPROCESS", "1")
    cohort, new, pipeline = example.make_dataset(numeric=numeric), example.make_dataset(29, prediction=True), example.make_pipeline()
    tuning = {**example.make_tuning(tmp_path / "study"), "n_trials": 1}
    prepared = _prepare_structure(pipeline, cohort, tuning, {})
    entry = next(entry for entry in prepared["catalogue"]["entries"] if _sequence(entry, pipeline) is pipeline[1]["_or_"][2])
    _enqueue_recipes(monkeypatch, prepared["catalogue"], [entry])
    train = np.flatnonzero(np.asarray(cohort.partitions) == "train")
    with _run(pipeline, cohort, tmp_path / "workspace", tuning) as result:
        probabilities = topology_probabilities(pipeline[1]["_or_"][2], cohort, train, new, np.arange(len(new.sample_ids)), result.tuning_best_params, prepared["classification"])
        expected = np.asarray(prepared["classification"]["label_names"])[probabilities.argmax(axis=1)]
        archive = result.export(tmp_path / "late.n4a")
    shutil.rmtree(tmp_path / "study")
    if (tmp_path / "workspace").exists():
        shutil.rmtree(tmp_path / "workspace")
    (tmp_path / "cohort.json").write_text(json.dumps(new.to_dict(), allow_nan=False))
    root = Path(nirs4all.__file__).resolve().parent
    names = ["pipeline/dagml/structural_classification.py", "pipeline/dagml/methods_classification.py", "pipeline/dagml/methods_multimodal.py",
             "pipeline/dagml/structural_tuning.py", "pipeline/dagml/result.py", "pipeline/dagml/core_archive_replay.py", "operators/models/multimodal.py"]
    proof = {"prediction": expected.tolist(), "classification": prepared["classification"],
             "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
             "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}
    (tmp_path / "expected.json").write_text(json.dumps(proof, allow_nan=False))
    script = textwrap.dedent("""\
        import hashlib, json, pathlib, sys
        import dag_ml, numpy as np, nirs4all
        from n4m import MultimodalClassifierPipeline
        from n4m.roles import RolePipeline
        from n4m.model_selection.optimizer import Optimizer
        from nirs4all_io import MultimodalDataset
        from nirs4all.pipeline.dagml.host_search_checkpoint import HostSearchOptimizer
        expected = json.loads(pathlib.Path(sys.argv[3]).read_text())
        root = pathlib.Path(nirs4all.__file__).resolve().parent
        assert 'site-packages' in root.parts, root
        for name, digest in expected['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
        archive = pathlib.Path(sys.argv[1])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected['archive_sha256']
        def forbidden(*args, **kwargs):
            raise AssertionError('classifier replay reached FIT/HPO')
        MultimodalClassifierPipeline.fit = forbidden
        RolePipeline.fit = forbidden
        Optimizer.__init__ = forbidden
        Optimizer.load = classmethod(forbidden)
        HostSearchOptimizer.__init__ = forbidden
        for name in ('run_host_hpo_search_in_process', 'execute_training', 'prepare_host_hpo_topology_catalogue', 'resolve_host_hpo_structural_winner'):
            setattr(dag_ml, name, forbidden)
        replay = dag_ml.replay_loaded_predictor_package
        def checked_replay(*args, **kwargs):
            outcome = replay(*args, **kwargs)
            outputs = outcome.to_dict()['outputs']
            assert len(outputs) == 1 and 'artifact_only' not in outputs[0] and 'refit_test_cohort' not in outputs[0]
            blocks = outputs[0]['predictions'] + outputs[0]['aggregated_predictions']
            assert blocks and all(block['partition'] == 'final' and block.get('fold_id') is None and block['producer_port'] == 'y_hat' for block in blocks)
            return outcome
        dag_ml.replay_loaded_predictor_package = checked_replay
        nirs4all.run = forbidden
        cohort = MultimodalDataset.from_dict(json.loads(pathlib.Path(sys.argv[2]).read_text()))
        actual = nirs4all.predict(archive, cohort, engine='dag-ml')
        np.testing.assert_array_equal(actual.y_pred.ravel(), expected['prediction'])
        assert actual.metadata['classification'] == expected['classification']
        assert actual.metadata['training_performed'] is False
        malformed = MultimodalDataset({name:source for name,source in cohort.sources.items() if name != 'series'},
            sample_ids=cohort.sample_ids, partitions=cohort.partitions, name=cohort.name)
        try:
            nirs4all.predict(archive, malformed, engine='dag-ml')
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('excluded original raw source was not required')
        print(json.dumps({'installed':str(root), 'prediction':actual.y_pred.ravel().tolist(), 'training_performed':False}))
    """)
    (tmp_path / "replay.py").write_text(script)
    environment = {**os.environ, "PYTHONPATH": "", "N4A_DAGML_INPROCESS": "1"}
    environment.pop("N4A_ENGINE", None)
    process = subprocess.run([installed, "-I", str(tmp_path / "replay.py"), str(archive), str(tmp_path / "cohort.json"), str(tmp_path / "expected.json")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=300)
    assert process.returncode == 0, process.stdout + process.stderr
    document = json.loads(process.stdout.strip().splitlines()[-1])
    assert document["prediction"] == expected.tolist() and document["training_performed"] is False
