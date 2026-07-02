"""Golden fixtures for Python ``RtResult`` / ``RtError`` runtime wire envelopes (B-018).

These fixtures mirror the Web W37 contract scenarios at the neutral envelope
level: success results carry top-level ``schema_version`` and no diagnostics,
fallback results carry explicit ``RtError`` diagnostics, and serialized errors
are detail-free dictionaries with no error-specific schema version.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from packaging.requirements import Requirement

from nirs4all.api.result import RunResult
from nirs4all.data.predictions import Predictions
from nirs4all.pipeline.dagml.errors import DagMlUnsupported
from nirs4all.pipeline.dagml.rt import RtError

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = Path(__file__).with_name("fixtures") / "runtime"
METRICS = {"rmse": 0.0, "r2": 1.0, "mae": 0.0}


def _read_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _declared_dependencies(requirements: list[str]) -> dict[str, Requirement]:
    return {Requirement(spec).name: Requirement(spec) for spec in requirements}


def _declared_dependency_names(requirements: list[str]) -> set[str]:
    return set(_declared_dependencies(requirements))


def _requirements(filename: str) -> dict[str, Requirement]:
    lines = []
    for raw in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return _declared_dependencies(lines)


def _requirements_names(filename: str) -> set[str]:
    return set(_requirements(filename))


def _pyproject() -> dict[str, Any]:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _score_set() -> dict[str, Any]:
    """Schema-shaped ScoreSet matching the Python success fixture."""
    reports: list[dict[str, Any]] = []
    for fold_id, row_count in (("avg", 4), ("fold-1", 2), ("fold-2", 2)):
        reports.append(
            {
                "prediction_id": f"pred:model:compat.0:variant:base:{fold_id}",
                "variant_id": "variant:base",
                "variant_label": None,
                "producer_node": "model:compat.0",
                "partition": "validation",
                "fold_id": fold_id,
                "level": "sample",
                "row_count": row_count,
                "target_width": 1,
                "target_names": ["moisture"],
                "metrics": METRICS,
            }
        )
    reports.append(
        {
            "prediction_id": "pred:model:compat.0:variant:base:final",
            "variant_id": "variant:base",
            "variant_label": None,
            "producer_node": "model:compat.0",
            "partition": "test",
            "fold_id": "final",
            "level": "sample",
            "row_count": 2,
            "target_width": 1,
            "target_names": ["moisture"],
            "metrics": METRICS,
        }
    )
    return {"schema_version": 1, "plan_id": "plan:py-rt-golden", "reports": reports}


def _add_prediction(
    predictions: Predictions,
    *,
    partition: str,
    fold_id: str,
    sample_indices: list[int],
    values: list[float],
    val_score: float | None = None,
    test_score: float | None = None,
) -> None:
    predictions.add_prediction(
        dataset_name="rt-golden-dataset",
        config_name="variant:base",
        model_name="rt-golden-pls",
        partition=partition,
        fold_id=fold_id,
        sample_indices=sample_indices,
        y_true=np.array(values, dtype=float),
        y_pred=np.array(values, dtype=float),
        val_score=val_score,
        test_score=test_score,
        scores=METRICS,
        metric="rmse",
        task_type="regression",
    )


def _golden_result(*, engine: str = "dag-ml") -> RunResult:
    """Build the deterministic in-memory RunResult projected by the fixtures."""
    predictions = Predictions()
    _add_prediction(predictions, partition="validation", fold_id="avg", sample_indices=[0, 1, 2, 3], values=[1.0, 2.0, 3.0, 4.0], val_score=0.0)
    _add_prediction(predictions, partition="validation", fold_id="fold-1", sample_indices=[0, 1], values=[1.0, 2.0], val_score=0.0)
    _add_prediction(predictions, partition="validation", fold_id="fold-2", sample_indices=[2, 3], values=[3.0, 4.0], val_score=0.0)
    _add_prediction(predictions, partition="test", fold_id="final", sample_indices=[4, 5], values=[5.0, 6.0], test_score=0.0)
    predictions.flush()

    result = RunResult(predictions=predictions, per_dataset={"rt-golden-dataset": {"engine": engine}})
    result._dagml_refit_artifacts = []  # noqa: SLF001
    if engine == "dag-ml":
        result._dagml_score_set = _score_set()  # noqa: SLF001
    return result


def _unsupported_shape_error() -> RtError:
    return RtError.from_dagml_error(DagMlUnsupported("branch duplication merge predictions is not covered"), verb="run")


def test_rt_result_success_matches_fixture() -> None:
    payload = _golden_result().to_rt_result().to_dict()

    assert payload == _read_fixture("rt_result.success.v1.json")
    assert "diagnostics" not in payload
    assert payload["reports"] == _score_set()["reports"]


def test_rt_result_legacy_fallback_matches_fixture() -> None:
    result = _golden_result(engine="legacy")
    result._rt_diagnostics = [_unsupported_shape_error()]  # noqa: SLF001

    payload = result.to_rt_result().to_dict()

    assert payload == _read_fixture("rt_result.legacy_fallback.v1.json")
    assert payload["manifest"]["engine"] == "legacy"
    assert payload["reports"] == []
    assert payload["diagnostics"] == [_read_fixture("rt_error.unsupported_shape.v1.json")]


@pytest.mark.parametrize(
    ("fixture_name", "error"),
    [
        ("rt_error.scheduler_fallback.v1.json", RtError("run", "runtime_error", "execute_campaign_phase_json crashed: scheduler boom", mitigation="Cross-validation re-ran through the libn4m chain over dag-ml folds; results are valid, but the dag-ml scheduler did not run this phase.")),
        ("rt_error.strict_scheduler_refusal.v1.json", RtError("run", "runtime_error", "execute_campaign_phase_json crashed: scheduler boom", mitigation="Set allow_fallback=true to permit a diagnosed libn4m fallback, or keep allow_fallback=false to fail closed.")),
        ("rt_error.unsupported_shape.v1.json", _unsupported_shape_error()),
    ],
)
def test_rt_error_wire_goldens_match_fixtures(fixture_name: str, error: RtError) -> None:
    payload = error.to_dict()

    assert payload == _read_fixture(fixture_name)
    assert "schema_version" not in payload
    assert "detail" not in payload


def test_parity_environment_dependencies_are_declared() -> None:
    """Parity import gates are backed by install metadata, not ad hoc local state."""
    pyproject = _pyproject()
    project_deps = _declared_dependencies(pyproject["project"]["dependencies"])
    project_dep_names = set(project_deps)
    optional = pyproject["project"]["optional-dependencies"]
    dev_deps = _declared_dependency_names(optional["dev"])
    explain_deps = _declared_dependency_names(optional["explain"])

    assert "referencing" in project_dep_names
    assert "jsonschema" in project_dep_names
    assert str(project_deps["jsonschema"].specifier) == ">=4.18.0"
    assert "shap" in explain_deps
    assert "shap" in dev_deps
    for filename in ("requirements.txt", "requirements-test.txt", "requirements-examples.txt"):
        requirements = _requirements(filename)
        assert "referencing" in requirements
        assert str(requirements["jsonschema"].specifier) == ">=4.18.0"
    assert "shap" in _requirements_names("requirements-test.txt")
    assert "shap" in _requirements_names("requirements-examples.txt")


def _find_sibling_schema_dir() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent.parent / "nirs4all-ecosystem" / "docs" / "contracts" / "runtime"
        if candidate.is_dir():
            return candidate
    return None


def _schema_registry(schema_dir: Path):
    from referencing import Registry, Resource

    resources = []
    for path in schema_dir.glob("*.v1.schema.json"):
        schema = json.loads(path.read_text(encoding="utf-8"))
        resources.append((schema["$id"], Resource.from_contents(schema)))

    dagml_contracts = schema_dir.parent.parent.parent.parent / "dag-ml" / "docs" / "contracts"
    for name in ("score_set.schema.json", "selection_decision.schema.json"):
        path = dagml_contracts / name
        if path.is_file():
            schema = json.loads(path.read_text(encoding="utf-8"))
            resources.append((schema["$id"], Resource.from_contents(schema)))

    return Registry().with_resources(resources)


@pytest.mark.parametrize(
    ("schema_id", "fixture_name"),
    [
        ("https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_result.v1.schema.json", "rt_result.success.v1.json"),
        ("https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_result.v1.schema.json", "rt_result.legacy_fallback.v1.json"),
        ("https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_error.v1.schema.json", "rt_error.scheduler_fallback.v1.json"),
        ("https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_error.v1.schema.json", "rt_error.strict_scheduler_refusal.v1.json"),
        ("https://github.com/GBeurier/nirs4all-ecosystem/schemas/runtime/rt_error.v1.schema.json", "rt_error.unsupported_shape.v1.json"),
    ],
)
def test_runtime_goldens_validate_against_current_schemas(schema_id: str, fixture_name: str) -> None:
    pytest.importorskip("jsonschema")
    pytest.importorskip("referencing")
    schema_dir = _find_sibling_schema_dir()
    if schema_dir is None:
        pytest.skip("sibling nirs4all-ecosystem runtime schemas not checked out")

    import jsonschema

    registry = _schema_registry(schema_dir)
    schema = registry.get_or_retrieve(schema_id).value.contents
    jsonschema.Draft202012Validator(schema, registry=registry).validate(_read_fixture(fixture_name))
