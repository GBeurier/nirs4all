"""Owner defects the SDK cannot fix, declared once as strict version-bounded expectations.

Each entry names the published owner versions that have the defect, its exact
exception and the tests it breaks. The tests are not edited and still run: the
expectation is strict, so any other failure fails and an unexpected pass fails.
Remove an entry once the owner release that fixes it is the declared floor.
"""

from __future__ import annotations

from importlib.metadata import version

import pytest
from packaging.version import Version


def _dagml_refuses_methods_standard_scale_ddof() -> pytest.MarkDecorator | None:
    """DAG-ML <= 0.4.2 refuses classifier states written by Methods >= 1.3.15.

    Methods ABI 2.22 records ``ddof`` in the N4ME state of
    ``preprocessing.scaling.standard_scale``. DAG-ML's bounded N4MC validator
    (dag-ml-core ``methods_classification_state::validate_encoder``) accepts only
    ``{with_mean, with_std}``, so the portable export of every multimodal classifier
    whose recipe scales a source raises "classifier learned state differs from its
    signed recipe, class order or bounded wire format".
    """
    if Version(version("nirs4all-methods")) < Version("1.3.15") or Version(version("dag-ml")) > Version("0.4.2"):
        return None
    import dag_ml

    return pytest.mark.xfail(
        raises=dag_ml.DagMlRuntimeError,
        strict=True,
        reason="owner defect: DAG-ML <= 0.4.2 refuses the standard_scale ddof parameter that Methods >= 1.3.15 writes in classifier states",
    )


# (test module, test function) -> parameters that select the affected cases (empty: every case).
_STANDARD_SCALE_DDOF_TESTS: dict[tuple[str, str], dict[str, object]] = {
    ("test_structural_hpo_classification", "test_direct_native_classifier_and_fixed_run_preserve_typed_labels_and_probabilities"): {},
    ("test_structural_hpo_classification", "test_every_topology_matches_independent_train_only_grouped_probability_oracle"): {},
    ("test_structural_hpo_classification", "test_forced_late_archive_preserves_test_scope_or_artifact_only_refit"): {},
    ("test_structural_hpo_classification", "test_resume_drift_refuses_before_callbacks_and_preserves_checkpoint_bytes"): {},
    ("test_structural_hpo_classification", "test_fixed_learned_late_uses_native_preflight_and_no_optimizer"): {},
    ("test_structural_hpo_classification", "test_large_original_int64_labels_survive_fixed_archive_replay"): {},
    ("test_methods_mixed_metadata", "test_portable_and_fresh_process_replay_preserve_mixed_categories"): {"classification": True},
    # Opt-in installed-replay gate (NIRS4ALL_CLASSIFICATION_INSTALLED_PYTHON); skipped by default.
    ("test_structural_classification_installed", "test_fresh_installed_late_classification_replays_only_actual_payloads"): {},
}


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    marker = _dagml_refuses_methods_standard_scale_ddof()
    if marker is None:
        return
    for item in items:
        if not isinstance(item, pytest.Function):
            continue
        selection = _STANDARD_SCALE_DDOF_TESTS.get((item.module.__name__.rsplit(".", 1)[-1], item.originalname))
        if selection is None:
            continue
        parameters = item.callspec.params if hasattr(item, "callspec") else {}
        if all(parameters.get(name) == value for name, value in selection.items()):
            item.add_marker(marker)
