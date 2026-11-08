"""Canonical Core ``.n4a`` bridge: one Core-owned fit, sealed bytes, thin glue.

Core's native producer (``nirs4all_core.produce_n4a_native``) is the one and only
fit of this profile. It runs cross-validation, selection and refit once and
returns an already sealed canonical container plus its owner view. Canonical
retrain is the same single fit: ``nirs4all_core.retrain_n4a_native`` reads the
validated recipe and source stored in the container and runs a fresh owner
cross-validation, selection and refit on a new record. This module only validates
request shapes at the public boundary, calls Core, projects the owner scores into
a :class:`RunResult`, and publishes the sealed bytes. It never trains, re-inspects
at export, or invents scientific fingerprints; routing, validation and replay
semantics stay with Core, DAG-ML and Methods.

The profile is selected explicitly with ``engine="native"`` and
``native_profile=CANONICAL_N4A_NATIVE_PROFILE``. It never falls back to the
interim Archive V2/V3 lanes or to the bundle path.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import numpy as np

from nirs4all.pipeline.dagml.core_archive_replay import CANONICAL_N4A_V1, _resolve_methods_library_identity
from nirs4all.pipeline.dagml.host_container import N4aSource, open_n4a_source
from nirs4all.pipeline.dagml.result import _scores_to_run_result

from .result import PredictResult, RunResult

CANONICAL_N4A_NATIVE_PROFILE = "nirs4all.n4a.container.v1"
"""Explicit ``native_profile`` selecting the canonical Core ``.n4a`` producer."""

CANONICAL_N4A_SCHEMA = CANONICAL_N4A_V1
"""Exact manifest ``schema`` that identifies a canonical container."""

_PREDICT_ENTRYPOINT = "predict-external"
_BRIDGE_FUNCTIONS = ("produce_n4a_native", "inspect_n4a_bytes", "predict_n4a_native")
_RETRAIN_FUNCTION = "retrain_n4a_native"
_RETRAIN_MODEL_NAME = "CanonicalN4a"


class CanonicalN4aError(RuntimeError):
    """The canonical Core ``.n4a`` profile refused or could not complete."""


class CanonicalN4aDependencyError(CanonicalN4aError):
    """The installed ``nirs4all_core`` lacks the canonical ``.n4a`` bridge."""


def _require_core_bridge(*extra: str) -> Any:
    """Import Core and require the base bridge plus any operation-specific ``extra`` functions."""

    try:
        core = importlib.import_module("nirs4all_core")
    except ImportError as error:
        raise CanonicalN4aDependencyError("the canonical .n4a profile requires a matching nirs4all-core native wheel") from error
    missing = [name for name in (*_BRIDGE_FUNCTIONS, *extra) if not callable(getattr(core, name, None))]
    if missing:
        raise CanonicalN4aDependencyError("installed nirs4all-core lacks the canonical .n4a bridge: " + ", ".join(missing))
    return core


def _check_verbose(verbose: Any) -> None:
    if isinstance(verbose, bool) or not isinstance(verbose, int) or not 0 <= verbose <= 3:
        raise ValueError("verbose must be an integer in 0..3")


def _label(prefix: str) -> str:
    """Opaque caller-chosen label; Core owns every scientific identity."""

    return f"{prefix}:nirs4all.native_n4a.{uuid.uuid4().hex}"


def _read_record(dataset: Any, source_id: Any) -> tuple[bytes, str]:
    """Return the exact IO record bytes and its source id; never re-encode."""

    if isinstance(dataset, Mapping):
        raise TypeError("the canonical .n4a profile requires dataset=<IO record path or bytes> and source_id=<id>")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("the canonical .n4a profile requires a nonempty source_id option")
    return _record_bytes(dataset, "dataset"), source_id


def _record_bytes(record: Any, label: str) -> bytes:
    if isinstance(record, (str, Path)):
        content = Path(record).read_bytes()
    elif isinstance(record, (bytes, bytearray, memoryview)):
        content = bytes(record)
    else:
        raise TypeError(f"{label} must be a path or bytes")
    if not content:
        raise ValueError(f"{label} is empty")
    return content


def _load_yaml_recipe(text: str) -> Any:
    """Parse a YAML recipe, refusing duplicate mapping keys instead of letting the last one win."""

    import yaml

    class _StrictLoader(yaml.SafeLoader):
        def construct_mapping(self, node: Any, deep: bool = False) -> dict[Any, Any]:
            if isinstance(node, yaml.MappingNode):
                seen = set()
                for key_node, _ in node.value:
                    if key_node.tag == "tag:yaml.org,2002:merge":
                        continue
                    key = self.construct_object(key_node, deep=True)
                    if key in seen:
                        raise ValueError(f"duplicate key {key!r} in the YAML recipe (line {key_node.start_mark.line + 1})")
                    seen.add(key)
            return super().construct_mapping(node, deep)

    return yaml.load(text, Loader=_StrictLoader)  # noqa: S506 - SafeLoader subclass


def _recipe_json(pipeline: Any) -> tuple[str, str]:
    """Return recipe JSON text and the final model method id for a native recipe."""

    if isinstance(pipeline, (str, Path)):
        path = Path(pipeline)
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix in {".yaml", ".yml"}:
            document = _load_yaml_recipe(text)
            text = json.dumps(document, allow_nan=False)
        elif suffix != ".json":
            raise ValueError("a canonical recipe path must be .json, .yaml or .yml")
    elif isinstance(pipeline, Mapping):
        document = dict(pipeline)
        text = json.dumps(document, allow_nan=False)
    else:
        raise TypeError("the canonical .n4a profile requires a native recipe mapping or a YAML/JSON recipe path, not a step list")
    parsed = json.loads(text)
    steps = parsed.get("steps") if isinstance(parsed, dict) else None
    if not isinstance(steps, list) or not steps or not isinstance(steps[-1], dict):
        raise ValueError("a canonical native recipe requires a nonempty 'steps' list")
    if steps[-1].get("role") == "classifier":
        raise NotImplementedError("the canonical .n4a profile projects regression scores only; a classifier final step is refused")
    return text, str(steps[-1].get("method_id", "native"))


def _owner_selection(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Read Core's single OOF SELECT decision from ``training.outcome``; never select here.

    Mirrors the owner's ``validate_selection_decision`` at the bridge boundary: exactly one
    decision bound to ``selected_variant_id``, evaluated on OOF evidence, with the metric the
    score set declares. Anything else is refused instead of being projected as false science.
    """

    def refuse(reason: str) -> CanonicalN4aError:
        return CanonicalN4aError(f"Core's training outcome has no usable owner selection: {reason}")

    selected_variant_id = outcome.get("selected_variant_id")
    output_id = outcome.get("selection_output_id")
    if not isinstance(selected_variant_id, str) or not selected_variant_id.strip():
        raise refuse("selected_variant_id is missing")
    if not isinstance(output_id, str) or not output_id.strip():
        raise refuse("selection_output_id is missing")
    outputs = outcome.get("outputs")
    bound = [
        output for output in (outputs if isinstance(outputs, list) else [])
        if isinstance(output, Mapping) and isinstance(output.get("binding"), Mapping) and output["binding"].get("binding_id") == output_id
    ]
    if len(bound) != 1:
        raise refuse("selection_output_id does not resolve exactly one output")
    bundle = outcome.get("execution_bundle")
    selections = bundle.get("selections") if isinstance(bundle, Mapping) else None
    if not isinstance(selections, Mapping) or len(selections) != 1:
        raise refuse("the execution bundle must contain exactly one SELECT decision")
    (policy_key, decision), = selections.items()
    if not isinstance(decision, Mapping) or decision.get("policy_id") != policy_key:
        raise refuse("the SELECT decision is not keyed by its policy id")
    if decision.get("selected_candidate_id") != selected_variant_id:
        raise refuse("the SELECT decision selects a different candidate than selected_variant_id")
    if decision.get("evaluation_scope") != "oof":
        raise refuse("the SELECT decision is not evaluated on OOF evidence")
    metric_name = decision.get("metric_name")
    objective = decision.get("objective")
    metric_level = decision.get("metric_level")
    score = decision.get("selected_score")
    if not isinstance(metric_name, str) or not metric_name or objective not in {"minimize", "maximize"} or not isinstance(metric_level, str) or not metric_level:
        raise refuse("the SELECT decision omits its metric, objective or level")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise refuse("the SELECT decision score is not finite")
    score_set = outcome.get("score_set")
    if not isinstance(score_set, Mapping) or score_set.get("selection_metric") != metric_name:
        raise refuse("the score set declares a different selection metric")
    from nirs4all.core.metrics import get_available_metrics, is_higher_better

    if metric_name not in get_available_metrics("regression"):
        raise refuse(f"metric {metric_name!r} is not a supported regression metric")
    if (objective == "maximize") != bool(is_higher_better(metric_name)):
        raise refuse("the SELECT objective contradicts the metric's direction")
    return {
        "selected_variant_id": selected_variant_id,
        "selection_output_id": output_id,
        "policy_id": policy_key,
        "metric_name": metric_name,
        "objective": objective,
        "metric_level": metric_level,
        "evaluation_scope": "oof",
        "selected_score": float(score),
    }


class CanonicalN4aRunResult(RunResult):
    """Score projection of one Core-sealed canonical container.

    ``export`` publishes the sealed bytes produced by the single Core fit; it
    never trains, replays or re-inspects. ``best_score`` (and the metric shortcut
    that matches the owner's selection metric) is Core's selected OOF score, never a
    retained-test measurement; other shortcuts are NaN rather than a test substitute.
    The train-only producer refuses non-train records, so it supplies no retained-test
    scores (``best["test_score"]`` and ``final_score`` are ``None``).
    """

    def __init__(
        self,
        projected: RunResult,
        *,
        archive_bytes: bytes,
        view: Mapping[str, Any],
        methods_library_path: str,
        selection: Mapping[str, Any],
    ) -> None:
        super().__init__(
            predictions=projected.predictions,
            per_dataset={
                dataset_name: {**info, "engine": "native", "native_profile": CANONICAL_N4A_NATIVE_PROFILE}
                for dataset_name, info in projected.per_dataset.items()
            },
        )
        self._archive_bytes = archive_bytes
        self._canonical_view = dict(view)
        self._methods_library_path = methods_library_path
        self._selection = dict(selection)
        self._native_archive_reference = {
            "container_id": str(view["manifest"]["container_id"]),
            "archive_sha256": str(view["receipt"]["container_sha256"]),
        }

    @property
    def native_archive_reference(self) -> dict[str, str]:
        """Return the container id and SHA-256 of the sealed bytes."""

        return dict(self._native_archive_reference)

    @property
    def selection(self) -> dict[str, Any]:
        """Return Core's owner SELECT decision (variant, metric, objective, level, OOF score)."""

        return dict(self._selection)

    @property
    def best_score(self) -> float:
        """Core's selected OOF score for the owner-selected variant."""

        return float(self._selection["selected_score"])

    def _selected_metric(self, metric: str, *test_score_aliases: str) -> float:
        return float(self._selection["selected_score"]) if self._selection["metric_name"] == metric else float("nan")

    @property
    def best_mae(self) -> float | None:
        """Core's selected OOF score when the owner selected on ``mae``; otherwise ``None``."""

        return float(self._selection["selected_score"]) if self._selection["metric_name"] == "mae" else None

    @property
    def canonical_view(self) -> dict[str, Any]:
        """Return the owner view (manifest, receipt, training) from the single fit."""

        view: dict[str, Any] = json.loads(json.dumps(self._canonical_view))
        return view

    def export(
        self,
        output_path: str | Path,
        format: str = "n4a",
        source: dict[str, Any] | None = None,
        chain_id: str | None = None,
        *,
        compatibility: str | None = None,
    ) -> Path:
        """Atomically write the already sealed canonical bytes."""

        if format != "n4a":
            raise ValueError("the canonical .n4a export supports only format='n4a'")
        if source is not None or chain_id is not None or compatibility is not None:
            raise NotImplementedError("the canonical .n4a export rejects bundle selectors and compatibility refits")
        path = Path(output_path)
        if path.suffix.lower() != ".n4a":
            raise ValueError("the canonical .n4a export requires a .n4a path")
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, staged = tempfile.mkstemp(prefix=".n4a-publish-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(self._archive_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(staged, path)
        except BaseException:
            Path(staged).unlink(missing_ok=True)
            raise
        return path

    def close(self) -> None:
        """No native handle is retained; the sealed bytes stay immutable."""


def run_native_n4a(
    pipeline: Any,
    dataset: Any,
    *,
    name: str = "",
    verbose: int = 1,
    save_artifacts: bool = True,
    save_charts: bool = False,
    plots_visible: bool = False,
    random_state: int | None = None,
    refit: Any = True,
    cache: Any = None,
    project: str | None = None,
    report_naming: str = "nirs",
    results_path: Any = None,
    session: Any = None,
    tuning: Any = None,
    calibration: Any = None,
    terminal_predict: Any = None,
    runner_kwargs: Mapping[str, Any] | None = None,
) -> CanonicalN4aRunResult:
    """Produce one canonical container through Core's single fit.

    Every option Core's producer cannot honor is refused explicitly instead of
    being ignored: seeds, cross-validation/tuning controls, calibration, sessions
    and workspaces.
    """

    if session is not None:
        raise NotImplementedError("the canonical .n4a profile does not support sessions")
    if tuning is not None or calibration is not None or terminal_predict is not None:
        raise NotImplementedError("the canonical .n4a profile does not support run-level tuning, calibration or terminal_predict")
    if random_state is not None:
        raise NotImplementedError("Core's canonical producer exposes no seed; random_state is refused rather than ignored")
    if refit is not True:
        raise NotImplementedError("the canonical .n4a profile always includes Core's single refit")
    if cache is not None or project is not None or results_path is not None:
        raise NotImplementedError("the canonical .n4a profile does not use cache, project or results_path")
    if save_charts or plots_visible:
        raise NotImplementedError("the canonical .n4a profile does not expose chart controls")
    if not save_artifacts:
        raise ValueError("the canonical .n4a profile requires save_artifacts=True")
    if report_naming != "nirs":
        raise NotImplementedError("the canonical .n4a profile supports report_naming='nirs' only")
    _check_verbose(verbose)
    options = dict(runner_kwargs or {})
    methods_library_override = options.pop("methods_library_path", None)
    source_id = options.pop("source_id", None)
    if "workspace_path" in options:
        raise NotImplementedError(
            "the canonical .n4a profile writes no workspace; workspace_path is refused rather than ignored"
        )
    if options:
        raise NotImplementedError(f"the canonical .n4a profile rejects runner options: {sorted(options)}")

    record, source_id = _read_record(dataset, source_id)
    recipe_json, model_name = _recipe_json(pipeline)
    core = _require_core_bridge()
    library_path, library_sha256 = _resolve_methods_library_identity(methods_library_override)

    from nirs4all.core.logging import configure_logging

    configure_logging(verbose=verbose)
    produced = core.produce_n4a_native(
        record,
        source_id,
        recipe_json,
        methods_library_path=library_path,
        methods_library_sha256=library_sha256,
        run_id=_label("run"),
        container_id=_label("container"),
        embed_training_record=True,
    )
    return _canonical_result(produced, name, model_name, library_path)


def retrain_canonical_n4a(
    source: Any,
    data: Any,
    *,
    name: str = "retrain_dataset",
    new_model: Any = None,
    epochs: int | None = None,
    verbose: int = 1,
    save_artifacts: bool = True,
    options: Mapping[str, Any] | None = None,
    opened_source: N4aSource | None = None,
) -> CanonicalN4aRunResult:
    """Retrain a canonical container on a new IO record through Core's single fit.

    ``source`` is a canonical ``.n4a`` path or the :class:`CanonicalN4aRunResult` it was
    produced as; ``data`` is the new exact IO record (path or bytes, or ``{"record": ...}``).
    Core reads the validated stored recipe and source id and runs fresh cross-validation,
    selection and refit through the same producer: this is always a full retrain, never a
    transfer or fine-tune, and the caller supplies no recipe or source id. The source
    container and result are left unchanged; the returned result owns new sealed bytes.

    Every option Core cannot honor is refused before any fit. The only supported option
    is ``methods_library_path``.
    """

    if new_model is not None or epochs is not None:
        raise NotImplementedError("canonical .n4a retrain is a fresh full retrain: new_model and epochs are refused")
    if not save_artifacts:
        raise ValueError("the canonical .n4a profile requires save_artifacts=True")
    _check_verbose(verbose)
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    pending = dict(options or {})
    methods_library_override = pending.pop("methods_library_path", None)
    if pending:
        raise NotImplementedError(
            f"canonical .n4a retrain supports only the methods_library_path option; refused: {sorted(pending)}"
        )
    if isinstance(data, Mapping):
        if set(data) != {"record"}:
            raise TypeError("canonical .n4a retrain requires data=<IO record path or bytes> or data={'record': <IO record path or bytes>}")
        data = data["record"]
    record = _record_bytes(data, "data")
    if not isinstance(source, (CanonicalN4aRunResult, str, Path)):
        raise TypeError("canonical .n4a retrain requires a canonical archive path or a CanonicalN4aRunResult")

    core = _require_core_bridge(_RETRAIN_FUNCTION)
    library_path, library_sha256 = _resolve_methods_library_identity(methods_library_override)
    with open_n4a_source(Path(source)) if isinstance(source, (str, Path)) and opened_source is None else nullcontext(opened_source) as opened:
        if not isinstance(source, CanonicalN4aRunResult):
            # A path source either opens here or borrows the supplied source; only a result can have no open.
            active_source = cast(N4aSource, opened)
            if active_source.path.resolve() != Path(source).resolve() or active_source.kind != "canonical" or active_source.container is None:
                raise ValueError("canonical retrain requires its matching Core V1 source")
            archive_input = active_source.container.core_file
        else:
            archive_input = source._archive_bytes

        from nirs4all.core.logging import configure_logging

        configure_logging(verbose=verbose)
        produced = core.retrain_n4a_native(
            archive_input,
            record=record,
            methods_library_path=library_path,
            methods_library_sha256=library_sha256,
            run_id=_label("run"),
            container_id=_label("container"),
            embed_training_record=True,
        )
    return _canonical_result(produced, name, _RETRAIN_MODEL_NAME, library_path)


def _canonical_result(produced: Any, name: str, model_name: str, library_path: str) -> CanonicalN4aRunResult:
    """Validate Core's sealed production and project its owner scores; shared by produce and retrain."""

    archive_bytes = produced["archive_bytes"]
    view = produced["view"]
    if not isinstance(archive_bytes, bytes) or not isinstance(view, Mapping):
        raise CanonicalN4aError("Core's canonical producer returned an invalid result")
    receipt = view.get("receipt")
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("container_sha256") != hashlib.sha256(archive_bytes).hexdigest()
        or receipt.get("byte_length") != len(archive_bytes)
    ):
        raise CanonicalN4aError("Core's canonical receipt does not match the sealed bytes")
    outcome = (view.get("training") or {}).get("outcome")
    score_set = outcome.get("score_set") if isinstance(outcome, Mapping) else None
    if not isinstance(outcome, Mapping) or not isinstance(score_set, Mapping) or not isinstance(view.get("manifest"), Mapping):
        raise CanonicalN4aError("Core's canonical view omitted the training outcome scores or manifest")
    selection = _owner_selection(outcome)
    try:
        projected = _scores_to_run_result(
            dict(score_set), name or "native", model_name, selection["metric_name"], "regression",
            selected_variant_id=selection["selected_variant_id"],
        )
    except ValueError as error:
        raise CanonicalN4aError(f"Core's score set cannot be bound to its owner selection: {error}") from error
    return CanonicalN4aRunResult(
        projected, archive_bytes=archive_bytes, view=view, methods_library_path=library_path, selection=selection
    )


def inspect_n4a(
    path: str | Path, *, source: N4aSource | None = None, include_host_json: bool = False,
) -> dict[str, Any]:
    """Inspect current container metadata without loading fitted Python objects.

    Defaults preserve the existing Core view. ``source`` borrows an SDK N4aSource;
    use ``open_n4a_source(core_file=...)`` to adapt a retained Core file. Borrowed
    sources remain open. Host JSON mode returns exact owner documents and opaque
    fitted-artifact descriptors, never models. JSON inspection and the existing
    bundle pre-model checks do not establish scientific scoring or executability.
    """
    if source is not None and not isinstance(source, N4aSource):
        raise TypeError("source must be an SDK N4aSource or None")
    if not isinstance(include_host_json, bool):
        raise TypeError("include_host_json must be a bool")
    candidate = Path(path)
    with open_n4a_source(candidate) if source is None else nullcontext(source) as opened:
        if opened.path.resolve() != candidate.resolve():
            raise ValueError("inspection source does not match its path context")
        if opened.container is None:
            raise ValueError("inspection requires a Core V1 .n4a container; re-export with current nirs4all")
        if opened.kind == "host":
            return _inspect_host_json(candidate, opened) if include_host_json else opened.container.view()
        if include_host_json:
            raise ValueError("host JSON inspection requires an SDK host container")
        core = _require_core_bridge()
        view = core.inspect_n4a_bytes(opened.container.core_file)
        if not isinstance(view, dict):
            raise CanonicalN4aError("Core's canonical inspection returned an invalid view")
        return view


def _inspect_host_json(path: Path, source: N4aSource) -> dict[str, Any]:
    """SDK-owned JSON selection; no artifact-provider access or fitted extraction."""
    from nirs4all.pipeline.dagml.host_container import (
        BUNDLE_CONTENT_TYPE,
        BUNDLE_MANIFEST_MEMBER,
        HOST_OPERATION,
        HOST_OWNER,
        JSON_CONTENT_TYPE,
    )

    container = source.require_host()
    roles = {
        BUNDLE_MANIFEST_MEMBER: "bundle_manifest",
        "pipeline.json": "prediction_config",
        "train_pipeline.json": "training_declaration",
        "chain.json": "chain",
        "trace.json": "trace",
        "fold_weights.json": "fold_weights",
        "dagml_source_training_capture.json": "source_training_capture",
    }
    bundle_bytes = None
    bundle_json = None
    if BUNDLE_MANIFEST_MEMBER in container.namelist():
        if container.artifact(BUNDLE_MANIFEST_MEMBER)["content_type"] != {"id": BUNDLE_CONTENT_TYPE, "version": 1}:
            raise ValueError("host bundle manifest has the wrong declared content type")
        bundle_bytes = container.read(BUNDLE_MANIFEST_MEMBER)
        bundle_json = json.loads(bundle_bytes)
        if not isinstance(bundle_json, dict):
            raise ValueError("host bundle manifest must be a JSON object")
        relation = bundle_json.get("relation_replay_manifest")
        if relation:
            if not isinstance(relation, dict) or not isinstance(relation.get("path", "relation_replay_manifest.json"), str):
                raise ValueError("invalid relation replay manifest reference")
            relation_path = relation.get("path", "relation_replay_manifest.json")
            if relation_path in roles:
                raise ValueError("relation replay manifest conflicts with another SDK document")
            roles[relation_path] = "relation_replay_manifest"
            if relation_path not in container.namelist():
                raise ValueError("relation replay manifest reference is missing")
    documents = []
    opaque = []
    for member in container.namelist():
        artifact = container.artifact(member)
        content_type = artifact["content_type"]
        descriptor = {
            "member": member, "artifact_id": artifact["id"], "content_type": content_type,
            "inventory_sha256": container.digest(member).removeprefix("sha256:"),
            "byte_length": container.size(member),
        }
        if content_type["id"] in (JSON_CONTENT_TYPE, BUNDLE_CONTENT_TYPE) and content_type["version"] != 1:
            raise ValueError(f"unsupported declared SDK JSON content version: {member}")
        role = roles.get(member, "host_json")
        expected = BUNDLE_CONTENT_TYPE if role == "bundle_manifest" else JSON_CONTENT_TYPE
        if member in roles and content_type != {"id": expected, "version": 1}:
            raise ValueError(f"SDK JSON document has the wrong declared content type: {member}")
        if content_type not in ({"id": JSON_CONTENT_TYPE, "version": 1}, {"id": BUNDLE_CONTENT_TYPE, "version": 1}):
            opaque.append(descriptor)
            continue
        raw = bundle_bytes if role == "bundle_manifest" and bundle_bytes is not None else container.read(member)
        value = bundle_json if role == "bundle_manifest" else json.loads(raw)
        if role != "host_json" and not isinstance(value, dict):
            raise ValueError(f"SDK {role} must be a JSON object: {member}")
        documents.append({**descriptor, "role": role, "bytes": raw, "json": value})
    has_bundle = any(item["content_type"] == {"id": BUNDLE_CONTENT_TYPE, "version": 1} for item in container.manifest["artifacts"])
    if has_bundle:
        from nirs4all.pipeline.bundle import BundleLoader

        # Its constructor runs only existing pre-model validation. Do not inspect
        # the provider, resolve prediction objects, load artifacts or execute it.
        with BundleLoader(path, source=source):
            pass
    return {
        "profile": HOST_OWNER, "operation": HOST_OPERATION if bundle_bytes is not None else None,
        "view": container.view(), "bundle_manifest_bytes": bundle_bytes,
        "documents": documents, "opaque_artifacts": opaque,
        "validation": {
            "bundle_loader_pre_model": has_bundle,
            "bundle_format": "checked" if has_bundle else "not_applicable",
            "independent_output_topology": "checked" if bundle_json is not None and bundle_json.get("dagml_native_export_shape") == "independent_by_source_multi" else "not_applicable",
            "source_training_capture": (
                "checked" if has_bundle else "not_evaluated"
            ) if any(item["role"] == "source_training_capture" for item in documents) else "not_present",
            "scientific_scoring": "not_evaluated", "fitted_objects": "not_loaded",
        },
    }


def _id_list(value: Any, label: str) -> list[str]:
    """A nonempty list of nonempty, unique string ids as supplied by Core."""

    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise CanonicalN4aError(f"Core's {label} sample ids are missing or not nonempty strings")
    if len(set(value)) != len(value):
        raise CanonicalN4aError(f"Core's {label} sample ids contain duplicates")
    return value


def predict_canonical_n4a(
    path: str | Path,
    data: Any,
    *,
    methods_library_path: str | Path | None = None,
    verbose: int = 0,
    source: N4aSource | None = None,
) -> PredictResult:
    """Replay a canonical container on one external IO record through Core.

    ``data`` is ``{"record": <IO record path or bytes>, "partition": <partition>}``.
    Core validates the container at this boundary and every refusal propagates;
    nothing is fitted and no other route is tried.

    Core's replay block is ordered by sample id, not by record row. Core also returns
    the physical source/partition order as ``sample_ids``; ``y_pred`` and
    ``metadata["sample_ids"]`` follow that order (the signed ``replay`` and ``receipt``
    are returned untouched). The SDK never parses the record: a missing, duplicated or
    differently-populated id list is refused.
    """

    _check_verbose(verbose)
    if not isinstance(data, Mapping) or set(data) != {"record", "partition"}:
        raise TypeError("canonical .n4a prediction requires data={'record': <IO record path or bytes>, 'partition': <partition>}")
    partition = data["partition"]
    if not isinstance(partition, str) or not partition.strip():
        raise ValueError("data['partition'] must be a nonempty string")
    record = _record_bytes(data["record"], "data['record']")

    core = _require_core_bridge()
    from nirs4all.core.logging import configure_logging

    configure_logging(verbose=verbose)
    with open_n4a_source(Path(path)) if source is None else nullcontext(source) as opened:
        if opened.path.resolve() != Path(path).resolve() or opened.kind != "canonical" or opened.container is None:
            raise ValueError("canonical prediction requires its matching Core V1 source")
        readability = opened.container.view()["readability"]
        training_view = readability["training_view"]
        if training_view.get("status") != "readable":
            raise CanonicalN4aError("Core's canonical training view is unavailable")
        archive_sha256 = readability["container_sha256"]
        library_path, library_sha256 = _resolve_methods_library_identity(methods_library_path)
        result = core.predict_n4a_native(
            opened.container.core_file,
            record=record,
            entrypoint=_PREDICT_ENTRYPOINT,
            source_id=training_view["source_id"],
            partition=partition,
            request_id=_label("request"),
            run_id=_label("run"),
            methods_library_path=library_path,
            methods_library_sha256=library_sha256,
            require_training_schema=True,
        )
    replay = result["replay"]
    outputs = replay.get("outputs") if isinstance(replay, Mapping) else None
    blocks = outputs[0].get("predictions") if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], Mapping) else None
    if not isinstance(blocks, list) or len(blocks) != 1 or not isinstance(blocks[0], Mapping):
        raise CanonicalN4aError("Core's canonical replay did not return exactly one prediction block")
    values = np.asarray(blocks[0].get("values"), dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise CanonicalN4aError("Core's canonical replay did not return a finite prediction matrix")
    block_ids = _id_list(blocks[0].get("sample_ids"), "replay block")
    physical_ids = _id_list(result.get("sample_ids"), "owner physical-order")
    if len(block_ids) != values.shape[0]:
        raise CanonicalN4aError("Core's replay block ids do not align with its prediction rows")
    if len(physical_ids) != len(block_ids) or set(physical_ids) != set(block_ids):
        raise CanonicalN4aError("Core's physical-order sample ids differ from the replay block ids (missing or extra ids)")
    position = {sample_id: row for row, sample_id in enumerate(block_ids)}
    values = values[[position[sample_id] for sample_id in physical_ids]]
    return PredictResult(
        y_pred=values,
        sample_ids=list(physical_ids),
        metadata={
            "engine": "native",
            "native_profile": CANONICAL_N4A_NATIVE_PROFILE,
            "archive_path": str(path),
            "archive_sha256": archive_sha256,
            "sample_ids": list(physical_ids),
            "training_performed": False,
            "replay": replay,
            "receipt": result["receipt"],
        },
        model_name="CanonicalN4a",
        preprocessing_steps=[],
    )
