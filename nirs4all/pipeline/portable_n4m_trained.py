"""Read and replay a cross-language trained n4m pipeline without host pickles.

The envelope carries a shared JSON recipe, fitted native preprocessing state,
and one native N4MM model. Numerical transforms and prediction remain in Methods.
This is a bounded native pipeline format, not a general DAG-ML archive.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

_STATELESS = {
    "n4m.SNV", "n4m.SavitzkyGolay", "n4m.LSNV", "n4m.RNV",
    "n4m.AreaNormalization", "n4m.Detrend",
}
_STATEFUL = {"n4m.MSC", "n4m.EMSC"}
_SELECTOR = "n4m.SPA"
_GENERIC_SELECTOR = "n4m.Selector"
_AFFINE_MODEL_PARAMS: dict[str, frozenset[str]] = {
    "n4m.Ridge": frozenset({"alpha"}),
    "n4m.RidgePLS": frozenset({"ridge_lambda"}),
    "n4m.RobustPLS": frozenset({"huber_k", "max_irls_iter"}),
    "n4m.CPPLS": frozenset({"gamma"}),
    "n4m.SparseSIMPLS": frozenset({"sparsity_lambda"}),
    "n4m.ECR": frozenset({"alpha"}),
    "n4m.ContinuumRegression": frozenset({"tau"}),
    "n4m.MIRPLS": frozenset(),
    "n4m.FusedSparsePLS": frozenset({"l1_lambda", "fusion_lambda"}),
    "n4m.BaggingPLS": frozenset({"n_estimators", "seed"}),
    "n4m.BoostingPLS": frozenset({"n_estimators", "learning_rate"}),
    "n4m.RandomSubspacePLS": frozenset({"n_estimators", "features_per_subspace", "seed"}),
    "n4m.NPLS": frozenset({"mode_j", "mode_k"}),
    "n4m.MBPLS": frozenset({"block_sizes"}),
}


class PortableN4MTrainedPipeline:
    """An imported native PLS or sparse PLS-DA predictor and retraining recipe.

    Use :meth:`close` or a context manager to release the N4MM handle. The
    original JSON recipe remains available for training a fresh model through
    the normal Python ``PipelineConfigs``/``StepParser`` path. Version 3 adds
    supervised SPA state to regression PLS recipes; its wire indices are zero
    based and ranked, while prediction projects columns in spectral order.
    Version 4 applies the same state contract to generic native selectors.
    Version 5 carries a predict-only affine N4MM and a declarative n4m
    regression recipe for fresh fitting; the payload does not attest which
    algorithm originally fitted its coefficients.
    """

    def __init__(self, document: dict[str, Any]) -> None:
        from pls4all import Context, Model, inspect_n4mm

        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        expected = {"schema", "manifest_json", "manifest_sha256", "model"}
        if (not isinstance(document, dict) or set(document) != expected
                or document["schema"] not in {
                    "nirs4all.n4m.trained_pipeline.v1",
                    "nirs4all.n4m.trained_pipeline.v2",
                    "nirs4all.n4m.trained_pipeline.v3",
                    "nirs4all.n4m.trained_pipeline.v4",
                    "nirs4all.n4m.trained_pipeline.v5",
                }):
            raise ValueError("unsupported trained n4m pipeline envelope")
        classification = document["schema"].endswith(".v2")
        selector_envelope = document["schema"].endswith(".v3")
        generic_envelope = document["schema"].endswith(".v4")
        affine_envelope = document["schema"].endswith(".v5")
        manifest_json = document["manifest_json"]
        manifest_hash = document["manifest_sha256"]
        if (not isinstance(manifest_json, str) or not isinstance(manifest_hash, str)
                or hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != manifest_hash):
            raise ValueError("trained pipeline manifest hash mismatch")
        manifest = json.loads(manifest_json)
        manifest_fields = {"recipe", "input_n_features", "feature_names",
                           "preprocessing_owner", "step_states"}
        if classification:
            manifest_fields |= {"task", "classes"}
        if affine_envelope:
            manifest_fields.add("fit_recipe_assertion")
        if not isinstance(manifest, dict) or set(manifest) != manifest_fields:
            raise ValueError("invalid trained pipeline manifest")
        classes: list[str] | None = None
        if classification:
            classes = manifest["classes"]
            if (manifest["task"] != "classification" or not isinstance(classes, list)
                    or len(classes) < 2 or any(not isinstance(value, str) or not value for value in classes)
                    or len(set(classes)) != len(classes)):
                raise ValueError("invalid ordered classification classes")
        width = manifest["input_n_features"]
        if type(width) is not int or width < 2:
            raise ValueError("invalid input feature width")
        names = manifest["feature_names"]
        if names is not None and (not isinstance(names, list) or len(names) != width
                                  or any(not isinstance(name, str) or not name for name in names)
                                  or len(set(names)) != width):
            raise ValueError("invalid ordered feature names")
        recipe = manifest["recipe"]
        if not isinstance(recipe, dict) or not isinstance(recipe.get("pipeline"), list):
            raise ValueError("invalid n4m recipe")
        nodes = recipe["pipeline"]
        if not nodes or not isinstance(nodes[-1], dict):
            raise ValueError("trained n4m pipeline needs one final model")
        has_selector = self._has_selector(nodes[:-1], _SELECTOR)
        has_generic_selector = self._has_selector(nodes[:-1], _GENERIC_SELECTOR)
        # Apply the same config resolver used by ordinary Python nirs4all.
        PipelineConfigs(recipe)
        if set(nodes[-1]) != {"model"}:
            raise ValueError("trained n4m pipeline needs one final model")
        model_node = nodes[-1]["model"]
        self._validate_model_node(model_node, classification, affine_envelope)
        if affine_envelope:
            assertion = manifest["fit_recipe_assertion"]
            expected_assertion = {
                "kind": "affine_recipe", "recipe_class": model_node["class"],
            }
            if model_node["class"] == "n4m.MBPLS":
                expected_assertion["block_sizes"] = model_node["params"]["block_sizes"]
            if (not isinstance(assertion, dict)
                    or assertion != expected_assertion):
                raise ValueError("affine fit recipe assertion differs from recipe")
        owner = manifest["preprocessing_owner"]
        if owner not in {"external", "embedded_methods"}:
            raise ValueError("unsupported preprocessing owner")
        if classification and owner != "external":
            raise ValueError("sparse PLS-DA requires external preprocessing")
        if selector_envelope and owner != "external":
            raise ValueError("trained SPA requires external preprocessing")
        if generic_envelope and owner != "external":
            raise ValueError("trained generic selector requires external preprocessing")
        if affine_envelope and owner != "external":
            raise ValueError("trained affine predictor requires external preprocessing")
        if not affine_envelope and selector_envelope != has_selector:
            raise ValueError("trained selector state requires v3 envelope")
        if not affine_envelope and generic_envelope != has_generic_selector:
            raise ValueError("trained generic selector state requires v4 envelope")
        if has_generic_selector and (has_selector or classification):
            raise ValueError("v4 generic selector cannot mix SPA or classification")
        states = manifest["step_states"]
        if not isinstance(states, list):
            raise ValueError("invalid fitted preprocessing state")
        if owner == "embedded_methods":
            if len(nodes) != 3 or len(states) != 2 or any(state is not None for state in states):
                raise ValueError("invalid embedded preprocessing state")
            self._validate_embedded_recipe(nodes[:2])
            output_width = width
        else:
            output_width = self._validate_steps(
                nodes[:-1], states, width,
                allow_selector=selector_envelope or affine_envelope,
                allow_generic_selector=generic_envelope or affine_envelope,
            )
        if affine_envelope and model_node["class"] == "n4m.NPLS":
            params = model_node["params"]
            if params["mode_j"] * params["mode_k"] != output_width:
                raise ValueError("NPLS tensor modes differ from fitted preprocessing width")
        if affine_envelope and model_node["class"] == "n4m.MBPLS":
            if sum(model_node["params"]["block_sizes"]) != output_width:
                raise ValueError("MBPLS block sizes differ from fitted preprocessing width")
        model = document["model"]
        if (not isinstance(model, dict) or set(model) != {"kind", "encoding", "sha256", "payload"}
                or model["kind"] != "n4m_model" or model["encoding"] != "base64-n4mm"
                or not isinstance(model["sha256"], str) or len(model["sha256"]) != 64
                or not isinstance(model["payload"], str)):
            raise ValueError("invalid N4MM model metadata")
        try:
            payload = base64.b64decode(model["payload"], validate=True)
        except binascii.Error as error:
            raise ValueError("invalid N4MM base64 payload") from error
        if not payload or hashlib.sha256(payload).hexdigest() != model["sha256"]:
            raise ValueError("N4MM payload hash mismatch")
        info = inspect_n4mm(payload)
        if affine_envelope:
            if (info.format_version != 1 or info.algorithm != 11 or info.solver != 0
                    or info.deflation != 0 or info.n_targets != 1
                    or info.n_components != 0 or info.n_features != output_width
                    or info.training_samples < 1 or info.capabilities != 5):
                raise ValueError("N4MM affine descriptor does not match recipe")
        elif classification:
            assert classes is not None
            if (info.format_version != 1 or info.algorithm != 11 or info.solver != 0
                    or info.deflation != 0 or info.n_targets != len(classes)
                    or info.n_components != 0 or info.n_features != output_width
                    or info.capabilities & 5 != 5):
                raise ValueError("N4MM sparse PLS-DA descriptor does not match recipe")
        elif (info.format_version != (2 if owner == "embedded_methods" else 1)
              or info.algorithm != 0 or info.solver != 1 or info.deflation != 0
              or info.n_targets != 1 or info.n_components != model_node["params"]["n_components"]
              or info.n_features != output_width
              or info.capabilities & (9 if owner == "embedded_methods" else 1)
                 != (9 if owner == "embedded_methods" else 1)):
            raise ValueError("N4MM descriptor does not match recipe")
        if owner == "embedded_methods":
            self._validate_embedded_info(info.pipeline, nodes[:2], width)
        elif info.pipeline is not None:
            raise ValueError("external preprocessing cannot use embedded N4MM state")
        context = Context()
        try:
            native_model = Model.from_bytes(context, payload)
        except BaseException:
            context.close()
            raise
        self.recipe = recipe
        self.feature_names = names
        self.input_n_features = width
        self.preprocessing_owner = owner
        self.task = "classification" if classification else "regression"
        self.classes = classes
        self._nodes = nodes[:-1]
        self._states = states
        self._context = context
        self._model = native_model
        self._document = json.loads(json.dumps(document, allow_nan=False))

    @staticmethod
    def _validate_model_node(model_node: Any, classification: bool,
                             affine: bool = False) -> None:
        if affine:
            if (not isinstance(model_node, dict)
                    or set(model_node) not in ({"class"}, {"class", "params"})
                    or model_node.get("class") not in _AFFINE_MODEL_PARAMS):
                raise ValueError("trained affine pipeline needs a qualified n4m model")
            name = model_node["class"]
            params = model_node.get("params", {})
            allowed = _AFFINE_MODEL_PARAMS[name]
            if (not isinstance(params, dict)
                    or not set(params) <= allowed | {"n_components"}
                    or (name == "n4m.Ridge" and "n_components" in params)
                    or (name != "n4m.Ridge" and (
                        type(params.get("n_components")) is not int
                        or not 1 <= params["n_components"] <= 2**31 - 1))):
                raise ValueError("invalid trained affine recipe parameters")
            if name == "n4m.NPLS" and (
                type(params.get("mode_j")) is not int
                or type(params.get("mode_k")) is not int
                or not 1 <= params["mode_j"] <= 2**31 - 1
                or not 1 <= params["mode_k"] <= 2**31 - 1
            ):
                raise ValueError("NPLS tensor modes must be positive bounded integers")
            if name == "n4m.MBPLS" and (
                not isinstance(params.get("block_sizes"), list)
                or len(params["block_sizes"]) < 2
                or any(type(size) is not int or not 1 <= size <= 2**31 - 1
                       for size in params["block_sizes"])
            ):
                raise ValueError("MBPLS block sizes must be positive bounded integers")
            for key, value in params.items():
                if key in {"n_components", "block_sizes"}:
                    continue
                if type(value) not in (int, float) or not np.isfinite(value):
                    raise ValueError("affine recipe parameters must be finite numbers")
                if key in {"max_irls_iter", "n_estimators", "features_per_subspace", "seed"}:
                    minimum = 0 if key == "seed" else 1
                    if type(value) is not int or not minimum <= value <= 2**31 - 1:
                        raise ValueError(f"{key} must be a bounded integer")
                elif key == "learning_rate" and not 0 < value <= 1:
                    raise ValueError("learning_rate must be in (0, 1]")
                elif key in {"alpha", "ridge_lambda", "sparsity_lambda",
                             "l1_lambda", "fusion_lambda"} and value < 0:
                    raise ValueError(f"{key} must be non-negative")
            return
        expected_class = "n4m.SparsePLSDA" if classification else "n4m.PLS"
        expected_params = {"n_components", "sparsity_lambda"} if classification else {"n_components"}
        if (not isinstance(model_node, dict) or model_node.get("class") != expected_class
                or set(model_node) != {"class", "params"}
                or not isinstance(model_node["params"], dict)
                or set(model_node["params"]) != expected_params
                or type(model_node["params"]["n_components"]) is not int
                or model_node["params"]["n_components"] < 1):
            raise ValueError(f"trained n4m pipeline needs fixed {expected_class}")
        if classification:
            sparsity = model_node["params"]["sparsity_lambda"]
            if (type(sparsity) not in (int, float) or not np.isfinite(sparsity)
                    or sparsity < 0):
                raise ValueError("invalid sparse PLS-DA regularization")

    @staticmethod
    def _validate_embedded_recipe(nodes: list[dict[str, Any]]) -> None:
        if (nodes[0] != {"class": "n4m.SNV"}
                or not isinstance(nodes[1], dict)
                or nodes[1].get("class") != "n4m.SavitzkyGolay"
                or not isinstance(nodes[1].get("params"), dict)):
            raise ValueError("unsupported embedded Methods recipe")
        params = nodes[1]["params"]
        if (set(params) != {"window_length", "polyorder", "deriv", "delta", "mode", "cval"}
                or params["deriv"] != 0 or params["delta"] != 1
                or params["mode"] != "interp" or params["cval"] != 0):
            raise ValueError("unsupported embedded Savitzky-Golay settings")

    @staticmethod
    def _validate_embedded_info(info: Any, nodes: list[dict[str, Any]], width: int) -> None:
        params = nodes[1]["params"]
        if (info is None or tuple(info.operators) != (4, 8)
                or info.semantic_profile != 1 or info.raw_n_features != width
                or info.model_n_features != width
                or info.savgol_window != params.get("window_length")
                or info.savgol_poly_degree != params.get("polyorder")
                or info.savgol_derivative != params.get("deriv")
                or info.savgol_delta != params.get("delta")
                or info.savgol_cval != params.get("cval")
                or info.savgol_mode != 4 or info.snv_axis != 1
                or info.snv_ddof != 0 or not info.snv_with_mean or not info.snv_with_std):
            raise ValueError("embedded N4MM preprocessing does not match recipe")

    @classmethod
    def _has_selector(cls, nodes: list[dict[str, Any]], selector: str = _SELECTOR) -> bool:
        for node in nodes:
            if not isinstance(node, dict):
                raise ValueError("invalid preprocessing node")
            if node.get("class") == selector:
                return True
            if "branch" in node:
                branches = node["branch"]
                if not isinstance(branches, dict):
                    raise ValueError("invalid feature branch")
                if any(cls._has_selector(branch, selector) for branch in branches.values()):
                    return True
        return False

    @staticmethod
    def _validate_selector_state(node: dict[str, Any], state: Any, width: int) -> list[int]:
        params = node.get("params")
        if (set(node) != {"class", "params"} or not isinstance(params, dict)
                or set(params) != {"top_k", "n_components"}
                or type(params["top_k"]) is not int or not 1 <= params["top_k"] <= width
                or type(params["n_components"]) is not int or params["n_components"] < 1
                or not isinstance(state, dict) or set(state) != {"kind", "selected_indices"}
                or state["kind"] != "selector"):
            raise ValueError("invalid fitted SPA state or recipe")
        selected = state["selected_indices"]
        if (not isinstance(selected, list) or len(selected) != params["top_k"]
                or any(type(index) is not int or index < 0 or index >= width for index in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("invalid fitted SPA selected_indices")
        return selected

    @staticmethod
    def _validate_generic_selector_state(node: dict[str, Any], state: Any, width: int) -> list[int]:
        from n4m.feature_selection import Selector

        params = node.get("params")
        if (set(node) != {"class", "params"} or not isinstance(params, dict)
                or set(params) != {"method", "n_components", "method_params"}
                or not isinstance(params["method_params"], dict)
                or not isinstance(state, dict) or set(state) != {"kind", "selected_indices"}
                or state["kind"] != "selector"):
            raise ValueError("invalid fitted generic selector state or recipe")
        # The same native argument contract validates omitted defaults, required
        # seeds, method vocabulary and widths, without fitting on prediction rows.
        Selector(**params)._arguments(width)
        selected = state["selected_indices"]
        if (not isinstance(selected, list) or not 1 <= len(selected) <= width
                or any(type(index) is not int or index < 0 or index >= width for index in selected)
                or len(set(selected)) != len(selected)):
            raise ValueError("invalid fitted generic selected_indices")
        return selected

    @classmethod
    def _validate_steps(cls, nodes: list[dict[str, Any]], states: list[Any], width: int,
                        *, allow_selector: bool = False,
                        allow_generic_selector: bool = False) -> int:
        # Branch/merge occupies two recipe nodes but one fitted state.
        if len(nodes) != len(states) and not any("branch" in node for node in nodes):
            raise ValueError("fitted preprocessing state count differs from recipe")
        index = 0
        state_index = 0
        while index < len(nodes):
            node = nodes[index]
            if not isinstance(node, dict):
                raise ValueError("invalid preprocessing node")
            if "branch" in node:
                if (set(node) != {"branch"} or index + 1 >= len(nodes)
                        or nodes[index + 1] != {"merge": "features"}):
                    raise ValueError("feature branch must be followed by merge")
                branches = node["branch"]
                state = states[state_index] if state_index < len(states) else None
                if (not isinstance(branches, dict) or len(branches) < 2
                        or not isinstance(state, dict) or set(state) != {"kind", "branches"}
                        or state["kind"] != "branch" or not isinstance(state["branches"], dict)
                        or list(state["branches"]) != list(branches)):
                    raise ValueError("fitted branch state differs from recipe")
                width = sum(cls._validate_steps(branch, state["branches"][name], width,
                                                allow_selector=allow_selector,
                                                allow_generic_selector=allow_generic_selector)
                            for name, branch in branches.items())
                index += 2
            elif "merge" in node:
                raise ValueError("merge has no preceding feature branch")
            else:
                state = states[state_index] if state_index < len(states) else None
                if set(node) not in ({"class"}, {"class", "params"}):
                    raise ValueError("unsupported preprocessing node")
                name = node["class"]
                if name in _STATEFUL:
                    if (not isinstance(state, dict) or set(state) != {"kind", "reference"}
                            or state["kind"] != name.removeprefix("n4m.").upper()):
                        raise ValueError("missing fitted native reference")
                    cls._reference(state["reference"], width)
                elif name == _SELECTOR and allow_selector:
                    width = len(cls._validate_selector_state(node, state, width))
                elif name == _GENERIC_SELECTOR and allow_generic_selector:
                    width = len(cls._validate_generic_selector_state(node, state, width))
                elif name in _STATELESS:
                    if state is not None:
                        raise ValueError("stateless preprocessing has fitted state")
                else:
                    raise ValueError(f"unsupported native preprocessing: {name}")
                index += 1
            state_index += 1
        if state_index != len(states):
            raise ValueError("extra fitted preprocessing state")
        return width

    @staticmethod
    def _reference(value: Any, width: int) -> np.ndarray:
        if (not isinstance(value, list) or len(value) != width
                or any(type(number) not in (int, float) for number in value)):
            raise ValueError("invalid fitted native reference")
        reference = np.asarray(value, dtype=np.float64)
        if not np.isfinite(reference).all():
            raise ValueError("non-finite fitted native reference")
        return reference

    @classmethod
    def from_json(cls, source: str | Path) -> PortableN4MTrainedPipeline:
        """Read a JSON document or filesystem path."""

        if isinstance(source, Path):
            content = source.read_text(encoding="utf-8")
        elif source.lstrip().startswith("{"):
            content = source
        else:
            content = Path(source).read_text(encoding="utf-8")
        value = json.loads(content)
        if not isinstance(value, dict):
            raise ValueError("trained pipeline JSON must be an object")
        return cls(value)

    @classmethod
    def fit_recipe(cls, recipe: dict[str, Any], X: Any, y: Any) -> PortableN4MTrainedPipeline:
        """Train a native n4m recipe in Python and produce the portable envelope.

        Methods owns every transform, SPA selection and PLS fit. For MSC/EMSC, the portable
        reference comes from the native getter when available. The published
        1.0.21 binding lacks that getter, so only its documented column-mean
        reference is reconstructed from the same training rows as a fallback.
        """

        from pls4all import Config, Context, Model, Solver
        from pls4all.migration import export_linear_predictor_n4mm

        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        PipelineConfigs(recipe)
        nodes = recipe.get("pipeline")
        if not isinstance(nodes, list) or not nodes or not isinstance(nodes[-1], dict):
            raise ValueError("portable native recipe needs a final model")
        model_node = nodes[-1].get("model")
        classification = isinstance(model_node, dict) and model_node.get("class") == "n4m.SparsePLSDA"
        affine = isinstance(model_node, dict) and model_node.get("class") in _AFFINE_MODEL_PARAMS
        cls._validate_model_node(model_node, classification, affine)
        if classification and (cls._has_selector(nodes[:-1], _SELECTOR)
                               or cls._has_selector(nodes[:-1], _GENERIC_SELECTOR)):
            raise ValueError("trained selector classification is not in the portable envelope")
        if (cls._has_selector(nodes[:-1], _SELECTOR)
                and cls._has_selector(nodes[:-1], _GENERIC_SELECTOR)):
            raise ValueError("cannot mix SPA and generic selector in one trained envelope")
        assert isinstance(model_node, dict)
        columns = getattr(X, "columns", None)
        names = list(columns) if columns is not None else None
        values = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y) if classification else np.asarray(y, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] < 2 or targets.ndim != 1
                or values.shape[0] != targets.shape[0]
                or not np.isfinite(values).all()
                or (not classification and not np.isfinite(targets).all())):
            raise ValueError("fit requires finite aligned training rows")
        if classification and (not all(isinstance(label, str) and label for label in targets.tolist())
                               or len(set(targets.tolist())) < 2):
            raise ValueError("sparse PLS-DA needs at least two non-empty string classes")
        if names is not None and (len(names) != values.shape[1]
                                  or any(not isinstance(name, str) or not name for name in names)
                                  or len(set(names)) != len(names)):
            raise ValueError("invalid ordered feature names")
        states, transformed = cls._fit_portable_steps(nodes[:-1], values, targets)
        classes: list[str] | None = None
        if classification:
            from nirs4all.pipeline.steps.parser import StepParser

            fitted = StepParser().parse(nodes[-1]).operator
            classes = sorted(set(targets.tolist()))
            class_codes = {label: index for index, label in enumerate(classes)}
            codes = np.asarray([class_codes[label] for label in targets.tolist()], dtype=np.int64)
            fitted.fit(transformed, codes)
            coefficients = np.asarray(fitted.coef_, dtype=np.float64).T
            intercept = np.asarray(fitted.y_mean_, dtype=np.float64) - np.asarray(fitted.x_mean_, dtype=np.float64) @ coefficients
            payload = export_linear_predictor_n4mm(
                coefficients.tolist(), intercept.tolist(), source_training_samples=values.shape[0],
            )
        elif affine:
            from nirs4all.pipeline.steps.parser import StepParser

            fitted = StepParser().parse(nodes[-1]).operator.fit(transformed, targets)
            coefficients = np.asarray(fitted.coef_, dtype=np.float64).reshape(-1, 1)
            intercept = np.asarray(fitted.intercept_, dtype=np.float64).reshape(-1)
            if (coefficients.shape != (transformed.shape[1], 1)
                    or intercept.shape != (1,)
                    or not np.isfinite(coefficients).all()
                    or not np.isfinite(intercept).all()):
                raise ValueError("native affine fit returned invalid coefficients")
            probe = np.concatenate((transformed[: min(3, len(transformed))],
                                    transformed[: min(3, len(transformed))] + 0.031))
            direct = np.asarray(fitted.predict(probe), dtype=np.float64).reshape(-1)
            affine_pred = (probe @ coefficients).reshape(-1) + intercept[0]
            if (direct.shape != affine_pred.shape or not np.isfinite(direct).all()
                    or not np.allclose(direct, affine_pred, rtol=1e-10, atol=1e-10)):
                raise ValueError("native affine fit differs from its coefficients")
            payload = export_linear_predictor_n4mm(
                coefficients.tolist(), intercept.tolist(), source_training_samples=values.shape[0],
            )
        else:
            with Context() as context, Config() as config:
                config.solver = Solver.SIMPLS
                config.n_components = model_node["params"]["n_components"]
                config.center_x = True
                config.scale_x = True
                config.center_y = True
                config.scale_y = True
                with Model.fit(context, config, transformed, targets) as model:
                    payload = model.to_bytes()
        manifest = {
            "recipe": recipe,
            "input_n_features": values.shape[1],
            "feature_names": names,
            "preprocessing_owner": "external",
            "step_states": states,
        }
        if classification:
            manifest["task"] = "classification"
            manifest["classes"] = classes
        if affine:
            manifest["fit_recipe_assertion"] = {
                "kind": "affine_recipe", "recipe_class": model_node["class"],
            }
            if model_node["class"] == "n4m.MBPLS":
                manifest["fit_recipe_assertion"]["block_sizes"] = model_node["params"]["block_sizes"]
        manifest_json = json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":"))
        document = {
            "schema": ("nirs4all.n4m.trained_pipeline.v2" if classification else
                       "nirs4all.n4m.trained_pipeline.v5" if affine else
                       "nirs4all.n4m.trained_pipeline.v4" if cls._has_selector(nodes[:-1], _GENERIC_SELECTOR) else
                       "nirs4all.n4m.trained_pipeline.v3" if cls._has_selector(nodes[:-1], _SELECTOR) else
                       "nirs4all.n4m.trained_pipeline.v1"),
            "manifest_json": manifest_json,
            "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
            "model": {
                "kind": "n4m_model", "encoding": "base64-n4mm",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "payload": base64.b64encode(payload).decode("ascii"),
            },
        }
        return cls(document)

    @classmethod
    def _fit_portable_steps(
        cls, nodes: list[dict[str, Any]], X: np.ndarray, y: np.ndarray,
    ) -> tuple[list[Any], np.ndarray]:
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        states: list[Any] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if not isinstance(node, dict):
                raise ValueError("invalid preprocessing node")
            if "branch" in node:
                if (set(node) != {"branch"} or index + 1 >= len(nodes)
                        or nodes[index + 1] != {"merge": "features"}
                        or not isinstance(node["branch"], dict)
                        or len(node["branch"]) < 2):
                    raise ValueError("unsupported feature branch")
                outputs = [cls._fit_portable_steps(branch, X, y)
                           for branch in node["branch"].values()]
                states.append({"kind": "branch", "branches": {
                    name: output[0] for name, output in zip(node["branch"], outputs, strict=True)
                }})
                X = np.concatenate([output[1] for output in outputs], axis=1)
                index += 2
            else:
                if set(node) not in ({"class"}, {"class", "params"}):
                    raise ValueError("unsupported preprocessing node")
                name = node["class"]
                if name not in _STATELESS | _STATEFUL | {_SELECTOR, _GENERIC_SELECTOR}:
                    raise ValueError(f"unsupported native preprocessing: {name}")
                operator = parser.parse(node).operator
                if name in {_SELECTOR, _GENERIC_SELECTOR}:
                    operator.fit(X, y)
                    selected = np.asarray(operator.selected_indices_).tolist()
                    state = {"kind": "selector", "selected_indices": selected}
                    if name == _SELECTOR:
                        cls._validate_selector_state(node, state, X.shape[1])
                    else:
                        cls._validate_generic_selector_state(node, state, X.shape[1])
                    states.append(state)
                else:
                    operator.fit(X)
                if name in _STATEFUL:
                    reference = np.asarray(
                        operator.reference_ if hasattr(operator, "reference_")
                        else np.mean(X, axis=0, dtype=np.float64),
                        dtype=np.float64,
                    )
                    states.append({"kind": name.removeprefix("n4m.").upper(),
                                   "reference": reference.tolist()})
                elif name not in {_SELECTOR, _GENERIC_SELECTOR}:
                    states.append(None)
                X = np.asarray(operator.transform(X), dtype=np.float64)
                index += 1
        return states, X

    def to_json(self, file: str | Path | None = None) -> str:
        """Return or write the exact portable trained-pipeline envelope."""

        serialized = json.dumps(self._document, ensure_ascii=False, allow_nan=False,
                                separators=(",", ":"))
        if file is not None:
            Path(file).write_text(serialized, encoding="utf-8")
        return serialized

    def _transform_steps(self, values: np.ndarray, nodes: list[dict[str, Any]], states: list[Any]) -> np.ndarray:
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        index = 0
        state_index = 0
        while index < len(nodes):
            node = nodes[index]
            state = states[state_index]
            if "branch" in node:
                values = np.concatenate([
                    self._transform_steps(values, branch, state["branches"][name])
                    for name, branch in node["branch"].items()
                ], axis=1)
                index += 2
            else:
                if node["class"] in {_SELECTOR, _GENERIC_SELECTOR}:
                    selected = (self._validate_selector_state(node, state, values.shape[1])
                                if node["class"] == _SELECTOR else
                                self._validate_generic_selector_state(node, state, values.shape[1]))
                    values = values[:, sorted(selected)]
                    index += 1
                    state_index += 1
                    continue
                operator = parser.parse(node).operator
                if node["class"] in _STATEFUL:
                    reference = self._reference(state["reference"], values.shape[1])
                    if hasattr(operator, "restore_reference"):
                        operator.restore_reference(reference)
                    else:
                        # A one-row native fit learns exactly this reference;
                        # no validation sample participates in state fitting.
                        operator.fit(reference.reshape(1, -1))
                else:
                    operator.fit(values)
                values = np.asarray(operator.transform(values), dtype=np.float64)
                index += 1
            state_index += 1
        return values

    def predict_scores(self, X: Any) -> np.ndarray:
        """Return raw native Methods scores for a classification pipeline."""

        if self.task != "classification":
            raise ValueError("decision scores require sparse PLS-DA classification")
        return self._predict_native(X)

    def _predict_native(self, X: Any) -> np.ndarray:

        if self._model is None:
            raise RuntimeError("trained pipeline is closed")
        if self.feature_names is not None:
            columns = getattr(X, "columns", None)
            if columns is None or list(columns) != self.feature_names:
                raise ValueError("feature names or order differ from training")
        values = np.asarray(X, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.input_n_features
                or not np.isfinite(values).all()):
            raise ValueError("input must be a finite samples-by-features matrix")
        if self.preprocessing_owner == "external":
            values = self._transform_steps(values, self._nodes, self._states)
        result = np.asarray(self._model.predict(self._context, values), dtype=np.float64)
        if (result.shape != (values.shape[0], len(self.classes) if self.classes is not None else 1)
                or not np.isfinite(result).all()):
            raise ValueError("native predictor returned invalid output shape or values")
        return result

    def predict(self, X: Any) -> np.ndarray:
        """Predict regression values or ordered classification labels."""

        result = self._predict_native(X)
        if self.task == "classification":
            assert self.classes is not None
            labels: np.ndarray = np.asarray(self.classes, dtype=str)[np.argmax(result, axis=1)]
            return labels
        return result.reshape(result.shape[0])

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return uncalibrated softmax probabilities for sparse PLS-DA."""

        scores = self.predict_scores(X)
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probabilities: np.ndarray = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return probabilities

    def retrain(self, X: Any, y: Any) -> RefittedN4MPipeline:
        """Fit the shared n4m recipe afresh on new Python training rows.

        This does not reuse the imported model or MSC/EMSC references. The
        returned in-memory pipeline is a new Methods fit, not a serialized
        cross-language package.
        """

        if self.feature_names is not None:
            columns = getattr(X, "columns", None)
            if columns is None or list(columns) != self.feature_names:
                raise ValueError("feature names or order differ from recipe")
        values = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y) if self.task == "classification" else np.asarray(y, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.input_n_features
                or targets.ndim != 1 or targets.shape[0] != values.shape[0]
                or not np.isfinite(values).all()
                or (self.task == "regression" and not np.isfinite(targets).all())):
            raise ValueError("retrain requires finite aligned training rows")
        if self.task == "classification":
            assert self.classes is not None
            if (not all(isinstance(label, str) and label for label in targets.tolist())
                    or set(targets.tolist()) != set(self.classes)):
                raise ValueError("retrain classes differ from the portable recipe")
        trained_steps, transformed = RefittedN4MPipeline._fit_steps(
            self._nodes, values, targets,
        )
        from nirs4all.pipeline.steps.parser import StepParser

        model = StepParser().parse(self.recipe["pipeline"][-1]).operator
        if self.task == "classification":
            assert self.classes is not None
            class_codes = {label: index for index, label in enumerate(self.classes)}
            targets = np.asarray([class_codes[label] for label in targets.tolist()], dtype=np.int64)
        model.fit(transformed, targets)
        return RefittedN4MPipeline(
            trained_steps, model, self.input_n_features, self.feature_names,
            self.task, self.classes,
        )

    def close(self) -> None:
        """Release native model resources, idempotently."""

        model, self._model = self._model, None
        if model is not None:
            try:
                model.close()
            finally:
                self._context.close()

    def __enter__(self) -> PortableN4MTrainedPipeline:
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()


class RefittedN4MPipeline:
    """Fresh native Python fit from a loaded cross-language recipe."""

    def __init__(
        self, steps: list[Any], model: Any, width: int, feature_names: list[str] | None,
        task: str = "regression", classes: list[str] | None = None,
    ) -> None:
        self._steps = steps
        self._model = model
        self.input_n_features = width
        self.feature_names = feature_names
        self.task = task
        self.classes = classes

    @classmethod
    def _fit_steps(cls, nodes: list[dict[str, Any]], X: np.ndarray,
                   y: np.ndarray) -> tuple[list[Any], np.ndarray]:
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        fitted: list[Any] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if "branch" in node:
                branch_fits = [cls._fit_steps(branch, X, y) for branch in node["branch"].values()]
                fitted.append([entry[0] for entry in branch_fits])
                X = np.concatenate([entry[1] for entry in branch_fits], axis=1)
                index += 2
            else:
                operator = parser.parse(node).operator
                if node["class"] in {_SELECTOR, _GENERIC_SELECTOR}:
                    operator.fit(X, y)
                else:
                    operator.fit(X)
                X = np.asarray(operator.transform(X), dtype=np.float64)
                fitted.append(operator)
                index += 1
        return fitted, X

    @classmethod
    def _transform_steps(cls, fitted: list[Any], X: np.ndarray) -> np.ndarray:
        for operator in fitted:
            if isinstance(operator, list):
                X = np.concatenate([cls._transform_steps(branch, X) for branch in operator], axis=1)
            else:
                X = np.asarray(operator.transform(X), dtype=np.float64)
        return X

    def predict(self, X: Any) -> np.ndarray:
        """Predict with the freshly fitted Methods operators and model."""

        if self.feature_names is not None:
            columns = getattr(X, "columns", None)
            if columns is None or list(columns) != self.feature_names:
                raise ValueError("feature names or order differ from training")
        values = np.asarray(X, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.input_n_features
                or not np.isfinite(values).all()):
            raise ValueError("input must be a finite samples-by-features matrix")
        transformed = self._transform_steps(self._steps, values)
        if self.task == "classification":
            predictions = np.asarray(self._model.predict(transformed), dtype=np.int64).reshape(values.shape[0])
            return np.asarray(self.classes, dtype=str)[predictions]
        return np.asarray(self._model.predict(transformed), dtype=np.float64).reshape(values.shape[0])


__all__ = ["PortableN4MTrainedPipeline", "RefittedN4MPipeline"]
