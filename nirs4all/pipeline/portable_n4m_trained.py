"""Read and replay a cross-language trained n4m pipeline without host pickles.

The envelope carries a shared JSON recipe, MSC/EMSC training references, and
one native N4MM model. Numerical transforms and prediction remain in Methods.
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


class PortableN4MTrainedPipeline:
    """An imported native PLS predictor and its shared retraining recipe.

    Use :meth:`close` or a context manager to release the N4MM handle. The
    original JSON recipe remains available for training a fresh model through
    the normal Python ``PipelineConfigs``/``StepParser`` path.
    """

    def __init__(self, document: dict[str, Any]) -> None:
        from pls4all import Context, Model, inspect_n4mm

        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        expected = {"schema", "manifest_json", "manifest_sha256", "model"}
        if (not isinstance(document, dict) or set(document) != expected
                or document["schema"] != "nirs4all.n4m.trained_pipeline.v1"):
            raise ValueError("unsupported trained n4m pipeline envelope")
        manifest_json = document["manifest_json"]
        manifest_hash = document["manifest_sha256"]
        if (not isinstance(manifest_json, str) or not isinstance(manifest_hash, str)
                or hashlib.sha256(manifest_json.encode("utf-8")).hexdigest() != manifest_hash):
            raise ValueError("trained pipeline manifest hash mismatch")
        manifest = json.loads(manifest_json)
        if (not isinstance(manifest, dict)
                or set(manifest) != {"recipe", "input_n_features", "feature_names",
                                     "preprocessing_owner", "step_states"}):
            raise ValueError("invalid trained pipeline manifest")
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
        # Apply the same config resolver used by ordinary Python nirs4all.
        PipelineConfigs(recipe)
        nodes = recipe["pipeline"]
        if not nodes or set(nodes[-1]) != {"model"}:
            raise ValueError("trained n4m pipeline needs one final model")
        model_node = nodes[-1]["model"]
        if (not isinstance(model_node, dict) or model_node.get("class") != "n4m.PLS"
                or set(model_node) != {"class", "params"}
                or not isinstance(model_node["params"], dict)
                or set(model_node["params"]) != {"n_components"}
                or type(model_node["params"]["n_components"]) is not int
                or model_node["params"]["n_components"] < 1):
            raise ValueError("trained n4m pipeline supports fixed n4m.PLS only")
        owner = manifest["preprocessing_owner"]
        if owner not in {"external", "embedded_methods"}:
            raise ValueError("unsupported preprocessing owner")
        states = manifest["step_states"]
        if not isinstance(states, list):
            raise ValueError("invalid fitted preprocessing state")
        if owner == "embedded_methods":
            if len(nodes) != 3 or len(states) != 2 or any(state is not None for state in states):
                raise ValueError("invalid embedded preprocessing state")
            self._validate_embedded_recipe(nodes[:2])
            output_width = width
        else:
            output_width = self._validate_steps(nodes[:-1], states, width)
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
        if (info.format_version != (2 if owner == "embedded_methods" else 1)
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
        self._nodes = nodes[:-1]
        self._states = states
        self._context = context
        self._model = native_model
        self._document = json.loads(json.dumps(document, allow_nan=False))

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
    def _validate_steps(cls, nodes: list[dict[str, Any]], states: list[Any], width: int) -> int:
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
                width = sum(cls._validate_steps(branch, state["branches"][name], width)
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

        Methods owns every transform and PLS fit. For MSC/EMSC, the portable
        mean reference is recorded from the exact training matrix passed to
        the native ``fit`` call; this mirrors the native reference definition.
        """

        from pls4all import Config, Context, Model, Solver

        from nirs4all.pipeline.config.pipeline_config import PipelineConfigs

        PipelineConfigs(recipe)
        nodes = recipe.get("pipeline")
        if not isinstance(nodes, list) or not nodes or not isinstance(nodes[-1], dict):
            raise ValueError("portable native recipe needs a final model")
        model_node = nodes[-1].get("model")
        if (not isinstance(model_node, dict) or model_node.get("class") != "n4m.PLS"
                or set(model_node) != {"class", "params"}
                or not isinstance(model_node["params"], dict)
                or set(model_node["params"]) != {"n_components"}
                or type(model_node["params"]["n_components"]) is not int
                or model_node["params"]["n_components"] < 1):
            raise ValueError("portable training supports fixed n4m.PLS only")
        columns = getattr(X, "columns", None)
        names = list(columns) if columns is not None else None
        values = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] < 2 or targets.ndim != 1
                or values.shape[0] != targets.shape[0]
                or not np.isfinite(values).all() or not np.isfinite(targets).all()):
            raise ValueError("fit requires finite aligned training rows")
        if names is not None and (len(names) != values.shape[1]
                                  or any(not isinstance(name, str) or not name for name in names)
                                  or len(set(names)) != len(names)):
            raise ValueError("invalid ordered feature names")
        states, transformed = cls._fit_portable_steps(nodes[:-1], values)
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
        manifest_json = json.dumps(manifest, ensure_ascii=False, allow_nan=False,
                                   separators=(",", ":"))
        document = {
            "schema": "nirs4all.n4m.trained_pipeline.v1",
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
        cls, nodes: list[dict[str, Any]], X: np.ndarray,
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
                outputs = [cls._fit_portable_steps(branch, X)
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
                if name not in _STATELESS | _STATEFUL:
                    raise ValueError(f"unsupported native preprocessing: {name}")
                operator = parser.parse(node).operator
                operator.fit(X)
                if name in _STATEFUL:
                    reference = np.mean(X, axis=0, dtype=np.float64)
                    states.append({"kind": name.removeprefix("n4m.").upper(),
                                   "reference": reference.tolist()})
                else:
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
                operator = parser.parse(node).operator
                if node["class"] in _STATEFUL:
                    operator.fit(self._reference(state["reference"], values.shape[1]).reshape(1, -1))
                else:
                    operator.fit(values)
                values = np.asarray(operator.transform(values), dtype=np.float64)
                index += 1
            state_index += 1
        return values

    def predict(self, X: Any) -> np.ndarray:
        """Predict from raw spectra with native Methods transforms and N4MM."""

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
        return result.reshape(values.shape[0])

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
        targets = np.asarray(y, dtype=np.float64)
        if (values.ndim != 2 or values.shape[1] != self.input_n_features
                or targets.ndim != 1 or targets.shape[0] != values.shape[0]
                or not np.isfinite(values).all() or not np.isfinite(targets).all()):
            raise ValueError("retrain requires finite aligned training rows")
        trained_steps, transformed = RefittedN4MPipeline._fit_steps(
            self._nodes, values,
        )
        from nirs4all.pipeline.steps.parser import StepParser

        model = StepParser().parse(self.recipe["pipeline"][-1]).operator
        model.fit(transformed, targets)
        return RefittedN4MPipeline(
            trained_steps, model, self.input_n_features, self.feature_names,
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
    ) -> None:
        self._steps = steps
        self._model = model
        self.input_n_features = width
        self.feature_names = feature_names

    @classmethod
    def _fit_steps(cls, nodes: list[dict[str, Any]], X: np.ndarray) -> tuple[list[Any], np.ndarray]:
        from nirs4all.pipeline.steps.parser import StepParser

        parser = StepParser()
        fitted: list[Any] = []
        index = 0
        while index < len(nodes):
            node = nodes[index]
            if "branch" in node:
                branch_fits = [cls._fit_steps(branch, X) for branch in node["branch"].values()]
                fitted.append([entry[0] for entry in branch_fits])
                X = np.concatenate([entry[1] for entry in branch_fits], axis=1)
                index += 2
            else:
                operator = parser.parse(node).operator
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
        return np.asarray(self._model.predict(transformed), dtype=np.float64).reshape(values.shape[0])


__all__ = ["PortableN4MTrainedPipeline", "RefittedN4MPipeline"]
