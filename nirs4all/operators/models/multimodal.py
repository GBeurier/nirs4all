"""Sklearn host operators for sample-aligned multimodal prediction.

The coordinator owns sample alignment, folds, selection and scoring. These
operators only learn source representations and a predictor for the rows given
to ``fit``. Raw tensors reach their source transformer without flattening.
"""

from collections.abc import Mapping
from typing import Any, Self

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin, clone, is_classifier
from sklearn.decomposition import PCA
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_consistent_length, check_is_fitted, column_or_1d


class TensorPCA(TransformerMixin, BaseEstimator):
    """Learn a PCA representation of fixed-shape images or temporal tensors.

    All non-sample dimensions are flattened inside this encoder, after checking
    their exact shape. PCA is fitted only on the supplied training rows. No raw
    samples are retained by this encoder.

    Args:
        n_components: Number of components, or a variance fraction accepted by
            sklearn PCA. ``None`` retains all available components.
        whiten: Whether to scale components to unit variance.
        random_state: Seed passed to sklearn PCA.
    """

    def __init__(self, n_components: int | float | None = None, *, whiten: bool = False, random_state: int | None = None):
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X: Any, y: Any = None) -> "TensorPCA":
        """Fit the encoder on an array with samples on its first axis."""
        values = check_array(X, allow_nd=True, dtype="numeric")
        flat = values.reshape(values.shape[0], -1)
        pca = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        pca.fit(flat, y)
        self.pca_ = pca
        self.input_shape_ = values.shape[1:]
        self.n_features_in_ = flat.shape[1]
        self.n_components_ = pca.n_components_
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Encode new samples with the fitted shape and PCA state."""
        check_is_fitted(self, ["pca_", "input_shape_"])
        values = check_array(X, allow_nd=True, dtype="numeric")
        if values.shape[1:] != self.input_shape_:
            raise ValueError(f"TensorPCA expected input shape {self.input_shape_}, got {values.shape[1:]}.")
        return np.asarray(self.pca_.transform(values.reshape(values.shape[0], -1)))


class _MultimodalEstimator(BaseEstimator):
    """Fit named source encoders and fuse their representations for prediction.

    ``X`` is a list of sample-aligned blocks in the insertion order of
    ``transformers``. A block can be a spectral matrix, an image/time tensor, or
    a mixed tabular matrix/DataFrame. Each transformer receives its original
    block and ``y``; it must produce a numeric two-dimensional representation.
    Use sklearn ``Pipeline`` for a source-specific transformation chain and
    ``ColumnTransformer`` for mixed numeric/categorical columns.

    ``early`` fusion concatenates encoded blocks. ``intermediate`` fusion passes
    the list of encoded blocks to a model that supports multiple inputs. Source
    weights multiply encoded features in either mode. Each fit clones all
    transformers and the model, so a coordinator can fit fold-local clones.

    Args:
        transformers: Ordered mapping from source name to sklearn transformer,
            ``None`` or ``"passthrough"``. Passthrough requires a numeric matrix.
        model: Cloneable predictor. Intermediate fusion requires list inputs.
        fusion: ``"early"`` or ``"intermediate"``.
        source_weights: Optional mapping of source names to nonnegative weights.
            Omitted sources have weight one. Zero permits a source ablation.
        missing_source_policy: ``"error"`` requires all modalities. With
            ``"zero_with_indicator"``, encoders see only present rows; absent
            encoded features are zero and each source gains a presence column.
            Source weights multiply both features and their presence column.

    Attributes:
        source_names_: Source order recorded during fitting.
        input_shapes_: Per-source shapes excluding the sample axis.
        transformers_: Fitted private clones keyed by source name.
        model_: Fitted private model clone.

    Nested parameters such as ``transformers__image__n_components``,
    ``model__alpha`` and ``source_weights__image`` can be tuned through sklearn's
    usual ``get_params`` / ``set_params`` API. The wrapper retains no training
    arrays; persistence includes the fitted encoders and model. Any sample
    retention performed by a user-supplied estimator remains its own contract.
    """

    def __init__(
        self,
        transformers: Mapping[str, Any],
        model: Any,
        *,
        fusion: str = "early",
        source_weights: Mapping[str, float] | None = None,
        missing_source_policy: str = "error",
    ):
        self.transformers = transformers
        self.model = model
        self.fusion = fusion
        self.source_weights = source_weights
        self.missing_source_policy = missing_source_policy

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Expose model, named source and source-weight parameters for tuning."""
        params: dict[str, Any] = super().get_params(deep=deep)
        if deep and isinstance(self.transformers, Mapping):
            for name, transformer in self.transformers.items():
                params[f"transformers__{name}"] = transformer
                if hasattr(transformer, "get_params"):
                    params.update({f"transformers__{name}__{key}": value for key, value in transformer.get_params(deep=True).items()})
                params[f"source_weights__{name}"] = (self.source_weights or {}).get(name, 1.0)
        return params

    def set_params(self, **params: Any) -> Self:
        """Set standard sklearn parameters, including entries in source maps."""
        top_level = {key: value for key, value in params.items() if "__" not in key}
        super().set_params(**top_level)
        nested = {key: value for key, value in params.items() if "__" in key}
        # Apply replacements before their nested settings, independent of order.
        for key, value in sorted(nested.items(), key=lambda item: item[0].count("__")):
            root, suffix = key.split("__", 1)
            if root not in {"transformers", "source_weights"}:
                super().set_params(**{key: value})
                continue
            name, _, subkey = suffix.partition("__")
            if name not in self.transformers:
                raise ValueError(f"Unknown source {name!r} in parameter {key!r}.")
            if root == "source_weights":
                if subkey:
                    raise ValueError(f"Source weight parameter {key!r} cannot have nested parameters.")
                self.source_weights = {**(self.source_weights or {}), name: value}
            elif subkey:
                transformer = self.transformers[name]
                if not hasattr(transformer, "set_params"):
                    raise ValueError(f"Source {name!r} has no configurable transformer.")
                transformer.set_params(**{subkey: value})
            else:
                self.transformers = {**self.transformers, name: value}
        return self

    @staticmethod
    def _validate_blocks(X: Any, names: tuple[str, ...], shapes: Mapping[str, tuple[int, ...]] | None = None) -> list[Any]:
        if not isinstance(X, list):
            raise ValueError("Multimodal prediction requires a list of source blocks in the configured source order.")
        if len(X) != len(names):
            raise ValueError(f"Expected {len(names)} source blocks {names}, got {len(X)}.")
        for name, block in zip(names, X, strict=True):
            shape = np.shape(block)
            if len(shape) < 2 or any(size == 0 for size in shape):
                raise ValueError(f"Source {name!r} requires a nonempty sample axis and at least one feature axis; got {shape}.")
            if shapes is not None and shape[1:] != shapes[name]:
                raise ValueError(f"Source {name!r} expected input shape {shapes[name]}, got {shape[1:]}.")
        check_consistent_length(*X)
        return X

    @staticmethod
    def _encoded_block(values: Any, name: str, n_samples: int) -> np.ndarray:
        if sparse.issparse(values):
            values = values.toarray()
        try:
            encoded = check_array(values, dtype="numeric")
            encoded = np.asarray(encoded, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Source {name!r} must produce a numeric 2-D representation: {exc}") from exc
        if encoded.shape[0] != n_samples:
            raise ValueError(f"Transformer for source {name!r} changed the sample count from {n_samples} to {encoded.shape[0]}.")
        return encoded

    @staticmethod
    def _validate_source_masks(source_masks: Mapping[str, Any] | None, names: tuple[str, ...], n_samples: int, policy: str, *, fitting: bool = False) -> dict[str, np.ndarray]:
        if not isinstance(policy, str) or policy not in {"error", "zero_with_indicator"}:
            raise ValueError("missing_source_policy must be 'error' or 'zero_with_indicator'.")
        if source_masks is not None and (not isinstance(source_masks, Mapping) or set(source_masks) != set(names)):
            raise ValueError(f"source_masks must be a mapping containing exactly the configured source names {names}.")
        masks = {}
        for name in names:
            raw = np.ones(n_samples, dtype=bool) if source_masks is None else source_masks[name]
            if np.ma.isMaskedArray(raw):
                raise ValueError(f"source_masks[{name!r}] must be an explicit boolean array, not a NumPy masked array.")
            mask = np.asarray(raw)
            if mask.dtype.kind != "b" or mask.shape != (n_samples,):
                raise ValueError(f"source_masks[{name!r}] must be boolean with shape ({n_samples},).")
            if policy == "error" and not mask.all():
                raise ValueError(f"Source {name!r} is missing rows; missing_source_policy='error' requires every source present.")
            if fitting and not mask.any():
                raise ValueError(f"Source {name!r} requires at least one present training row.")
            masks[name] = mask
        return masks

    @staticmethod
    def _take_rows(block: Any, rows: np.ndarray) -> Any:
        return block.iloc[rows] if hasattr(block, "iloc") else block[rows] if sparse.issparse(block) else np.asarray(block)[rows]

    @staticmethod
    def _place_encoded(values: np.ndarray, mask: np.ndarray, weight: float, policy: str) -> np.ndarray:
        if policy == "error":
            return values * weight
        encoded = np.zeros((len(mask), values.shape[1] + 1), dtype=float)
        encoded[mask, :-1] = values
        encoded[:, -1] = mask
        return encoded * weight

    def fit(self, X: list[Any], y: Any, *, source_masks: Mapping[str, Any] | None = None) -> Self:
        """Fit fresh source encoders and a model on exactly the supplied rows."""
        if not isinstance(self.transformers, Mapping) or not self.transformers:
            raise ValueError("transformers must be a nonempty ordered mapping of source names to transformers.")
        names = tuple(self.transformers)
        if any(not isinstance(name, str) or not name or "__" in name for name in names):
            raise ValueError("Source names must be nonempty strings without '__'.")
        if self.fusion not in {"early", "intermediate"}:
            raise ValueError("fusion must be 'early' or 'intermediate'.")
        blocks = self._validate_blocks(X, names)
        if y is None:
            raise ValueError("Multimodal prediction requires targets y.")
        check_consistent_length(blocks[0], y)
        masks = self._validate_source_masks(source_masks, names, np.shape(blocks[0])[0], self.missing_source_policy, fitting=True)
        if self.source_weights is not None and not isinstance(self.source_weights, Mapping):
            raise ValueError("source_weights must be a mapping of configured source names to weights.")
        configured_weights = self.source_weights or {}
        unknown_weights = set(configured_weights) - set(names)
        if unknown_weights:
            raise ValueError(f"Source weights reference unknown sources: {sorted(unknown_weights)}.")
        weights = {name: float(configured_weights.get(name, 1.0)) for name in names}
        if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
            raise ValueError("Source weights must be finite and nonnegative.")

        fitted: dict[str, Any] = {}
        encoded = []
        for name, block in zip(names, blocks, strict=True):
            mask = masks[name]
            present_block = block if mask.all() else self._take_rows(block, mask)
            present_y = y if mask.all() else self._take_rows(y, mask)
            transformer = self.transformers[name]
            if transformer is None or isinstance(transformer, str) and transformer == "passthrough":
                fitted[name] = None
                output = present_block
            else:
                if not hasattr(transformer, "fit") or not hasattr(transformer, "transform"):
                    raise ValueError(f"Source {name!r} requires a cloneable transformer with fit and transform.")
                fitted[name] = clone(transformer)
                fitted[name].fit(present_block, present_y)
                output = fitted[name].transform(present_block)
            values = self._encoded_block(output, name, int(mask.sum()))
            encoded.append(self._place_encoded(values, mask, weights[name], self.missing_source_policy))
        model = clone(self.model)
        model.fit(np.concatenate(encoded, axis=1) if self.fusion == "early" else encoded, y)

        self.source_names_ = names
        self.input_shapes_ = {name: np.shape(block)[1:] for name, block in zip(names, blocks, strict=True)}
        self.output_widths_ = {name: block.shape[1] for name, block in zip(names, encoded, strict=True)}
        self.source_weights_ = weights
        self.fusion_ = self.fusion
        self.missing_source_policy_ = self.missing_source_policy
        self.transformers_ = fitted
        self.model_ = model
        self.n_features_in_ = sum(int(np.prod(shape)) for shape in self.input_shapes_.values())
        return self

    def _encode(self, X: list[Any], *, source_masks: Mapping[str, Any] | None = None) -> np.ndarray | list[np.ndarray]:
        """Apply fitted source encoders with the recorded fusion contract."""
        check_is_fitted(self, ["model_", "transformers_", "source_names_"])
        blocks = self._validate_blocks(X, self.source_names_, self.input_shapes_)
        encoded = []
        n_samples = np.shape(blocks[0])[0]
        masks = self._validate_source_masks(source_masks, self.source_names_, n_samples, self.missing_source_policy_)
        for name, block in zip(self.source_names_, blocks, strict=True):
            mask = masks[name]
            if not mask.any():
                encoded.append(np.zeros((n_samples, self.output_widths_[name]), dtype=float))
                continue
            present_block = block if mask.all() else self._take_rows(block, mask)
            transformer = self.transformers_[name]
            output = present_block if transformer is None else transformer.transform(present_block)
            values = self._encoded_block(output, name, int(mask.sum()))
            values = self._place_encoded(values, mask, self.source_weights_[name], self.missing_source_policy_)
            if values.shape[1] != self.output_widths_[name]:
                raise ValueError(f"Source {name!r} changed encoded width from {self.output_widths_[name]} to {values.shape[1]}.")
            encoded.append(values)
        return np.concatenate(encoded, axis=1) if self.fusion_ == "early" else encoded

    def predict(self, X: list[Any], *, source_masks: Mapping[str, Any] | None = None) -> np.ndarray:
        """Predict using the recorded source order, encoders and fusion state."""
        fused = self._encode(X, source_masks=source_masks)
        return np.asarray(self.model_.predict(fused))


class MultimodalRegressor(RegressorMixin, _MultimodalEstimator):
    """Learn source encoders and a regressor from aligned raw source blocks.

    ``transformers`` maps source names to sklearn transformers, ``None`` or
    ``"passthrough"``. Its insertion order defines the input block order.
    ``fusion='early'`` concatenates encoded features; ``'intermediate'`` passes
    a list of encoded blocks to ``model``. Optional ``source_weights`` multiply
    each encoded block. All learned components are cloned and fitted on the
    supplied rows only. Nested sklearn parameters remain configurable.

    Targets may be one-dimensional or contain multiple output columns when
    supported by the underlying regressor. Target dimensions and prediction
    shapes are preserved; the wrapper never flattens multi-output targets.

    ``target_policy='complete'`` fits one joint model and requires every target
    cell to be observed. ``'per_target'`` fits a complete, independent encoder
    and model chain for each target using only its observed rows. Pass an
    explicit boolean ``target_mask`` to ``fit``; True means observed. Without
    a mask every cell is observed, so nonfinite targets are rejected.
    Per-target fits expose ``target_models_`` and ``target_counts_``, without
    a shared fitted ``model_`` or ``transformers_``. No target is imputed.

    ``source_masks`` maps every source name to a boolean sample vector (True
    means present). It may be passed to fit and predict. Masks default to all
    present. ``missing_source_policy='zero_with_indicator'`` learns encoders
    only from present rows, fills absent encoded features with zero and appends
    one weighted presence column per source, even when every row is present.
    Each source requires at least one present training row for every target;
    prediction may contain entirely absent sources without calling encoders.
    """

    def __init__(
        self,
        transformers: Mapping[str, Any],
        model: Any,
        *,
        fusion: str = "early",
        source_weights: Mapping[str, float] | None = None,
        target_policy: str = "complete",
        missing_source_policy: str = "error",
    ):
        super().__init__(transformers, model, fusion=fusion, source_weights=source_weights, missing_source_policy=missing_source_policy)
        self.target_policy = target_policy

    def fit(self, X: list[Any], y: Any, *, target_mask: Any = None, source_masks: Mapping[str, Any] | None = None) -> Self:
        """Fit joint or target-specific chains using only observed target cells."""
        if not isinstance(self.target_policy, str) or self.target_policy not in {"complete", "per_target"}:
            raise ValueError("target_policy must be 'complete' or 'per_target'.")
        if y is None:
            raise ValueError("MultimodalRegressor requires regression targets y.")
        targets = np.asarray(y)
        if targets.ndim not in (1, 2) or any(size == 0 for size in targets.shape):
            raise ValueError("Regression targets must be a nonempty 1-D or 2-D array.")
        observed = np.ones(targets.shape, dtype=bool) if target_mask is None else np.asarray(target_mask)
        if observed.shape != targets.shape or observed.dtype.kind != "b":
            raise ValueError("target_mask must be boolean with exactly the same shape as y.")
        if self.target_policy == "complete" and not observed.all():
            raise ValueError("target_policy='complete' requires every target cell to be observed; use 'per_target' for partial targets.")
        observed_values = np.asarray(targets[observed], dtype=float)
        if not np.isfinite(observed_values).all():
            raise ValueError("Observed regression targets must be finite; mask absent values explicitly.")
        matrix = targets.reshape(-1, 1) if targets.ndim == 1 else targets
        observed_matrix = observed.reshape(-1, 1) if targets.ndim == 1 else observed
        counts = observed_matrix.sum(axis=0)
        if np.any(counts == 0):
            raise ValueError(f"Every target requires observed training rows; empty target indices: {np.flatnonzero(counts == 0).tolist()}.")
        if self.target_policy == "complete":
            super().fit(X, y, source_masks=source_masks)
            self.__dict__.pop("target_models_", None)
        else:
            if not isinstance(self.transformers, Mapping) or not self.transformers:
                raise ValueError("transformers must be a nonempty ordered mapping of source names to transformers.")
            blocks = self._validate_blocks(X, tuple(self.transformers))
            check_consistent_length(blocks[0], targets)
            masks = self._validate_source_masks(source_masks, tuple(self.transformers), len(targets), self.missing_source_policy, fitting=True)
            for name, mask in masks.items():
                if np.any(~(observed_matrix & mask[:, None]).any(axis=0)):
                    raise ValueError(f"Source {name!r} requires at least one present training row for every target.")
            models = []
            for index in range(matrix.shape[1]):
                rows = observed_matrix[:, index]
                subset = [self._take_rows(block, rows) for block in blocks]
                model = clone(self).set_params(target_policy="complete")
                model.fit(subset, matrix[rows, index], source_masks={name: mask[rows] for name, mask in masks.items()})
                models.append(model)
            self.target_models_ = tuple(models)
            self.source_names_ = models[0].source_names_
            self.input_shapes_ = dict(models[0].input_shapes_)
            self.source_weights_ = dict(models[0].source_weights_)
            self.fusion_ = models[0].fusion_
            self.missing_source_policy_ = models[0].missing_source_policy_
            self.n_features_in_ = models[0].n_features_in_
            for attribute in ("model_", "transformers_", "output_widths_"):
                self.__dict__.pop(attribute, None)
        self.target_counts_ = counts
        self.target_ndim_ = targets.ndim
        self.n_outputs_ = matrix.shape[1]
        return self

    def predict(self, X: list[Any], *, source_masks: Mapping[str, Any] | None = None) -> np.ndarray:
        """Predict every fitted target, retaining the original target rank."""
        if not hasattr(self, "target_models_"):
            return super().predict(X, source_masks=source_masks)
        blocks = self._validate_blocks(X, self.source_names_, self.input_shapes_)
        rows = np.shape(blocks[0])[0]
        columns = []
        for model in self.target_models_:
            prediction = np.asarray(model.predict(blocks, source_masks=source_masks))
            if prediction.shape not in {(rows,), (rows, 1)}:
                raise ValueError("A per-target model must predict exactly one value per sample.")
            columns.append(prediction.reshape(rows))
        values = np.column_stack(columns)
        return values[:, 0] if self.target_ndim_ == 1 else values

    def __sklearn_tags__(self) -> Any:
        from sklearn.utils import get_tags

        tags = super().__sklearn_tags__()
        tags.target_tags.multi_output = self.target_policy == "per_target" or get_tags(self.model).target_tags.multi_output
        return tags

    def _more_tags(self) -> dict[str, Any]:
        # sklearn 1.5 uses dictionary tags; 1.6+ uses __sklearn_tags__ above.
        get_model_tags = getattr(self.model, "_get_tags", None)
        model_tags: dict[str, Any] = get_model_tags() if get_model_tags is not None else {}
        return {"multioutput": self.target_policy == "per_target" or model_tags.get("multioutput", False)}


class MultimodalClassifier(ClassifierMixin, _MultimodalEstimator):
    """Learn source encoders and a classifier from aligned raw source blocks.

    Accepts the same ``transformers``, ``model``, ``fusion`` and
    ``source_weights`` and ``missing_source_policy`` parameters as
    :class:`MultimodalRegressor`. Optional ``source_masks`` are forwarded by
    fit, predict and predict_proba. Each encoder
    receives the original raw block and training labels; all learned components
    are cloned before fitting. Binary and multiclass labels may be strings or
    noncontiguous numbers. Classification requires a single target column.

    ``classes_`` preserves the fitted classifier's exact class order.
    ``predict_proba`` is available only when the configured classifier supports
    it; probability column ``j`` corresponds to ``classes_[j]``. Neither labels
    nor probabilities are encoded, reordered or scored by this wrapper.
    """

    def fit(self, X: list[Any], y: Any, *, source_masks: Mapping[str, Any] | None = None) -> Self:
        """Fit private source encoders and a classifier on the supplied rows."""
        if not is_classifier(self.model):
            raise ValueError("MultimodalClassifier requires a sklearn-compatible classifier as model.")
        if y is None:
            raise ValueError("MultimodalClassifier requires classification targets y.")
        targets = column_or_1d(y, warn=True)
        check_classification_targets(targets)
        super().fit(X, targets, source_masks=source_masks)
        self.classes_ = np.asarray(self.model_.classes_).copy()
        return self

    @available_if(lambda self: hasattr(getattr(self, "model_", self.model), "predict_proba"))
    def predict_proba(self, X: list[Any], *, source_masks: Mapping[str, Any] | None = None) -> np.ndarray:
        """Return probabilities in the fitted classifier's ``classes_`` order."""
        fused = self._encode(X, source_masks=source_masks)
        return np.asarray(self.model_.predict_proba(fused))
