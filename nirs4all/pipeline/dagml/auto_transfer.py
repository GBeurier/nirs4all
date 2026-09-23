"""Host operator for transfer-guided preprocessing in a DAG-ML transform task."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DagMLAutoTransferPreprocessor(TransformerMixin, BaseEstimator):
    """Select a preprocessing from explicit source and target partition views.

    DAG-ML owns the all-observations fit scope and invokes the host transform in
    each native fold/refit task. The selector's numerical work remains in the
    nirs4all operator, and its recommendation is retained in the fitted chain.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config

    def fit(self, X: Any, y: Any = None) -> DagMLAutoTransferPreprocessor:
        """Reject an unpartitioned fit, which cannot preserve transfer semantics."""
        raise ValueError("auto_transfer_preproc requires partitioned source and target fit views")

    def fit_with_views(self, views: dict[str, tuple[np.ndarray, np.ndarray | None]]) -> DagMLAutoTransferPreprocessor:
        """Run the legacy selector on DAG-owned source and target cohorts."""
        self.select_with_views(views)
        return self.fit_selected(self.recommendation_, views["train"][0])

    def select_with_views(self, views: dict[str, tuple[np.ndarray, np.ndarray | None]]) -> DagMLAutoTransferPreprocessor:
        """Choose once from the joint feature view before source-local fitting."""
        from nirs4all.analysis import TransferPreprocessingSelector
        from nirs4all.controllers.data.auto_transfer_preproc import AutoTransferPreprocessingController

        controller = AutoTransferPreprocessingController()
        config = controller._parse_config(self.config)  # noqa: SLF001 -- one shared config contract with legacy
        source = views.get(config["source_partition"])
        target = views.get(config["target_partition"])
        if source is None or target is None or len(source[0]) == 0 or len(target[0]) == 0:
            raise ValueError("auto_transfer_preproc requires nonempty source and target partitions")
        selector = TransferPreprocessingSelector(**controller._build_selector_kwargs(config))  # noqa: SLF001
        results = selector.fit(source[0], target[0], source[1], target[1])
        self.recommendation_ = {
            "pipeline_spec": results.to_pipeline_spec(
                top_k=config["top_k"], use_augmentation=config["use_augmentation"],
            ),
            "best_name": results.best.name,
            "transfer_score": results.best.transfer_score,
            "improvement_pct": results.best.improvement_pct,
        }
        return self

    def fit_selected(self, recommendation: dict[str, Any], train_x: np.ndarray) -> DagMLAutoTransferPreprocessor:
        """Fit the selected preprocessing on one source's training features."""
        from nirs4all.controllers.data.auto_transfer_preproc import AutoTransferPreprocessingController

        config = AutoTransferPreprocessingController()._parse_config(self.config)  # noqa: SLF001
        self.recommendation_ = recommendation
        self._chains: list[list[Any]] = []
        self._augment = False
        if config["apply_recommendation"]:
            from nirs4all.analysis import get_base_preprocessings

            preprocessings = get_base_preprocessings()
            spec = self.recommendation_["pipeline_spec"]
            if isinstance(spec, str):
                names = [spec]
            elif isinstance(spec, list) and all(isinstance(name, str) for name in spec):
                names = spec
            elif isinstance(spec, dict) and isinstance(spec.get("feature_augmentation"), list):
                names = spec["feature_augmentation"]
                self._augment = True
            else:
                raise ValueError(f"unsupported transfer preprocessing recommendation {spec!r}")
            current = np.asarray(train_x)
            for name in names:
                chain: list[Any] = []
                branch = np.asarray(train_x) if self._augment else current
                for component in name.split(">"):
                    if component not in preprocessings:
                        raise ValueError(f"unknown transfer preprocessing {component!r}")
                    transform = deepcopy(preprocessings[component])
                    transform.fit(branch)
                    branch = np.asarray(transform.transform(branch))
                    chain.append(transform)
                self._chains.append(chain)
                if not self._augment:
                    current = branch
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Replay the selected transforms, or preserve X in analysis-only mode."""
        check_is_fitted(self, "recommendation_")
        raw = np.asarray(X)
        if self._augment:
            outputs = [raw]
            for chain in self._chains:
                current = raw
                for transform in chain:
                    current = np.asarray(transform.transform(current))
                outputs.append(current)
            return np.hstack(outputs)
        current = raw
        for chain in self._chains:
            for transform in chain:
                current = np.asarray(transform.transform(current))
        return current
