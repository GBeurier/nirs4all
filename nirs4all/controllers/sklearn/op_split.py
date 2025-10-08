from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple, TYPE_CHECKING, List
import copy
from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:  # pragma: no cover
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset


def _needs(splitter: Any) -> Tuple[bool, bool]:
    """Return booleans *(needs_y, needs_groups)* for the given splitter.

    Introspects the signature of ``split`` *plus* estimator tags (when
    available) so it works for *any* class respecting the sklearn contract.
    """
    split_fn = getattr(splitter, "split", None)
    if not callable(split_fn):
        # No split method → cannot be a valid splitter
        return False, False

    sig = inspect.signature(split_fn)
    params = sig.parameters

    needs_y = "y" in params # and params["y"].default is inspect._empty
    needs_g = "groups" in params and params["groups"].default is inspect._empty

    # Honour estimator tags (sklearn >=1.3)
    if hasattr(splitter, "_get_tags"):
        tags = splitter._get_tags()
        needs_y = needs_y or tags.get("requires_y", False)

    return needs_y, needs_g


@register_controller
class CrossValidatorController(OperatorController):
    """Controller for **any** sklearn‑compatible splitter (native or custom)."""

    priority = 10  # processed early but after mandatory pre‑processing steps

    @classmethod
    def matches(cls, step: Any, operator: Any, keyword: str) -> bool:  # noqa: D401
        """Return *True* if *operator* behaves like a splitter.

        **Criteria** – must expose a callable ``split`` whose first positional
        argument is named *X*.  Optional presence of ``get_n_splits`` is a plus
        but not mandatory, so user‑defined simple splitters are still accepted.
        """
        split_fn = getattr(operator, "split", None)
        if not callable(split_fn):
            return False
        try:
            sig = inspect.signature(split_fn)
        except (TypeError, ValueError):  # edge‑cases: C‑extensions or cythonised
            return True  # accept – we can still attempt runtime call
        params: List[inspect.Parameter] = [
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        return bool(params) and params[0].name == "X"

    @classmethod
    def use_multi_source(cls) -> bool:  # noqa: D401
        """Cross‑validators themselves are single‑source operators."""
        return False

    @classmethod
    def supports_prediction_mode(cls) -> bool:
        """Cross-validators should not execute during prediction mode."""
        return True

    def execute(  # type: ignore[override]
        self,
        step: Any,
        operator: Any,
        dataset: "SpectroDataset",
        context: Dict[str, Any],
        runner: "PipelineRunner",
        source: int = -1,
        mode: str = "train",
        loaded_binaries: Any = None,
        prediction_store: Any = None
    ): ##TODO manage groups
        """Run ``operator.split`` and store the resulting folds on *dataset*.

        * Smartly supplies ``y`` / ``groups`` only if required.
        * Maps local indices back to the global index space.
        * Stores the list of folds into the dataset for subsequent steps.
        """
        # Skip execution in prediction mode
        # print(f"🔄 Executing cross‑validation with {operator.__class__.__name__}")

        local_context = copy.deepcopy(context)
        local_context["partition"] = "train"
        needs_y, needs_g = _needs(operator)
        X = dataset.x(local_context, layout="2d", concat_source=True)
        # print(X.shape)
        y = dataset.y(local_context) if needs_y else None
        # groups = dataset.groups(local_context) if needs_g else None
        groups = None

        n_samples = X.shape[0]
        # print(f"🔄 Creating folds for {n_samples} samples using {operator.__class__.__name__}")
        kwargs: Dict[str, Any] = {}
        if needs_y:
            if y is None:
                raise ValueError(
                    f"{operator.__class__.__name__} requires y but dataset.y returned None"
                )
            kwargs["y"] = y
        if needs_g:
            if groups is None:
                raise ValueError(
                    f"{operator.__class__.__name__} requires groups but dataset.groups returned None"
                )
            kwargs["groups"] = groups


        if mode != "predict" and mode != "explain":
            folds = list(operator.split(X, **kwargs))  # Convert to list to avoid iterator consumption

            if dataset.x({"partition": "test"}).shape[0] == 0:
                print("⚠️ No test partition found; using first fold as test set.")
                fold_1 = folds[0]
                dataset._indexer.update_by_indices(
                    fold_1[1], {"partition": "test"}
                )
                return context, []
            else:
                dataset.set_folds(folds)

                headers = [f"fold_{i}" for i in range(len(folds))]
                binary = ",".join(headers).encode("utf-8") + b"\n"
                max_train_samples = max(len(train_idx) for train_idx, _ in folds)

                for row_idx in range(max_train_samples):
                    row_values = []
                    for fold_idx, (train_idx, val_idx) in enumerate(folds):
                        if row_idx < len(train_idx):
                            row_values.append(str(train_idx[row_idx]))
                        else:
                            row_values.append("")  # Empty cell if this fold has fewer samples
                    binary += ",".join(row_values).encode("utf-8") + b"\n"

                folds_name = f"folds_{operator.__class__.__name__}"
                if hasattr(operator, "random_state"):
                    seed = getattr(operator, "random_state")
                    if seed is not None:
                        folds_name += f"_seed{seed}"
                folds_name += ".csv"

            # print(f"Generated {len(folds)} folds.")

            return context, [(folds_name, binary)]
        else:
            n_folds = operator.get_n_splits(**kwargs) if hasattr(operator, "get_n_splits") else 1
            dataset.set_folds([(list(range(n_samples)), [])] * n_folds)
            return context, []