from __future__ import annotations

import inspect
from typing import Any, Dict, Tuple, TYPE_CHECKING, List

from nirs4all.controllers.controller import OperatorController
from nirs4all.controllers.registry import register_controller

if TYPE_CHECKING:  # pragma: no cover
    from nirs4all.pipeline.runner import PipelineRunner
    from nirs4all.dataset.dataset import SpectroDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

    needs_y = "y" in params and params["y"].default is inspect._empty
    needs_g = "groups" in params and params["groups"].default is inspect._empty

    # Honour estimator tags (sklearn >=1.3)
    if hasattr(splitter, "_get_tags"):
        tags = splitter._get_tags()
        needs_y = needs_y or tags.get("requires_y", False)

    return needs_y, needs_g


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

@register_controller
class CrossValidatorController(OperatorController):
    """Controller for **any** sklearn‑compatible splitter (native or custom)."""

    priority = 20  # processed early but after mandatory pre‑processing steps

    # ------------------------------------------------------------------
    # Framework integration helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------
    def execute(  # type: ignore[override]
        self,
        step: Any,
        operator: Any,
        dataset: "SpectroDataset",
        context: Dict[str, Any],
        runner: "PipelineRunner",
        source: int = -1,
    ):
        """Run ``operator.split`` and store the resulting folds on *dataset*.

        * Smartly supplies ``y`` / ``groups`` only if required.
        * Maps local indices back to the global index space.
        * Stores the list of folds into the dataset for subsequent steps.
        """
        print(f"🔄 Executing cross‑validation with {operator.__class__.__name__}")

        # -----------------------------
        # Retrieve data
        # -----------------------------
        X = dataset.x(context, layout="2d", source=source)
        if isinstance(X, tuple):
            # Keep first source to avoid mismatched lengths
            X = X[0]
        y = dataset.y(context)
        groups = dataset.groups(context) if hasattr(dataset, "groups") else None

        n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
        print(
            f"🔄 Creating folds for {n_samples} samples using {operator.__class__.__name__}"
        )

        current_indices, _ = dataset.features.index.get_indices(context)

        # -----------------------------
        # Build kwargs for split()
        # -----------------------------
        needs_y, needs_g = _needs(operator)
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

        # -----------------------------
        # Generate splits
        # -----------------------------
        fold_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(operator.split(X, **kwargs)):
            train_samples = [current_indices[i] for i in train_idx]
            val_samples = [current_indices[i] for i in val_idx]
            fold_splits.append((train_samples, val_samples))
            print(
                f"   📊 Fold {fold_idx}: {len(train_samples)} train, {len(val_samples)} val samples"
            )

        dataset.set_folds(fold_splits)
        print(f"✅ Successfully created {len(fold_splits)} CV folds")

        return context
