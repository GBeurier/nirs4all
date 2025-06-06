from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

import numpy as np
import polars as pl


def _to_list_of_float(pred) -> list[float]:
    """
    Helper – normalises a single prediction to `list[float]`.

    • scalar (np.number / float / int)     → [value]
    • 1-D numpy array / sequence           → list(value)
    • 2-D numpy array (1 × n)              → list(row)
    """
    if isinstance(pred, (float, int, np.number)):
        return [float(pred)]

    if isinstance(pred, (np.ndarray, list, tuple)):
        arr = np.asarray(pred).astype("float32").squeeze()
        if arr.ndim == 0:
            return [float(arr)]
        if arr.ndim == 1:
            return arr.tolist()
        if arr.ndim == 2 and arr.shape[0] == 1:
            return arr[0].tolist()

    raise TypeError(f"Unsupported y_pred shape/type: {type(pred)=}, {pred=}")


class PredictionTable:
    """
    Stores model outputs in typed Arrow columns for efficient batching.

    Columns
    -------
    row        : pl.Int64
    model_id   : pl.Utf8
    step_id    : pl.Utf8
    y_pred     : pl.List(pl.Float32)     (1-or-many floats per row)
    context    : pl.Struct (optional)    arbitrary extra metadata
    """

    # ─────────────────────────────── ctor ────────────────────────────────
    def __init__(self, df: pl.DataFrame | None = None) -> None:
        if df is None:
            self.df = pl.DataFrame(
                {
                    "row": pl.Series([], dtype=pl.Int64),
                    "model_id": pl.Series([], dtype=pl.Utf8),
                    "step_id": pl.Series([], dtype=pl.Utf8),
                    "y_pred": pl.Series([], dtype=pl.List(pl.Float32)),
                    # context column added lazily on first insert
                }
            )
        else:
            self.df = df

    # ───────────────────── public API: add_predictions ───────────────────
    def add_predictions(
        self,
        row_ids: Sequence[int] | int,
        y_pred,
        model_id: str,
        step_id: str,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Append a batch of predictions.

        * `row_ids` must align 1-to-1 with rows in `y_pred`
        * `y_pred` can be scalar, 1-D, 2-D or iterable – it is normalised
          to a list[float] stored in a List(Float32) column
        * `context` is any JSON-serialisable dict. Keys are unioned across
          all rows in the batch and stored in a Struct column.
        """
        # ---------- normalise inputs ------------------------------------
        if isinstance(row_ids, (int, np.integer)):
            row_ids = [int(row_ids)]

        # ensure row_ids iterable length matches y_pred records
        if (
            hasattr(y_pred, "shape")
            and len(getattr(y_pred, "shape", ())) == 2
            and y_pred.shape[0] == len(row_ids)
        ):
            preds_iter = y_pred
        elif (
            hasattr(y_pred, "shape")
            and len(getattr(y_pred, "shape", ())) == 1
            and len(row_ids) == 1
        ):
            preds_iter = [y_pred]
        else:
            preds_iter = y_pred

        y_pred_column = pl.Series(
            "y_pred",
            [_to_list_of_float(p) for p in preds_iter],
        ).cast(pl.List(pl.Float32))

        # ---------- context handling ------------------------------------
        if context is not None:
            # expand single dict to all rows
            ctx_list = [context] * len(row_ids)
            # union of keys
            union_keys = {k for d in ctx_list for k in d.keys()}
            # guarantee equal schema per row
            ctx_list = [
                {k: d.get(k) for k in union_keys} for d in ctx_list
            ]
            context_series = pl.Series("context", ctx_list)
            new_rows = pl.DataFrame(
                {
                    "row": row_ids,
                    "model_id": [model_id] * len(row_ids),
                    "step_id": [step_id] * len(row_ids),
                }
            ).with_columns([y_pred_column, context_series])

        else:  # no context
            new_rows = pl.DataFrame(
                {
                    "row": row_ids,
                    "model_id": [model_id] * len(row_ids),
                    "step_id": [step_id] * len(row_ids),
                }
            ).with_columns([y_pred_column])

        self.df = pl.concat([self.df, new_rows], how="vertical", rechunk=True)

    # ───────────────────────────── filter ────────────────────────────────
    def filter(self, **predicates) -> pl.DataFrame:
        """Vectorised filtering by column == value predicates."""
        out = self.df
        for col, value in predicates.items():
            if col in out.columns:
                out = out.filter(pl.col(col) == value)
        return out

    # ─────────────────────────── duplicate ───────────────────────────────
    def duplicate(self, row_mask, n_copies: int) -> pl.DataFrame:
        """Return *n* full copies of the selected rows (no insertion)."""
        base = self.df[row_mask] if not isinstance(row_mask, (list, tuple)) else self.df.filter(pl.col("row").is_in(row_mask))
        if n_copies <= 0:
            return pl.DataFrame()
        return pl.concat([base] * n_copies, rechunk=True)
