from typing import Dict, List, Optional, Sequence, Tuple, Union, cast

import numpy as np
import warnings
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, LabelEncoder


class Targets:
    """Light‑weight manager for target arrays and their processing pipelines.

    * **Storage model** – A *processing version* (e.g. ``"raw"``, ``"numeric"``,
      ``"scaled"``) maps to a **single** ``numpy.ndarray`` of shape
      ``(n_samples, n_targets)``.  All versions share the **same sample order** so
      a row index is unambiguous across versions.
    * **Transformers** – For every derived processing we keep the tuple
      ``(transformer, source_processing)`` so we can chain
      ``inverse_transform`` calls back to an earlier version.
    * **Mutability rules** – Targets must be added *once* via :pymeth:`add_targets`.
      After additional processings have been created, calling
      :pymeth:`add_targets` again would break alignment and is therefore
      forbidden (raises :class:`ValueError`).
    """

    # ---------------------------------------------------------------------
    # Construction helpers
    # ---------------------------------------------------------------------

    def __init__(self) -> None:
        self._data: Dict[str, np.ndarray] = {}
        # key -> (transformer, source_processing)
        self._transformers: Dict[str, Tuple[TransformerMixin, str]] = {}

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_samples(self) -> int:
        """Number of rows stored (0 if empty)."""
        if "raw" not in self._data:
            return 0
        return int(self._data["raw"].shape[0])

    @property
    def processing_versions(self) -> List[str]:
        """Sorted list of available processing names."""
        return sorted(self._data)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_targets(self, y: Union[np.ndarray, Sequence]) -> None:
        """Register *raw* targets and automatically build a *numeric* version.

        Parameters
        ----------
        y
            Array‑like of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.

        Raises
        ------
        ValueError
            If targets were already added, i.e. any processing exists.
        """
        if len(self._transformers.keys()) > 2:  # "raw" + "numeric"
            raise ValueError(
                "Targets already added – you cannot append new samples once "
                "derived processings have been created."
            )

        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        # Append new y to raw values
        if "raw" in self._data:
            self._data["raw"] = np.vstack([self._data["raw"], y])
        else:
            self._data["raw"] = y.copy()

        # Build numeric representation + transformer
        numeric, transformer = self._make_numeric(y)
        if "numeric" in self._data:
            self._data["numeric"] = np.vstack([self._data["numeric"], numeric])
        else:
            self._data["numeric"] = numeric

        self._transformers["numeric"] = (transformer, "raw")

    # ------------------------------------------------------------------
    # Access helpers
    # ------------------------------------------------------------------

    def y(
        self,
        indices: Optional[Union[List[int], np.ndarray]] = None,
        processing: str = "numeric",
    ) -> np.ndarray:
        """Return targets for *processing* and *indices*.

        ``indices`` can be any 1‑D sequence convertible to ``np.ndarray``.  If
        *None* every sample is returned (a view, *not* a copy).
        """
        if processing not in self._data:
            raise ValueError(
                f"Unknown processing '{processing}'. Available: {self.processing_versions}"
            )

        data = self._data[processing]

        if indices is None:
            return data

        indices = np.asarray(indices, dtype=int)
        return data[indices]

    # ------------------------------------------------------------------
    # Transformation plumbing
    # ------------------------------------------------------------------

    def set_y(
        self,
        y: Union[np.ndarray, Sequence],
        transformer: TransformerMixin,
        new_processing: str,
        source_processing: str = "numeric",
    ) -> None:
        """Register an *already transformed* target array.

        All rows *must* align with ``source_processing``.
        """
        if new_processing in self._data:
            warnings.warn(
                f"Processing '{new_processing}' already exists – it will be overwritten.",
                stacklevel=2,
            )

        if source_processing not in self._data:
            raise ValueError(f"Source processing '{source_processing}' not found.")

        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        if y.shape[0] != self._data[source_processing].shape[0]:
            raise ValueError(
                "Shape mismatch – transformed data does not contain the same number "
                "of samples as the source processing."
            )

        self._data[new_processing] = y.copy()
        self._transformers[new_processing] = (transformer, source_processing)

    def invert_transform(
        self,
        y_pred: np.ndarray,
        *,
        from_processing: str,
        to_processing: str = "raw",
    ) -> np.ndarray:
        """Inverse‑transform predictions from *from_processing* back to *to_processing*."""
        if from_processing == to_processing:
            return y_pred

        if from_processing not in self._transformers:
            raise ValueError(f"No transformer chain starting at '{from_processing}'.")

        current = y_pred
        processing = from_processing
        visited = set()

        while processing != to_processing:
            if processing in visited:
                raise ValueError("Circular transformer dependency detected.")
            visited.add(processing)

            transformer, source_proc = self._transformers[processing]
            current = transformer.inverse_transform(current)
            processing = source_proc

        return current

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def get_processing_info(self) -> Dict[str, Dict[str, str]]:
        """Return meta information for every stored processing version."""
        info: Dict[str, Dict[str, str]] = {}
        for name, data in self._data.items():
            meta: Dict[str, str] = {
                "shape": str(data.shape),
                "dtype": str(data.dtype),
            }
            if name in self._transformers:
                trf, src = self._transformers[name]
                meta["transformer"] = type(trf).__name__
                meta["source"] = src
            info[name] = meta
        return info

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------

    def __str__(self) -> str:  # pragma: no cover – debugging aid
        if not self._data:
            return "Targets: <empty>"
        lines = [f"Targets: {self.n_samples} samples"]
        for p in self.processing_versions:
            lines.append(f"  • {p}: {self._data[p].shape}")
        return "\n".join(lines)

    __repr__ = __str__

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_numeric(self, y_raw: np.ndarray) -> Tuple[np.ndarray, TransformerMixin]:
        """Convert *y_raw* to purely numeric data and return (numeric, transformer)."""
        # If we already have a numeric transformer, reuse it
        if "numeric" in self._transformers:
            existing_transformer, _ = self._transformers["numeric"]
            # Check if transformer has transform method
            if hasattr(existing_transformer, 'transform'):
                y_numeric = existing_transformer.transform(y_raw)
                return y_numeric.astype(np.float32), existing_transformer

        # Otherwise, create a new transformer
        y_numeric = y_raw.copy()
        transformers = []

        for col in range(y_raw.shape[1]):
            col_data = y_raw[:, col]

            if col_data.dtype.kind in {"U", "S", "O"}:  # strings / objects
                le = LabelEncoder()
                y_numeric[:, col] = le.fit_transform(col_data)
                transformers.append((f"col_{col}", le, [col]))
            elif not np.issubdtype(col_data.dtype, np.number):
                # Attempt numeric coercion, else fallback to LabelEncoder
                try:
                    y_numeric[:, col] = col_data.astype(np.float32)
                except (ValueError, TypeError):
                    le = LabelEncoder()
                    y_numeric[:, col] = le.fit_transform(col_data.astype(str))
                    transformers.append((f"col_{col}", le, [col]))

        # Build ColumnTransformer (identity if no converters were required)
        if transformers:
            transformer: TransformerMixin = ColumnTransformer(transformers, remainder="passthrough")
            transformer.fit(y_raw)
        else:
            transformer = FunctionTransformer(validate=False)
            transformer.fit(y_raw)

        return y_numeric.astype(np.float32), transformer
