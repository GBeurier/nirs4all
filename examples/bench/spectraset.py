from __future__ import annotations

import numpy as np
import polars as pl
from typing import Any, Sequence, Union

class SpectraDataset:
    """
    Dataset ultra-rapide pour ML basé sur polars/Arrow.
    
    Chaque spectre est stocké dans une colonne 'spectrum' de type List[Float64],
    et un identifiant unique 'row_id' est assigné à chaque entrée.
    Les index par défaut sont gérés dynamiquement, avec broadcast ou vecteur dédié.
    """

    DEFAULT_INDEX = (
        "origin", "sample", "type", "set",
        "processing", "augmentation", "branch"
    )
    _DEFAULT_VALUES: dict[str, Any] = {
        "set": "train",
        "processing": "raw",
        "augmentation": "raw",
        "branch": 0,
    }

    def __init__(self) -> None:
        self.df: pl.DataFrame | None = None
        self._next_id: int = 0

    def _mask(self, **filters: Any) -> pl.Expr:
        exprs = []
        for key, val in filters.items():
            if isinstance(val, (list, tuple, set, np.ndarray)):
                exprs.append(pl.col(key).is_in(val))
            else:
                exprs.append(pl.col(key) == val)
        return exprs[0] if len(exprs) == 1 else pl.all_horizontal(exprs)

    def _select(self, **filters: Any) -> pl.DataFrame:
        if self.df is None:
            return pl.DataFrame()
        return self.df.filter(self._mask(**filters)) if filters else self.df

    def add_spectra(
        self,
        spectra: Sequence[Sequence[float]],
        target: Sequence[Any] | None = None,
        **index_values: Union[Any, Sequence[Any]],
    ) -> None:
        n = len(spectra)
        tgt = target if target is not None else [None] * n
        if len(tgt) != n:
            raise ValueError("La longueur de 'target' ne correspond pas au nombre de spectres")

        data: dict[str, list[Any]] = {}
        for k in self.DEFAULT_INDEX:
            if k in index_values:
                v = index_values[k]
                if isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
                    if len(v) != n:
                        raise ValueError(f"Index '{k}' longueur {len(v)} != {n}")
                    data[k] = list(v)
                else:
                    data[k] = [v] * n
            else:
                default = self._DEFAULT_VALUES.get(k)
                data[k] = [default] * n

        data["spectrum"] = [list(vec) for vec in spectra]
        data["target"] = list(tgt)
        data["row_id"] = list(range(self._next_id, self._next_id + n))
        self._next_id += n

        new_df = pl.DataFrame(data)
        if self.df is None:
            self.df = new_df
        else:
            self.df = pl.concat([self.df, new_df], how="vertical")

    def change_spectra(
        self,
        new_spectra: Sequence[Sequence[float]],
        **filters: Any,
    ) -> None:
        if self.df is None:
            return
        mask = self._mask(**filters)
        n = self.df.filter(mask).height
        if n != len(new_spectra):
            raise ValueError("new_spectra count mismatch")
        self.df = self.df.with_columns(
            pl.when(mask)
              .then(pl.Series(new_spectra, dtype=pl.List(pl.Float64)))
              .otherwise(pl.col("spectrum"))
              .alias("spectrum")
        )

    def add_tag(
        self,
        tag_name: str,
        tag_values: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """
        Ajoute une nouvelle colonne `tag_name`.
        - Si `tag_values` scalaire → broadcast sur toutes les lignes filtrées.
        - Si `tag_values` séquence de longueur m → assigné séquentiellement aux m lignes filtrées.
        """
        if self.df is None:
            raise ValueError("Dataset vide.")
        if tag_name in self.df.columns:
            raise ValueError(f"Le tag '{tag_name}' existe déjà ; utilisez set_tag.")

        df = self.df
        mask_series = df.select(self._mask(**filters).alias("_mask"))
        mask = mask_series.to_series()
        total = len(mask)
        # Construire liste complète
        full: list[Any] = [None] * total
        if isinstance(tag_values, Sequence) and not isinstance(tag_values, (str, bytes)):
            values = list(tag_values)
            if sum(mask) != len(values):
                raise ValueError("Le nombre de 'tag_values' ne correspond pas aux lignes filtrées")
            it = iter(values)
            for i, m in enumerate(mask):
                if m:
                    full[i] = next(it)
        else:
            for i, m in enumerate(mask):
                if m:
                    full[i] = tag_values
        self.df = df.with_columns(pl.Series(tag_name, full))

    def set_tag(
        self,
        tag_name: str,
        new_value: Union[Any, Sequence[Any]],
        **filters: Any,
    ) -> None:
        """
        Change les valeurs d'un tag existant.
        - Si `new_value` scalaire → broadcast.
        - Si `new_value` séquence de longueur m → assigné séquentiellement aux m lignes filtrées.
        """
        if self.df is None:
            raise ValueError("Dataset vide.")
        if tag_name not in self.df.columns:
            raise KeyError(f"Tag '{tag_name}' inexistant.")

        df = self.df
        mask_series = df.select(self._mask(**filters).alias("_mask")).to_series()
        total = len(mask_series)
        full = df.get_column(tag_name).to_list()
        if isinstance(new_value, Sequence) and not isinstance(new_value, (str, bytes)):
            vals = list(new_value)
            if sum(mask_series) != len(vals):
                raise ValueError("Le nombre de 'new_value' ne correspond pas aux lignes filtrées")
            it = iter(vals)
            for i, m in enumerate(mask_series):
                if m:
                    full[i] = next(it)
        else:
            for i, m in enumerate(mask_series):
                if m:
                    full[i] = new_value
        self.df = df.with_columns(pl.Series(tag_name, full))

    def X(
        self,
        pad: bool = False,
        pad_value: float = np.nan,
        as_arrow: bool = False,
        return_ids: bool = False,
        **filters: Any,
    ) -> Union[np.ndarray, tuple[Union[np.ndarray, "pyarrow.ListArray"], np.ndarray]]:
        sub = self._select(**filters)
        spectra = sub["spectrum"].to_list()

        if as_arrow and not pad:
            import pyarrow as pa
            arr = pa.array(spectra, type=pa.list_(pa.float64()))
            ids = sub["row_id"].to_numpy() if return_ids else None
            return (arr, ids) if return_ids else arr

        if pad:
            max_len = max((len(v) for v in spectra), default=0)
            out = np.full((len(spectra), max_len), pad_value, dtype=np.float64)
            for i, vec in enumerate(spectra):
                out[i, : len(vec)] = vec
        else:
            out = np.array(spectra, dtype=object)

        if return_ids:
            ids = sub["row_id"].to_numpy()
            return out, ids
        return out

    def y(self, **filters: Any) -> np.ndarray:
        sub = self._select(**filters)
        return sub["target"].to_numpy()

    def to_arrow_table(self) -> "pyarrow.Table":
        return self.df.to_arrow()  # type: ignore

    def __len__(self) -> int:
        return self.df.height if self.df is not None else 0

    def __repr__(self) -> str:
        cols = self.df.columns if self.df is not None else []
        return f"SpectraDataset(n={len(self)}, cols={cols})"
