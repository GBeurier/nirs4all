from __future__ import annotations
import polars as pl
from typing import Sequence


class TargetTable:
    """
    Table Y (supervision). Indexée par 'row'.
    Colonnes libres (classification, régression, multi-label…).
    """

    def __init__(self, df: pl.DataFrame) -> None:
        if "row" not in df.columns:
            raise ValueError("TargetTable requires 'row' column")
        self.df = df

    def get(self, row_mask, columns: Sequence[str] | str = "all") -> pl.DataFrame:
        """Récupère les cibles pour les lignes spécifiées."""
        if isinstance(row_mask, (list, tuple)):
            filtered = self.df.filter(pl.col("row").is_in(row_mask))
        else:
            filtered = self.df.filter(row_mask)

        if columns == "all":
            return filtered
        elif isinstance(columns, str):
            return filtered.select(["row", columns])
        else:
            cols = ["row"] + list(columns)
            return filtered.select(cols)

    def add(self, df: pl.DataFrame) -> None:
        """Ajoute de nouvelles lignes de cibles."""
        self.df = pl.concat([self.df, df])

    def update(self, row_mask, **kv) -> None:
        """Met à jour les cibles pour les lignes spécifiées."""
        if isinstance(row_mask, (list, tuple)):
            mask = pl.col("row").is_in(row_mask)
        else:
            mask = row_mask

        for col, value in kv.items():
            if col not in self.df.columns:
                # Ajouter la colonne si elle n'existe pas
                self.df = self.df.with_columns(pl.lit(None).alias(col))

            self.df = self.df.with_columns(
                pl.when(mask).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

    def duplicate(self, row_mask, n_copies: int) -> pl.DataFrame:
        """Duplique des lignes dans la table des targets."""
        if isinstance(row_mask, (list, tuple)):
            base_df = self.df.filter(pl.arange(0, pl.len()).is_in(row_mask))
        else:
            base_df = self.df.filter(row_mask)

        duplicates = []
        for _ in range(n_copies):
            duplicates.append(base_df.clone())

        return pl.concat(duplicates) if duplicates else pl.DataFrame()
