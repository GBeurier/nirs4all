from __future__ import annotations
import polars as pl
from typing import Mapping, Sequence


class IndexFrame:
    REQUIRED = ("row", "sample", "origin", "partition", "group", "branch", "processing")

    def __init__(self, df: pl.DataFrame) -> None:
        missing = [c for c in self.REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.df = df
        # central counter – _never_ goes backwards
        if self.df.is_empty():
            self._next_row_id: int = 0
        else:
            self._next_row_id = int(self.df.select(pl.col("row").max()).item()) + 1

    def reserve_row_ids(self, n: int = 1) -> list[int]:
        """
        Atomically reserve **n** fresh, unique row IDs.

        The IDs are consecutive, monotonically increasing and never reused
        even if rows get deleted later.
        """
        start = self._next_row_id
        self._next_row_id += n
        return list(range(start, start + n))

    def filter(self, **predicates) -> pl.DataFrame:
        result = self.df
        for col, value in predicates.items():
            if col in result.columns:
                result = result.filter(pl.col(col) == value)
        return result

    def rows(self, **predicates) -> Sequence[int]:
        filtered = self.filter(**predicates)
        return filtered.select("row").to_series().to_list()

    def duplicate(self, row_mask, n_copies: int,
                  override: Mapping[str, object] | None = None) -> pl.DataFrame:

        base_df = self.df.filter(
            pl.col("row").is_in(row_mask)
            if isinstance(row_mask, (list, tuple))
            else row_mask
        )
        if base_df.is_empty() or n_copies <= 0:
            return pl.DataFrame()            # nothing to duplicate

        duplicates = []
        for _ in range(n_copies):
            dup = base_df.clone()

            # allocate exactly len(dup) fresh IDs in one go
            new_rows = self.reserve_row_ids(len(dup))
            dup = dup.with_columns(
                pl.Series("row", new_rows, dtype=dup.select("row").dtypes[0])
            )

            if override:
                for col, value in override.items():
                    dtype = self.df.select(col).dtypes[0]
                    dup = dup.with_columns(pl.lit(value, dtype=dtype).alias(col))

            duplicates.append(dup)

        return pl.concat(duplicates, rechunk=True)

    def update(self, row_mask, **updates) -> None:
        if isinstance(row_mask, (list, tuple)):
            mask = pl.col("row").is_in(row_mask)
        else:
            mask = row_mask

        for col, value in updates.items():
            if col not in self.df.columns:
                # Add the column if it doesn't exist
                self.df = self.df.with_columns(pl.lit(None).alias(col))

            self.df = self.df.with_columns(
                pl.when(mask).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )
