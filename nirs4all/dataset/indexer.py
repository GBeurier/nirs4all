import polars as pl
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np


class Indexer:
    """
    Gestionnaire d'index de samples pour analyses ML/DL avec optimisation
    des accès contigus et gestion des filtres.
    """

    def __init__(self):
        self.df = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32), # row index
            "sample": pl.Series([], dtype=pl.Int32), # index de sample, int = sample original, null = augmentation
            "origin": pl.Series([], dtype=pl.Int32), # null = sample original, int = augmentation
            "partition": pl.Series([], dtype=pl.Categorical), # train, test
            "group": pl.Series([], dtype=pl.Int8),
            "branch": pl.Series([], dtype=pl.Int8),
            "processing": pl.Series([], dtype=pl.Categorical),
            "augmentation": pl.Series([], dtype=pl.Categorical),
        })

        self.default_values = {
            "partition": "train",
            "group": 0,
            "branch": 0,
            "processing": "raw",
        }

    def _apply_filters(self, filters: Dict[str, Any]) -> pl.DataFrame:
        condition = self._build_filter_condition(filters)
        return self.df.filter(condition)

    def _build_filter_condition(self, filters: Dict[str, Any]) -> pl.Expr:
        conditions = []
        for col, value in filters.items():
            if col not in self.df.columns:
                continue
            if isinstance(value, list):
                conditions.append(pl.col(col).is_in(value))
            elif value is None:
                conditions.append(pl.col(col).is_null())
            else:
                conditions.append(pl.col(col) == value)

        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond

        return condition

    def x_indices(self, filters: Dict[str, Any]) -> np.ndarray:
        filtered_df = self._apply_filters(filters) if filters else self.df
        indices = filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32)
        return indices

    def y_indices(self, filters: Dict[str, Any]) -> np.ndarray:
        filtered_df = self._apply_filters(filters) if filters else self.df
        result = filtered_df.with_columns(
            pl.when(pl.col("origin").is_null() | pl.col("origin").is_nan())
            .then(pl.col("sample"))
            .otherwise(pl.col("origin")).cast(pl.Int32).alias("y_index")
        )
        return result["y_index"].to_numpy().astype(np.int32)

    def add_new_samples(
        self,
        samples: List[int],
        partition: str = "train",
        group: int = 0,
        branch: int = 0,
        processing: str = "raw",
        augmentation: Optional[str] = None,
    ) -> List[int]:
        pass

    # def add_samples(self, samples: List[int], partition: str = "train", group: int = 0)


    def add_rows(
        self,
        n_rows: int,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> List[int]:

        if n_rows <= 0:
            return []

        overrides = overrides or {}

        next_row_index = self.next_row_index()
        next_sample_index = self.next_sample_index()

        cols: Dict[str, pl.Series] = {}
        for col in self.df.columns:
            if col == "row":
                row_indices = range(next_row_index, next_row_index + n_rows)
                cols[col] = pl.Series(row_indices, dtype=pl.Int32)
                continue
            if col == "sample":
                if col in overrides:
                    val = overrides[col]
                    if isinstance(val, list):
                        if len(val) != n_rows:
                            raise ValueError(
                                f"Override list for '{col}' should have {n_rows} elements"
                            )
                        cols[col] = pl.Series(val, dtype=pl.Int32)
                    else:
                        cols[col] = pl.Series([val] * n_rows, dtype=pl.Int32)
                else:
                    index_values = range(next_sample_index, next_sample_index + n_rows)
                    cols[col] = pl.Series(index_values, dtype=pl.Int32)
                    cols["origin"] = pl.Series(index_values, dtype=pl.Int32)
                continue

            val = overrides.get(col, self.default_values.get(col, None))
            expected_dtype = self.df.schema[col]

            if isinstance(val, list):
                if len(val) != n_rows:
                    raise ValueError(
                        f"Override list for '{col}' should have {n_rows} elements"
                    )
                cols[col] = pl.Series(val, dtype=expected_dtype)
            else:
                cols[col] = pl.Series([val] * n_rows, dtype=expected_dtype)

        new_df = pl.DataFrame(cols)
        indices = new_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32).tolist()
        self.df = pl.concat([self.df, new_df], how="vertical")
        return indices















    def update_by_filter(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> None:
        condition = self._build_filter_condition(filters)

        for col, value in updates.items():
            self.df = self.df.with_columns(
                pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

    def next_row_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["row"].max()) + 1

    def next_sample_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["sample"].max()) + 1

    def get_column_values(self, col: str, filters: Optional[Dict[str, Any]] = None) -> List[Any]:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")

        # Apply filters if provided, otherwise use the full dataframe
        filtered_df = self._apply_filters(filters) if filters else self.df
        return filtered_df.select(pl.col(col)).to_series().to_list()

    def uniques(self, col: str) -> List[Any]:
        if col not in self.df.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        return self.df.select(pl.col(col)).unique().to_series().to_list()



    def augment_rows(self, samples: List[int], count: Union[int, List[int]], augmentation_id: str) -> List[int]:

        if isinstance(count, int):
            count = [count] * len(samples)

        if len(count) != len(samples):
            raise ValueError("count must be an int or a list with the same length as samples")

        overrides = {"origin": samples.copy()}
        filtered_df = self.df.filter(pl.col("sample").is_in(samples))
        processings = filtered_df.select(pl.col("processing")).to_series().to_list()
        overrides["processing"] = [processings[i] * count[i] for i in range(len(samples))]
        overrides["augmentation"] = augmentation_id

        return self.add_rows(sum(count), overrides=overrides)









    def __repr__(self):
        return str(self.df)