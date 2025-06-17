import polars as pl
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np


class FeatureIndex:
    """
    Gestionnaire d'index de samples pour analyses ML/DL avec optimisation
    des accès contigus et gestion des filtres.
    """

    def __init__(self):
        self.df = pl.DataFrame({
            "row": pl.Series([], dtype=pl.Int32),
            "sample": pl.Series([], dtype=pl.Int32),
            "origin": pl.Series([], dtype=pl.Int32),
            "partition": pl.Series([], dtype=pl.Categorical),
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


    def next_row_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["row"].max()) + 1

    def next_sample_index(self) -> int:
        if len(self.df) == 0:
            return 0
        return int(self.df["sample"].max()) + 1

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

    def get_indices(self, filters: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        filtered_df = self._apply_filters(filters) if filters else self.df

        indices = sorted(set(filtered_df.select(pl.col("sample")).to_series().to_numpy().astype(np.int32).tolist()))
        processings = sorted(set(filtered_df.select(pl.col("processing")).to_series().to_list()))

        return indices, processings

    def update_by_filter(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> None:
        condition = self._build_filter_condition(filters)

        for col, value in updates.items():
            self.df = self.df.with_columns(
                pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

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

        # Combine toutes les conditions avec AND
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond

        return condition




    # def get_contiguous_ranges(self, filters: Dict[str, Any]) -> np.ndarray:
    #     """
    #     Récupère les ranges contigus d'index pour optimiser les accès numpy.

    #     Args:
    #         filters: Dictionnaire de filtres {colonne: valeur}

    #     Returns:
    #         Liste de tuples (start, end) représentant les ranges contigus

    #     Example:
    #         >>> manager.get_contiguous_ranges({"partition": "train"})
    #         [(0, 3), (5, 6), (7, 10)]  # Représente [0,1,2], [5], [7,8,9]
    #     """
    #     indices = self.get_indices(filters)
    #     if not indices:
    #         return []

    #     return np.array(indices)

    # def get_indices_from_list(self, index_list: List[int]) -> List[int]:
    #     """
    #     Filtre une liste d'index pour ne garder que ceux présents dans le DataFrame.

    #     Args:
    #         index_list: Liste d'index à filtrer

    #     Returns:
    #         Liste des index valides triée
    #     """
    #     valid_indices = self.df.filter(
    #         pl.col(self.index_col).is_in(index_list)
    #     ).select(pl.col(self.index_col)).to_series().to_list()

    #     return sorted(valid_indices)

    # def get_contiguous_ranges_from_list(self, index_list: List[int]) -> List[Tuple[int, int]]:
    #     """
    #     Convertit une liste d'index en ranges contigus.

    #     Args:
    #         index_list: Liste d'index

    #     Returns:
    #         Liste de tuples (start, end) représentant les ranges contigus
    #     """
    #     valid_indices = self.get_indices_from_list(index_list)
    #     if not valid_indices:
    #         return []

    #     return self._indices_to_ranges(valid_indices)


    # def update_by_indices(self, indices: List[int], updates: Dict[str, Any]) -> None:
    #     """
    #     Met à jour les colonnes pour des index spécifiques.

    #     Args:
    #         indices: Liste des index de samples à mettre à jour
    #         updates: Dictionnaire des colonnes à mettre à jour {colonne: valeur}

    #     Example:
    #         >>> manager.update_by_indices([0, 1, 5], {"branch": 1, "processing": "augmented"})
    #     """
    #     condition = pl.col(self.index_col).is_in(indices)

    #     for col, value in updates.items():
    #         self.df = self.df.with_columns(
    #             pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
    #         )


    # def get_stats(self, group_by: List[str] = None) -> pl.DataFrame:
    #     """
    #     Récupère des statistiques sur les samples.

    #     Args:
    #         group_by: Liste des colonnes pour le groupement

    #     Returns:
    #         DataFrame avec les statistiques
    #     """
    #     if group_by is None:
    #         group_by = ["partition", "processing"]

    #     return self.df.group_by(group_by).agg([
    #         pl.count().alias("count"),
    #         pl.col(self.index_col).min().alias("min_index"),
    #         pl.col(self.index_col).max().alias("max_index")
    #     ]).sort(group_by)



    # def _indices_to_ranges(self, indices: List[int]) -> List[Tuple[int, int]]:
    #     """
    #     Convertit une liste d'index triée en ranges contigus.

    #     Args:
    #         indices: Liste d'index triée

    #     Returns:
    #         Liste de tuples (start, end) où end est exclusif
    #     """
    #     if not indices:
    #         return []

    #     ranges = []
    #     start = indices[0]
    #     end = start

    #     for i in range(1, len(indices)):
    #         if indices[i] == end + 1:
    #             end = indices[i]
    #         else:
    #             ranges.append((start, end + 1))  # end exclusif
    #             start = indices[i]
    #             end = start

    #     ranges.append((start, end + 1))  # Ajouter le dernier range
    #     return ranges

    # def optimize_numpy_access(self, filters: Dict[str, Any],
    #                         data_array: np.ndarray) -> np.ndarray:
    #     """
    #     Optimise l'accès aux données numpy en utilisant les ranges contigus.

    #     Args:
    #         filters: Filtres pour sélectionner les samples
    #         data_array: Array numpy contenant les données (indexé par sample)

    #     Returns:
    #         Array numpy contenant les données filtrées
    #     """
    #     ranges = self.get_contiguous_ranges(filters)

    #     if not ranges:
    #         return np.array([])

    #     # Construire la liste des slices
    #     slices = []
    #     for start, end in ranges:
    #         slices.append(data_array[start:end])

    #     # Concatener tous les slices
    #     return np.concatenate(slices, axis=0)

    # def get_sample_mapping(self, filters: Dict[str, Any]) -> Dict[int, int]:
    #     """
    #     Crée un mapping entre index de sample et position dans le tableau filtré.

    #     Args:
    #         filters: Filtres pour sélectionner les samples

    #     Returns:
    #         Dictionnaire {sample_index: position_in_filtered_array}
    #     """
    #     indices = self.get_indices(filters)
    #     return {sample_idx: pos for pos, sample_idx in enumerate(indices)}

    def __repr__(self):
        return str(self.df)