import polars as pl
from typing import Dict, List, Tuple, Union, Any, Optional
import numpy as np


class SampleIndexManager:
    """
    Gestionnaire d'index de samples pour analyses ML/DL avec optimisation
    des accès contigus et gestion des filtres.
    """

    def __init__(self, df: pl.DataFrame, index_col: str = "sample"):
        """
        Initialise le gestionnaire avec un DataFrame Polars.

        Args:
            df: DataFrame Polars contenant les métadonnées des samples
            index_col: Nom de la colonne contenant les index des samples
        """
        self.df = df
        self.index_col = index_col
        self._ensure_sorted()

    def _ensure_sorted(self):
        """S'assure que le DataFrame est trié par l'index des samples."""
        # Vérifier si le DataFrame est déjà trié
        indices = self.df.select(pl.col(self.index_col)).to_series().to_list()
        is_sorted = all(indices[i] <= indices[i + 1] for i in range(len(indices) - 1))

        if not is_sorted:
            self.df = self.df.sort(self.index_col)

    def get_indices(self, filters: Dict[str, Any]) -> List[int]:
        """
        Récupère la liste des index de samples selon les filtres spécifiés.

        Args:
            filters: Dictionnaire de filtres {colonne: valeur}

        Returns:
            Liste triée des index de samples correspondants

        Example:
            >>> manager.get_indices({"partition": "train", "branch": 0})
            [0, 1, 2, 5, 7, 8, 9]
        """
        filtered_df = self._apply_filters(filters)
        return filtered_df.select(pl.col(self.index_col)).to_series().to_list()

    def get_contiguous_ranges(self, filters: Dict[str, Any]) -> List[Tuple[int, int]]:
        """
        Récupère les ranges contigus d'index pour optimiser les accès numpy.

        Args:
            filters: Dictionnaire de filtres {colonne: valeur}

        Returns:
            Liste de tuples (start, end) représentant les ranges contigus

        Example:
            >>> manager.get_contiguous_ranges({"partition": "train"})
            [(0, 3), (5, 6), (7, 10)]  # Représente [0,1,2], [5], [7,8,9]
        """
        indices = self.get_indices(filters)
        if not indices:
            return []

        return self._indices_to_ranges(indices)

    def get_indices_from_list(self, index_list: List[int]) -> List[int]:
        """
        Filtre une liste d'index pour ne garder que ceux présents dans le DataFrame.

        Args:
            index_list: Liste d'index à filtrer

        Returns:
            Liste des index valides triée
        """
        valid_indices = self.df.filter(
            pl.col(self.index_col).is_in(index_list)
        ).select(pl.col(self.index_col)).to_series().to_list()

        return sorted(valid_indices)

    def get_contiguous_ranges_from_list(self, index_list: List[int]) -> List[Tuple[int, int]]:
        """
        Convertit une liste d'index en ranges contigus.

        Args:
            index_list: Liste d'index

        Returns:
            Liste de tuples (start, end) représentant les ranges contigus
        """
        valid_indices = self.get_indices_from_list(index_list)
        if not valid_indices:
            return []

        return self._indices_to_ranges(valid_indices)

    def update_by_filter(self, filters: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """
        Met à jour les colonnes selon un filtre.

        Args:
            filters: Dictionnaire de filtres pour sélectionner les lignes
            updates: Dictionnaire des colonnes à mettre à jour {colonne: valeur}

        Example:
            >>> manager.update_by_filter(
            ...     {"partition": "train", "processing": "raw"},
            ...     {"processing": "normalized", "group": 1}
            ... )
        """
        # Construire la condition de filtre
        condition = self._build_filter_condition(filters)

        # Appliquer les mises à jour
        for col, value in updates.items():
            self.df = self.df.with_columns(
                pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

    def update_by_indices(self, indices: List[int], updates: Dict[str, Any]) -> None:
        """
        Met à jour les colonnes pour des index spécifiques.

        Args:
            indices: Liste des index de samples à mettre à jour
            updates: Dictionnaire des colonnes à mettre à jour {colonne: valeur}

        Example:
            >>> manager.update_by_indices([0, 1, 5], {"branch": 1, "processing": "augmented"})
        """
        condition = pl.col(self.index_col).is_in(indices)

        for col, value in updates.items():
            self.df = self.df.with_columns(
                pl.when(condition).then(pl.lit(value)).otherwise(pl.col(col)).alias(col)
            )

    def add_samples(self, new_df: pl.DataFrame) -> None:
        """
        Ajoute de nouveaux samples au DataFrame.

        Args:
            new_df: DataFrame contenant les nouveaux samples
        """
        self.df = pl.concat([self.df, new_df], how="vertical")
        self._ensure_sorted()

    def get_stats(self, group_by: List[str] = None) -> pl.DataFrame:
        """
        Récupère des statistiques sur les samples.

        Args:
            group_by: Liste des colonnes pour le groupement

        Returns:
            DataFrame avec les statistiques
        """
        if group_by is None:
            group_by = ["partition", "processing"]

        return self.df.group_by(group_by).agg([
            pl.count().alias("count"),
            pl.col(self.index_col).min().alias("min_index"),
            pl.col(self.index_col).max().alias("max_index")
        ]).sort(group_by)

    def _apply_filters(self, filters: Dict[str, Any]) -> pl.DataFrame:
        """Applique les filtres au DataFrame."""
        condition = self._build_filter_condition(filters)
        return self.df.filter(condition)

    def _build_filter_condition(self, filters: Dict[str, Any]) -> pl.Expr:
        """Construit une condition Polars à partir des filtres."""
        conditions = []
        for col, value in filters.items():
            if isinstance(value, list):
                conditions.append(pl.col(col).is_in(value))
            else:
                conditions.append(pl.col(col) == value)

        # Combine toutes les conditions avec AND
        condition = conditions[0]
        for cond in conditions[1:]:
            condition = condition & cond

        return condition

    def _indices_to_ranges(self, indices: List[int]) -> List[Tuple[int, int]]:
        """
        Convertit une liste d'index triée en ranges contigus.

        Args:
            indices: Liste d'index triée

        Returns:
            Liste de tuples (start, end) où end est exclusif
        """
        if not indices:
            return []

        ranges = []
        start = indices[0]
        end = start

        for i in range(1, len(indices)):
            if indices[i] == end + 1:
                end = indices[i]
            else:
                ranges.append((start, end + 1))  # end exclusif
                start = indices[i]
                end = start

        ranges.append((start, end + 1))  # Ajouter le dernier range
        return ranges

    def optimize_numpy_access(self, filters: Dict[str, Any],
                            data_array: np.ndarray) -> np.ndarray:
        """
        Optimise l'accès aux données numpy en utilisant les ranges contigus.

        Args:
            filters: Filtres pour sélectionner les samples
            data_array: Array numpy contenant les données (indexé par sample)

        Returns:
            Array numpy contenant les données filtrées
        """
        ranges = self.get_contiguous_ranges(filters)

        if not ranges:
            return np.array([])

        # Construire la liste des slices
        slices = []
        for start, end in ranges:
            slices.append(data_array[start:end])

        # Concatener tous les slices
        return np.concatenate(slices, axis=0)

    def get_sample_mapping(self, filters: Dict[str, Any]) -> Dict[int, int]:
        """
        Crée un mapping entre index de sample et position dans le tableau filtré.

        Args:
            filters: Filtres pour sélectionner les samples

        Returns:
            Dictionnaire {sample_index: position_in_filtered_array}
        """
        indices = self.get_indices(filters)
        return {sample_idx: pos for pos, sample_idx in enumerate(indices)}
