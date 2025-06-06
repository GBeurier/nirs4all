from __future__ import annotations
from pathlib import Path
from typing import Sequence, Mapping, Hashable, Literal
import numpy as np
import polars as pl
import pickle

from core.blocks import Block
from core.views import TensorView
from core.store import FeatureStore
from index.frame import IndexFrame
from targets import TargetTable
from predictions import PredictionTable
from processing import ProcessingRegistry, TransformationPath


Layout = Literal["2d_concat", "2d_interlace", "3d_stack", "3d_transpose"]


class SpectraDataset:
    """
    Facade : I/O, index, features, targets, preds.
    Aucune logique métier → contrôleurs externes.
    """

    def __init__(self, index: IndexFrame | None = None, store: FeatureStore | None = None, processing: ProcessingRegistry | None = None) -> None:
        self.index: IndexFrame | None = index if index else None
        self.store = store if store else FeatureStore()
        self.processing = processing if processing else ProcessingRegistry()
        self.targets: TargetTable | None = None
        self.predictions = PredictionTable()

        # ------------------------------------------------------------------ #
    # Public mirror of store ref-count helpers
    # ------------------------------------------------------------------ #
    def inc_ref(self, processing_id: int, n: int = 1) -> None:
        """Forward to self.store.inc_ref – keeps ‘business’ code decoupled."""
        self.store.inc_ref(processing_id, n)

    def dec_ref(self, processing_id: int, n: int = 1) -> None:
        """Forward to self.store.dec_ref."""
        self.store.dec_ref(processing_id, n)

    # ------------------------------------------------------------------ #
    # Row deletion with ref-count synchronisation
    # ------------------------------------------------------------------ #
    def drop_rows(self, row_mask, *, prune: bool = False) -> None:
        """
        Physically remove rows from the index / targets / predictions.

        Ref-counts for the corresponding processing_id’s are decremented
        so that orphan blocks can be collected later.
        """
        if self.index is None:
            raise ValueError("No index to drop rows from")

        # Build a Polars expression when a list / ndarray is provided
        mask_expr = (
            pl.col("row").is_in(row_mask)
            if isinstance(row_mask, (list, tuple, np.ndarray))
            else row_mask
        )

        removed = self.index.df.filter(mask_expr)
        if removed.is_empty():
            return                                          # nothing to remove

        # Histogram of processing_id → #rows removed
        hist = (
            removed.group_by("processing")
            .agg(pl.count().alias("n"))
            .to_dict(as_series=False)
        )

        # 1 – shrink tables --------------------------------------------------
        self.index.df = self.index.df.filter(~mask_expr)
        if self.targets is not None:
            self.targets.df = self.targets.df.filter(~mask_expr)
        if self.predictions is not None:
            self.predictions.df = self.predictions.df.filter(~mask_expr)

        # 2 – ref-count bookkeeping -----------------------------------------
        for proc_id, n in zip(hist["processing"], hist["n"]):
            self.dec_ref(proc_id, n)

        # 3 – optional GC ----------------------------------------------------
        if prune:
            self.prune_unused_blocks()


    @classmethod
    def load(cls, root: Path | str, mmap: str | None = "r") -> "SpectraDataset":
        """Charge un dataset depuis le disque."""
        root_path = Path(root)
        dataset = cls()

        # Charger l'index
        index_file = root_path / "index.parquet"
        if index_file.exists():
            df = pl.read_parquet(index_file)
            dataset.index = IndexFrame(df)

        # Charger les targets
        targets_file = root_path / "targets.parquet"
        if targets_file.exists():
            df = pl.read_parquet(targets_file)
            dataset.targets = TargetTable(df)

        # Charger les predictions
        preds_file = root_path / "predictions.parquet"
        if preds_file.exists():
            df = pl.read_parquet(preds_file)
            dataset.predictions = PredictionTable(df)

        # Charger le processing registry
        dataset.processing = ProcessingRegistry(root_path / "processing")
        dataset.processing.load()

        # Charger les blocs de features
        blocks_dir = root_path / "blocks"
        if blocks_dir.exists():
            for block_file in blocks_dir.glob("*.pkl"):
                with open(block_file, "rb") as f:
                    block_data = pickle.load(f)
                    block = Block(
                        block_data["data"],
                        block_data["source_id"],
                        block_data["processing_id"]
                    )
                    dataset.store.add_block(
                        block,
                        block_data["source_id"],
                        block_data["processing_id"],
                        1  # n_users default
                    )

        return dataset

    def save(self, root: Path | str, overwrite=False) -> None:
        """Sauvegarde le dataset sur disque."""
        root_path = Path(root)

        if root_path.exists() and not overwrite:
            raise FileExistsError(f"Directory {root_path} already exists")

        root_path.mkdir(parents=True, exist_ok=overwrite)

        # Sauvegarder l'index
        if self.index:
            self.index.df.write_parquet(root_path / "index.parquet")

        # Sauvegarder les targets
        if self.targets:
            self.targets.df.write_parquet(root_path / "targets.parquet")

        # Sauvegarder les predictions
        self.predictions.df.write_parquet(root_path / "predictions.parquet")

        # Sauvegarder le processing registry
        self.processing.cache_dir = root_path / "processing"
        self.processing.save()

        # Sauvegarder les blocs
        blocks_dir = root_path / "blocks"
        blocks_dir.mkdir(exist_ok=True)

        for i, ((source_id, processing_id), block) in enumerate(self.store.iter_blocks()):
            block_data = {
                "data": block.data,
                "source_id": source_id,
                "processing_id": processing_id
            }

            with open(blocks_dir / f"block_{i}.pkl", "wb") as f:
                pickle.dump(block_data, f)

    def view(self, row_mask,
             source_ids: Sequence[int] | str = "all",
             processing_ids: Sequence[int] | str = "all") -> list[TensorView]:
        """Retourne des vues sur les features."""
        views = []

        # Résoudre "all" en listes concrètes
        if source_ids == "all":
            source_ids = list(set(k[0] for k in self.store.list_blocks()))
        if processing_ids == "all":
            processing_ids = list(set(k[1] for k in self.store.list_blocks()))

        # Convertir row_mask en sample indices pour l'accès aux données
        if isinstance(row_mask, (list, tuple)):
            if self.index is None:
                raise ValueError("No index available")
            # Filtrer par row IDs et récupérer les sample indices correspondants
            filtered_df = self.index.df.filter(pl.col("row").is_in(row_mask))
            sample_indices = filtered_df.select("sample").to_series().to_list()
        else:
            # Supposer que row_mask est déjà un predicat
            if self.index is None:
                raise ValueError("No index available")
            filtered_df = self.index.df.filter(row_mask)
            sample_indices = filtered_df.select("sample").to_series().to_list()

        for src_id in source_ids:
            for proc_id in processing_ids:
                if self.store.has_block(src_id, proc_id):
                    view = self.store.view(sample_indices, src_id, proc_id)
                    views.append(view)

        return views

    def materialize(self, row_mask,
                    source_ids: Sequence[int] | str = "all",
                    processing_ids: Sequence[int] | str = "all",
                    layout: Layout = "2d_concat",
                    dtype=np.float32) -> np.ndarray:
        """Matérialise les features selon le layout spécifié."""
        views = self.view(row_mask, source_ids, processing_ids)

        if not views:
            return np.array([], dtype=dtype)

        arrays = [view.eval().astype(dtype, copy=False) for view in views]

        if layout == "2d_concat":
            return np.hstack(arrays) if len(arrays) > 1 else arrays[0]
        elif layout == "2d_interlace":
            # Entrelacement des features
            n_s, n_f = len(arrays), arrays[0].shape[1]
            out = np.empty((arrays[0].shape[0], n_s * n_f), dtype=dtype)
            for i, a in enumerate(arrays):
                out[:, i::n_s] = a       # 1 seule passe mémoire
            return out
        elif layout == "3d_stack":
            return np.stack(arrays, axis=0).transpose(1, 0, 2)
        elif layout == "3d_transpose":
            return np.stack(arrays, axis=2)
        else:
            raise ValueError(f"Unknown layout: {layout}")

    def merge_sources(self,
                      source_ids: Sequence[int],
                      processing_ids: Sequence[int] | str = "all",
                      new_source_id: int | None = None,
                      materialize: bool = False
                      ) -> int | list[TensorView]:
        """Fusionne plusieurs sources."""
        # Résoudre "all" en liste concrète
        if processing_ids == "all":
            proc_ids = list(set(k[1] for k in self.store.list_blocks() if k[0] in source_ids))
        else:
            proc_ids = list(processing_ids)

        if materialize:
            if new_source_id is None:
                new_source_id = max(k[0] for k in self.store.list_blocks()) + 1

            for proc_id in proc_ids:
                blocks_to_merge = []
                for src_id in source_ids:
                    if self.store.has_block(src_id, proc_id):
                        blocks_to_merge.append(self.store.get_block(src_id, proc_id))

                if blocks_to_merge:
                    merged_block = self.store.concat_blocks(
                        blocks_to_merge, new_source_id, proc_id
                    )
                    self.store.add_block(merged_block, new_source_id, proc_id, 1)

            return new_source_id
        else:
            # Retourner des vues concaténées
            views = []
            for proc_id in proc_ids:
                for src_id in source_ids:
                    if self.store.has_block(src_id, proc_id):
                        view = self.store.view(slice(None), src_id, proc_id)
                        views.append(view)
            return views

    def split_source(self,
                     source_id: int,
                     splits: Sequence[slice],
                     new_source_ids: Sequence[int] | None = None,
                     processing_ids: Sequence[int] | str = "all",
                     materialize: bool = False
                     ) -> Sequence[int] | Sequence[TensorView]:
        """Découpe une source en sous-blocs."""
        # Résoudre "all" en liste concrète
        if processing_ids == "all":
            proc_ids = list(set(k[1] for k in self.store.list_blocks() if k[0] == source_id))
        else:
            proc_ids = list(processing_ids)

        if new_source_ids is None:
            max_id = max(k[0] for k in self.store.list_blocks())
            new_source_ids = list(range(max_id + 1, max_id + 1 + len(splits)))

        if materialize:
            for proc_id in proc_ids:
                if (source_id, proc_id) in self.store._blocks:
                    original_block = self.store._blocks[(source_id, proc_id)]

                    for i, col_slice in enumerate(splits):
                        split_block = self.store.slice_block(
                            original_block, col_slice, new_source_ids[i], proc_id
                        )
                        self.store.add_block(split_block, new_source_ids[i], proc_id, 1)

            return new_source_ids
        else:
            # Retourner des vues
            views = []
            for proc_id in proc_ids:
                if (source_id, proc_id) in self.store._blocks:
                    block = self.store._blocks[(source_id, proc_id)]
                    for col_slice in splits:
                        # Créer une vue avec transformation slice
                        transform = lambda x, cs=col_slice: x[:, cs]
                        view = TensorView(block, slice(None), transform)
                        views.append(view)
            return views

    def register_chain(self, base_proc: int,
                       chain: Sequence[Hashable],
                       n_features: int,
                       dtype,
                       meta_extra: Mapping | None = None) -> int:
        """Enregistre une chaîne de processing."""
        base_path = self.processing.chain(base_proc)
        new_path = base_path
        for step in chain:
            new_path = new_path.plus(step)

        return self.processing.get_or_create(new_path, n_features, dtype, meta_extra)

    def set_processing(
        self,
        row_mask,
        new_processing_id: int,
        prune: bool = False,
    ) -> None:
        """
        Update the 'processing' column in the index *and* keep ref-counts
        in sync.

        • ref––counts for the former processing_id(s) are decremented
        • ref––count for `new_processing_id` is incremented
        • optional pruning at the end
        """
        if self.index is None:
            raise ValueError("No index available")

        # --- locate affected rows & their current processing_ids ------------
        if isinstance(row_mask, (list, tuple)):
            mask = pl.col("row").is_in(row_mask)
        else:                                           # already a Polars expr
            mask = row_mask

        affected = self.index.df.filter(mask)
        if affected.height == 0:
            return                                      # nothing to do

        # number of rows that change
        n_rows = affected.height

        # histogram of current processing_ids
        counts = (
            affected
            .group_by("processing")
            .agg(pl.count().alias("n"))
            .to_dict(as_series=False)
        )
        # counts == {"processing":[...], "n":[...]}

        # --- update the index first -----------------------------------------
        self.index.update(row_mask, processing=new_processing_id)

        # --- ref-count bookkeeping ------------------------------------------
        for old_proc, n in zip(counts["processing"], counts["n"]):
            # if the row already had the new_processing_id → net 0
            if old_proc != new_processing_id:
                self.store.dec_ref(old_proc, n)

        self.store.inc_ref(new_processing_id, n_rows)

        # --- optional garbage-collection ------------------------------------
        if prune:
            self.prune_unused_blocks()


    def duplicate_rows(self, row_mask,
                       n_copies: int,
                       override: Mapping[str, object] | None = None) -> pl.DataFrame:
        """Duplique des lignes dans l'index et les tables associées."""
        if self.index is None:
            raise ValueError("No index available")

        # Dupliquer l'index
        duplicates = self.index.duplicate(row_mask, n_copies, override)
        if len(duplicates) > 0:
            # Ajouter au DataFrame principal - utiliser vertical_relaxed pour tolérer les différences de types
            self.index.df = pl.concat([self.index.df, duplicates], how="vertical_relaxed")

            proc_hist = (
                duplicates
                .group_by("processing")
                .agg(pl.count().alias("n"))
                .to_dict(as_series=False)
            )
            for proc_id, n in zip(proc_hist["processing"], proc_hist["n"]):
                self.store.inc_ref(proc_id, n)

            # Dupliquer les targets si présentes
            if self.targets is not None:
                target_duplicates = self.targets.duplicate(row_mask, n_copies)
                if len(target_duplicates) > 0:
                    self.targets.df = pl.concat([self.targets.df, target_duplicates], how="vertical_relaxed")

            # Dupliquer les predictions si présentes
            if self.predictions is not None:
                pred_duplicates = self.predictions.duplicate(row_mask, n_copies)
                if len(pred_duplicates) > 0:
                    self.predictions.df = pl.concat([self.predictions.df, pred_duplicates], how="vertical_relaxed")

        return duplicates

    def get_targets(self, row_mask,
                    columns: Sequence[str] | str = "all") -> np.ndarray | pl.DataFrame:
        """Récupère les targets."""
        if self.targets is None:
            raise ValueError("No targets available")
        return self.targets.get(row_mask, columns)

    def update_targets(self, row_mask, **kv) -> None:
        """Met à jour les targets."""
        if self.targets is None:
            raise ValueError("No targets available")
        self.targets.update(row_mask, **kv)

    def add_predictions(self,
                        row_mask,
                        y_pred,
                        model_id: str,
                        step_id: str,
                        context: Mapping | None = None) -> None:
        """Ajoute des prédictions."""
        if isinstance(row_mask, (list, tuple)):
            row_ids = row_mask
        else:
            # Extraire les row IDs depuis l'index
            if self.index is None:
                raise ValueError("No index available")
            filtered = self.index.df.filter(row_mask)
            row_ids = filtered.select("row").to_series().to_list()

        self.predictions.add_predictions(row_ids, y_pred, model_id, step_id, context)

    def preds(self, **predicates) -> pl.DataFrame:
        """Filtre les prédictions."""
        return self.predictions.filter(**predicates)

    def prune_unused_blocks(self) -> None:
        """Supprime les blocs non utilisés."""
        self.store.prune_unused()

    def get_spectra(self, row_mask, source_id: int, processing_id: int) -> np.ndarray:
        """Récupère les données spectrales pour les lignes et bloc donnés."""
        if not self.store.has_block(source_id, processing_id):
            raise KeyError(f"Block ({source_id}, {processing_id}) not found")

        view = self.store.view(row_mask, source_id, processing_id)
        return view.eval()

    def add_block(self, data: np.ndarray, processing_id: int, n_users: int = 1) -> int:
        """Ajoute un bloc de données et retourne le source_id."""
        # Générer un nouvel ID de source
        existing_blocks = self.store.list_blocks()
        if existing_blocks:
            max_source_id = max(source_id for source_id, _ in existing_blocks)
            new_source_id = max_source_id + 1
        else:
            new_source_id = 1

        block = Block(data, new_source_id, processing_id)
        self.store.add_block(block, new_source_id, processing_id, n_users)
        return new_source_id
