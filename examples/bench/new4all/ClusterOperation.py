import polars as pl
from sklearn.base import ClusterMixin
from .PipelineOperation import PipelineOperation
from .DatasetView import DatasetView
from typing import Optional, Dict, Any
from .SpectraDataset import SpectraDataset
from .PipelineContext import PipelineContext

class ClusterOperation(PipelineOperation):
    """Handles clustering efficiently."""

    def __init__(self, clusterer: ClusterMixin, fit_on: str = "train"):
        self.clusterer = clusterer
        self.fit_on = fit_on

    def execute(self, dataset: SpectraDataset, context: 'PipelineContext'):
        train_view = dataset.select(partition=self.fit_on, **context.current_filters)
        if len(train_view) == 0:
            raise ValueError(f"No {self.fit_on} data found for clustering")

        X_train = train_view.get_features()
        cluster_labels = self.clusterer.fit_predict(X_train)

        # Update group labels
        current_indices = dataset.indices
        sample_mask = pl.col("sample").is_in(train_view.sample_ids)

        # Create mapping of sample_id to cluster label
        sample_to_cluster = dict(zip(train_view.sample_ids, cluster_labels))

        # Update indices with new group labels
        dataset.indices = current_indices.with_columns(
            pl.col("sample").map_elements(
                lambda s: sample_to_cluster.get(s, pl.col("group")),
                return_dtype=pl.Int32
            ).alias("group")
        )

    def get_name(self) -> str:
        return f"Cluster({self.clusterer.__class__.__name__})"