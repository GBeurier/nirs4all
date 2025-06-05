"""
ClusterMixin Operation - Wraps sklearn cluster algorithms for pipeline execution

This operation:
1. Fits clustering algorithm on train partition data
2. Assigns cluster labels to all samples
3. Updates group index in dataset
4. Optionally designates centroids for each cluster
"""
import numpy as np
import hashlib
from typing import Optional, List, Union, Dict
from sklearn.base import ClusterMixin, clone

from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class OperationCluster(PipelineOperation):
    """Pipeline operation that wraps sklearn ClusterMixin objects"""

    def __init__(self,
                 clusterer: ClusterMixin,
                 fit_partition: str = "train",
                 cluster_partitions: Optional[List[str]] = None,
                 designate_centroids: bool = True,
                 centroid_strategy: str = "nearest",
                 operation_name: Optional[str] = None):
        """
        Initialize clustering operation

        Args:
            clusterer: sklearn ClusterMixin object to wrap
            fit_partition: partition to fit the clusterer on (default: "train")
            cluster_partitions: partitions to assign clusters to (None = all partitions)
            designate_centroids: whether to designate centroid samples
            centroid_strategy: how to choose centroids ("nearest", "random", "new")
            operation_name: custom name for this operation
        """
        self.clusterer = clusterer
        self.fit_partition = fit_partition
        self.cluster_partitions = cluster_partitions
        self.designate_centroids = designate_centroids
        self.centroid_strategy = centroid_strategy
        self.operation_name = operation_name
        self.fitted_clusterer = None
        self.is_fitted = False
        self.cluster_centers_ = None
        self.centroid_samples = {}  # {cluster_id: sample_id}

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute clustering: fit on train, assign clusters to specified partitions"""
        print(f"🔄 Executing {self.get_name()}")

        # Get fitting data from specified partition
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        if len(fit_view) == 0:
            raise ValueError(f"No data found in partition '{self.fit_partition}' for fitting")
          # Get features (concatenate all sources for clustering)
        X_fit = fit_view.get_features(representation="2d_concatenated")

        print(f"  📊 Fitting on {len(fit_view)} samples from '{self.fit_partition}' partition")
        print(f"  🔧 Feature shape: {X_fit.shape}")

        # Fit clusterer
        self.fitted_clusterer = clone(self.clusterer)
        cluster_labels_fit = self.fitted_clusterer.fit_predict(X_fit)
        n_clusters = len(np.unique(cluster_labels_fit))

        print(f"  ✅ Clusterer fitted: {n_clusters} clusters found")

        # Store cluster centers if available
        if hasattr(self.fitted_clusterer, 'cluster_centers_'):
            self.cluster_centers_ = self.fitted_clusterer.cluster_centers_

        self.is_fitted = True

        # Determine partitions to assign clusters to
        partitions_to_cluster = self.cluster_partitions or dataset.get_partition_names()

        # Assign clusters to each partition
        for partition in partitions_to_cluster:
            partition_view = dataset.select(partition=partition, **context.current_filters)
            if len(partition_view) == 0:
                print(f"  ⚠️ Skipping partition '{partition}' - no data found")
                continue

            print(f"  🔄 Assigning clusters to partition '{partition}': {len(partition_view)} samples")

            # Get features and predict clusters
            X_partition = partition_view.get_features(representation="2d_concatenated")
            cluster_labels = self.fitted_clusterer.predict(X_partition)

            # Update group index in dataset
            sample_ids = partition_view.sample_ids
            for sample_id, cluster_label in zip(sample_ids, cluster_labels):
                dataset.update_groups([sample_id], int(cluster_label))

            print(f"    ✅ Assigned {n_clusters} clusters to {len(sample_ids)} samples")

        # Designate centroids if requested
        if self.designate_centroids:
            self._designate_centroids(dataset, context)

    def _designate_centroids(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Designate centroid samples for each cluster"""
        print(f"  🎯 Designating centroids using strategy: {self.centroid_strategy}")

        # Get all samples with cluster assignments
        all_samples = dataset.select(**context.current_filters)
        if len(all_samples) == 0:
            return

        # Group samples by cluster
        cluster_groups = {}
        for idx, (sample_id, group_id) in enumerate(zip(all_samples.sample_ids, all_samples.group_ids)):
            if group_id not in cluster_groups:
                cluster_groups[group_id] = []
            cluster_groups[group_id].append((sample_id, idx))

        # Designate one centroid per cluster
        for cluster_id, samples in cluster_groups.items():
            if not samples:
                continue

            if self.centroid_strategy == "random":
                # Random sample from cluster
                centroid_sample_id, _ = samples[np.random.randint(len(samples))]

            elif self.centroid_strategy == "nearest" and self.cluster_centers_ is not None:
                # Sample nearest to cluster center
                sample_indices = [idx for _, idx in samples]
                X_cluster = all_samples.get_features(representation="2d_concatenated")[sample_indices]

                # Find nearest to cluster center
                center = self.cluster_centers_[cluster_id]
                distances = np.linalg.norm(X_cluster - center, axis=1)
                nearest_idx = np.argmin(distances)
                centroid_sample_id, _ = samples[nearest_idx]

            else:
                # Default: first sample in cluster
                centroid_sample_id, _ = samples[0]

            # Store centroid designation
            self.centroid_samples[cluster_id] = centroid_sample_id

            # You could add a "centroid" index to the dataset here if needed
            # For now, we'll store it in the operation for later use

        print(f"    ✅ Designated {len(self.centroid_samples)} centroids: {list(self.centroid_samples.values())}")

    def get_centroids(self) -> Dict[int, str]:
        """Get mapping of cluster_id -> centroid_sample_id"""
        return self.centroid_samples.copy()

    def get_name(self) -> str:
        """Get operation name"""
        if self.operation_name:
            return self.operation_name
        return f"Cluster({self.clusterer.__class__.__name__})"

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if clustering can be executed"""
        fit_view = dataset.select(partition=self.fit_partition, **context.current_filters)
        return len(fit_view) > 1  # Need at least 2 samples to cluster
