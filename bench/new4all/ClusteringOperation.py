"""
ClusteringOperation - Clustering operations with centroid storage
"""
import numpy as np
from typing import Dict, List, Optional, Any
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from PipelineOperation import PipelineOperation
from SpectraDataset import SpectraDataset
from PipelineContext import PipelineContext


class ClusteringOperation(PipelineOperation):
    """Operation for clustering data and storing centroids"""

    def __init__(self,
                 clustering_method: str = "kmeans",
                 n_clusters: int = 3,
                 store_centroids: bool = True,
                 evaluate_clustering: bool = True,
                 **clustering_params):
        """
        Initialize clustering operation

        Parameters:
        -----------
        clustering_method : str
            Clustering algorithm: "kmeans", "dbscan", "hierarchical"
        n_clusters : int
            Number of clusters (for applicable methods)
        store_centroids : bool
            Whether to store cluster centroids
        evaluate_clustering : bool
            Whether to compute clustering metrics
        **clustering_params : dict
            Additional parameters for clustering algorithm
        """
        super().__init__()
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        self.store_centroids = store_centroids
        self.evaluate_clustering = evaluate_clustering
        self.clustering_params = clustering_params

        # Fitted clusterer
        self.clusterer = None
        self.cluster_labels = None
        self.centroids = None
        self.clustering_metrics = {}

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute clustering operation"""
        if not self.can_execute(dataset, context):
            raise ValueError("Cannot execute clustering - no data available")

        # Get data for clustering
        X_data = self.get_clustering_data(dataset, context)

        # Create and fit clusterer
        self.clusterer = self.create_clusterer()

        # Fit clustering
        if hasattr(self.clusterer, 'fit_predict'):
            self.cluster_labels = self.clusterer.fit_predict(X_data)
        else:
            self.clusterer.fit(X_data)
            self.cluster_labels = self.clusterer.labels_

        # Store centroids if requested
        if self.store_centroids:
            self.centroids = self.compute_centroids(X_data, self.cluster_labels)

        # Evaluate clustering if requested
        if self.evaluate_clustering:
            self.clustering_metrics = self.evaluate_clusters(X_data, self.cluster_labels)

        # Store results in context
        context.clustering_results = {
            'labels': self.cluster_labels,
            'centroids': self.centroids,
            'metrics': self.clustering_metrics,
            'clusterer': self.clusterer
        }

        # Add cluster labels as new targets if possible
        self.add_cluster_targets(dataset)        # Store centroids in dataset and set selection to centroids
        if self.store_centroids and self.centroids is not None:
            self._store_centroids_in_dataset(dataset, context)
            # Set pipeline context to work with centroids
            old_filters = context.push_filters(processing="centroids")
            context.centroid_filters = old_filters  # Store for unclustering

        print(f"Clustering completed: {len(np.unique(self.cluster_labels))} clusters found")
        if self.clustering_metrics:
            for metric, value in self.clustering_metrics.items():
                print(f"  {metric}: {value:.4f}")

    def _store_centroids_in_dataset(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Store cluster centroids as new samples in the dataset"""
        try:
            # Add centroids as new samples with special processing tag
            centroid_sample_ids = dataset.add_data(
                features=[self.centroids],  # Centroids as single source
                targets=None,  # Centroids don't have targets initially
                partition="centroids",
                processing="centroids",
                group=0,
                branch=context.current_branch
            )

            # Mark these samples as centroids in indices
            import polars as pl
            mask = dataset.indices['sample'].is_in(centroid_sample_ids)
            dataset.indices = dataset.indices.with_columns([
                pl.when(mask)
                .then(pl.lit("centroids"))
                .otherwise(pl.col('processing'))
                .alias('processing')
            ])

            # Store centroid mapping for unclustering
            context.centroid_mapping = {
                'centroid_sample_ids': centroid_sample_ids,
                'cluster_labels': self.cluster_labels,
                'original_sample_count': len(self.cluster_labels)
            }

            print(f"Stored {len(centroid_sample_ids)} centroids in dataset")

        except Exception as e:
            print(f"Could not store centroids in dataset: {e}")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if clustering can be executed"""
        return len(dataset) > 0

    def get_name(self) -> str:
        """Get operation name"""
        return f"ClusteringOperation({self.clustering_method})"

    def get_clustering_data(self, dataset: SpectraDataset, context: PipelineContext) -> np.ndarray:
        """Get data for clustering"""
        # For clustering, we need concatenated features like models do
        # Get all samples
        all_indices = np.arange(len(dataset))
        X = dataset.get_features(all_indices, concatenate=True)
        return X

    def create_clusterer(self):
        """Create clustering algorithm instance"""
        if self.clustering_method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                **self.clustering_params
            )
        elif self.clustering_method == "dbscan":
            return DBSCAN(
                **self.clustering_params
            )
        elif self.clustering_method == "hierarchical":
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
                **self.clustering_params
            )
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

    def compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids"""
        unique_labels = np.unique(labels)
        # Filter out noise points (label -1 in DBSCAN)
        unique_labels = unique_labels[unique_labels >= 0]

        centroids = np.zeros((len(unique_labels), X.shape[1]))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            centroids[i] = np.mean(X[mask], axis=0)

        return centroids

    def evaluate_clusters(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering quality"""
        metrics = {}

        # Only evaluate if we have more than 1 cluster
        n_clusters = len(np.unique(labels[labels >= 0]))
        if n_clusters > 1:
            try:
                # Silhouette score
                metrics['silhouette_score'] = silhouette_score(X, labels)
            except Exception as e:
                print(f"Could not compute silhouette score: {e}")

            # Inertia for K-means
            if self.clustering_method == "kmeans" and hasattr(self.clusterer, 'inertia_'):
                metrics['inertia'] = self.clusterer.inertia_

        # Number of clusters and noise points
        metrics['n_clusters'] = n_clusters
        metrics['n_noise_points'] = np.sum(labels == -1)
        metrics['n_samples'] = len(labels)

        return metrics

    def add_cluster_targets(self, dataset: SpectraDataset) -> None:
        """Add cluster labels as new target variable"""
        if hasattr(dataset, 'target_manager') and dataset.target_manager is not None:
            try:
                # Get all sample IDs from the dataset
                all_sample_ids = list(range(len(dataset)))

                # Add cluster labels as classification targets
                dataset.target_manager.add_targets(
                    all_sample_ids,
                    self.cluster_labels
                )

                print(f"Added cluster labels as targets for {len(all_sample_ids)} samples")

            except Exception as e:
                print(f"Could not add cluster targets: {e}")

    def predict_clusters(self, X_new: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data"""
        if self.clusterer is None:
            raise ValueError("Clusterer not fitted yet")

        if hasattr(self.clusterer, 'predict'):
            return self.clusterer.predict(X_new)
        else:
            # For methods without predict (like DBSCAN), assign to nearest centroid
            if self.centroids is None:
                raise ValueError("No centroids available for prediction")            # Compute distances to centroids
            distances = np.linalg.norm(
                X_new[:, np.newaxis, :] - self.centroids[np.newaxis, :, :],
                axis=2
            )
            return np.argmin(distances, axis=1)

    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary of clustering results"""
        if self.cluster_labels is None:
            return {}

        unique_labels = np.unique(self.cluster_labels)
        cluster_sizes = {int(label): np.sum(self.cluster_labels == label)
                         for label in unique_labels}

        summary = {
            'n_clusters': len(unique_labels[unique_labels >= 0]),
            'cluster_sizes': cluster_sizes,
            'metrics': self.clustering_metrics,
            'method': self.clustering_method
        }

        if self.centroids is not None:
            summary['centroid_shape'] = self.centroids.shape

        return summary


class ClusteringStrategy:
    """Strategy for different clustering approaches"""

    @classmethod
    def hierarchical_clustering(cls, max_clusters: int = 10) -> List[ClusteringOperation]:
        """Create hierarchical clustering with different cluster numbers"""
        operations = []

        for n_clusters in range(2, max_clusters + 1):
            operations.append(ClusteringOperation(
                clustering_method="hierarchical",
                n_clusters=n_clusters,
                store_centroids=True,
                evaluate_clustering=True
            ))

        return operations

    @classmethod
    def density_clustering(cls) -> ClusteringOperation:
        """Create DBSCAN clustering operation"""
        return ClusteringOperation(
            clustering_method="dbscan",
            store_centroids=True,
            evaluate_clustering=True,
            eps=0.3,
            min_samples=5
        )

    @classmethod
    def optimal_kmeans(cls, max_clusters: int = 10) -> List[ClusteringOperation]:
        """Create K-means clustering with different K values for elbow method"""
        operations = []

        for k in range(2, max_clusters + 1):
            operations.append(ClusteringOperation(
                clustering_method="kmeans",
                n_clusters=k,
                store_centroids=True,
                evaluate_clustering=True
            ))

        return operations


class UnclusterOperation(PipelineOperation):
    """Operation to uncluster data and restore full sample selection"""

    def __init__(self):
        """Initialize uncluster operation"""
        super().__init__()

    def execute(self, dataset: SpectraDataset, context: PipelineContext) -> None:
        """Execute unclustering operation"""
        if not self.can_execute(dataset, context):
            print("Cannot execute unclustering: no clustering results found")
            return

        # Check if we have centroid mapping
        if not hasattr(context, 'centroid_mapping') or context.centroid_mapping is None:
            print("Warning: No centroid mapping found, cannot properly uncluster")
            # Just restore filters
            if hasattr(context, 'centroid_filters'):
                context.pop_filters(context.centroid_filters)
                delattr(context, 'centroid_filters')
            return

        # Get clustering results from context
        clustering_results = getattr(context, 'clustering_results', {})
        cluster_labels = clustering_results.get('labels')
        centroid_mapping = context.centroid_mapping

        if cluster_labels is None:
            print("Warning: No cluster labels found")
            return

        # Assign samples to cluster groups based on their cluster labels
        self._assign_samples_to_groups(dataset, cluster_labels)

        # Remove centroids from dataset (optional - depends on use case)
        # self._remove_centroids_from_dataset(dataset, centroid_mapping)

        # Restore original filters (remove centroid selection)
        if hasattr(context, 'centroid_filters'):
            context.pop_filters(context.centroid_filters)
            delattr(context, 'centroid_filters')

        # Clear centroid mapping
        context.centroid_mapping = None

        print("Unclustering completed: restored full sample selection")

    def _assign_samples_to_groups(self, dataset: SpectraDataset, cluster_labels: np.ndarray) -> None:
        """Assign samples to groups based on cluster labels"""
        try:
            import polars as pl

            # Get original samples (not centroids)
            original_mask = dataset.indices['processing'] != "centroids"
            original_indices = dataset.indices.filter(original_mask)

            if len(original_indices) != len(cluster_labels):
                print(f"Warning: Sample count mismatch. Expected {len(cluster_labels)}, got {len(original_indices)}")
                return

            # Create mapping of sample_id to cluster group
            sample_ids = original_indices['sample'].to_list()

            # Update group column based on cluster labels
            for sample_id, cluster_group in zip(sample_ids, cluster_labels):
                dataset.indices = dataset.indices.with_columns([
                    pl.when(pl.col('sample') == sample_id)
                    .then(int(cluster_group))
                    .otherwise(pl.col('group'))
                    .alias('group')
                ])

            print(f"Assigned {len(sample_ids)} samples to {len(np.unique(cluster_labels))} cluster groups")

        except Exception as e:
            print(f"Error assigning samples to groups: {e}")

    def _remove_centroids_from_dataset(self, dataset: SpectraDataset, centroid_mapping: dict) -> None:
        """Remove centroid samples from dataset"""
        try:
            import polars as pl

            centroid_sample_ids = centroid_mapping.get('centroid_sample_ids', [])

            # Remove centroids from indices
            mask = ~dataset.indices['sample'].is_in(centroid_sample_ids)
            dataset.indices = dataset.indices.filter(mask)

            # Note: In a full implementation, we'd also need to remove from features
            # This would require more complex feature management in SpectraDataset

            print(f"Removed {len(centroid_sample_ids)} centroid samples")

        except Exception as e:
            print(f"Error removing centroids: {e}")

    def can_execute(self, dataset: SpectraDataset, context: PipelineContext) -> bool:
        """Check if unclustering can be executed"""
        return (hasattr(context, 'clustering_results') and
                context.clustering_results is not None)

    def get_name(self) -> str:
        """Get operation name"""
        return "UnclusterOperation"
