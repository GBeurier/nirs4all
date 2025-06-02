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
        self.add_cluster_targets(dataset)
        print(f"Clustering completed: {len(np.unique(self.cluster_labels))} clusters found")
        if self.clustering_metrics:
            print(f"Clustering metrics: {self.clustering_metrics}")

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
                random_state=42,
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
                raise ValueError("No centroids available for prediction")

            # Compute distances to centroids
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
