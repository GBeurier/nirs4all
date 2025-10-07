# pip install numpy pandas scipy scikit-learn matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import subspace_angles
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist

class PreprocPCAEvaluator:
    def __init__(self, r_components=10, knn=10):
        self.r = r_components
        self.knn = knn
        self.df_ = None
        self.cache_ = {}
        self.raw_pcas_ = {}  # Store raw PCA results for visualization
        self.cross_dataset_df_ = None  # Store inter-dataset distance metrics
        self.pp_pcas_ = {}  # Store preprocessed PCA results: {(dataset, preproc): (Z, U, evr)}

    # ---------------- utils ----------------
    @staticmethod
    def _center(X): return X - X.mean(0, keepdims=True)

    def _pca(self, X, r):
        Xc = self._center(X)
        r = min(r, Xc.shape[1])
        p = PCA(n_components=r, random_state=0).fit(Xc)
        Z, U = p.transform(Xc), p.components_.T
        evr = float(p.explained_variance_ratio_.sum())
        return Z, U, evr

    @staticmethod
    def _grassmann(U, V):
        th = subspace_angles(U, V)
        return float(np.sqrt((th**2).sum()))

    def _cka(self, X, Y):
        Xc, Yc = self._center(X), self._center(Y)
        hsic = np.linalg.norm(Xc.T @ Yc, 'fro')**2
        den  = np.linalg.norm(Xc.T @ Xc, 'fro') * np.linalg.norm(Yc.T @ Yc, 'fro')
        return float(hsic/den) if den>0 else np.nan

    def _rv(self, X, Y):
        Xc, Yc = self._center(X), self._center(Y)
        A, B = Xc @ Xc.T, Yc @ Yc.T
        num = np.trace(A @ B)
        den = np.sqrt(np.trace(A @ A) * np.trace(B @ B))
        return float(num/den) if den>0 else np.nan

    @staticmethod
    def _procrustes(Z1, Z2):
        _, _, d = procrustes(Z1, Z2)
        return float(d)

    def _trust(self, Zref, Znew, k):
        n = Zref.shape[0]
        k = max(2, min(k, n-2))
        nnr = NearestNeighbors(n_neighbors=n-1).fit(Zref).kneighbors(return_distance=False)
        nnn = NearestNeighbors(n_neighbors=n-1).fit(Znew).kneighbors(return_distance=False)
        # ranks[i, j] = rank of sample j in the neighborhood of sample i in reference space
        ranks = np.zeros((n, n), dtype=int)
        for i in range(n):
            ranks[i, nnr[i]] = np.arange(n-1)
        s = 0.0
        for i in range(n):
            Ui = set(nnn[i, 1:1+k])
            Ki = set(nnr[i, 1:1+k])
            for v in Ui - Ki:
                s += (ranks[i, v] - (k-1))
        Z = n*k*(2*n - 3*k - 1)/2
        return 1.0 - (2.0/Z)*s if Z>0 else np.nan

    # ---------------- core API ----------------
    def fit(self, raw_data: dict[str, np.ndarray], pp_data: dict[str, dict[str, np.ndarray]]):
        """
        raw_data: {"dataset": X_raw_(n,m), ...}
        pp_data:  Can be either:
                  - {"pp_name": {"dataset": X_pp_(n,p), ...}, ...} OR
                  - {"dataset": {"pp_name": X_pp_(n,p), ...}, ...}
                  (will automatically detect and pivot if needed)

        Assumes rows (samples) are aligned within each dataset across raw and pp.
        """
        # Auto-detect structure and pivot if needed
        pp_data = self._ensure_pp_structure(pp_data, raw_data)

        rows = []
        self.cache_.clear()
        self.raw_pcas_.clear()

        # precompute raw PCA per dataset
        for dname, Xr in raw_data.items():
            Zr, Ur, evr_r = self._pca(np.asarray(Xr), self.r)
            self.raw_pcas_[dname] = (Zr, Ur, evr_r)

        # iterate preprocessings
        for pp_name, dmap in pp_data.items():
            for dname, Xp in dmap.items():
                if dname not in self.raw_pcas_:
                    continue  # skip if no matching raw dataset
                Zr, Ur, evr_r = self.raw_pcas_[dname]
                Xp = np.asarray(Xp)
                if Xp.shape[0] != Zr.shape[0]:
                    raise ValueError(f"n_samples mismatch for dataset '{dname}' in '{pp_name}'")
                Zp, Up, evr_p = self._pca(Xp, min(self.r, Zr.shape[1]))

                r_use = min(Ur.shape[1], Up.shape[1], Zr.shape[1], Zp.shape[1])
                Ur_, Up_ = Ur[:, :r_use], Up[:, :r_use]
                Zr_, Zp_ = Zr[:, :r_use], Zp[:, :r_use]

                # Grassmann distance only makes sense when feature spaces have same dimensionality
                # If preprocessing changes feature dimension, we skip it (set to NaN)
                grassmann_dist = np.nan
                if Ur_.shape[0] == Up_.shape[0]:  # same number of features
                    grassmann_dist = self._grassmann(Ur_, Up_)

                rows.append({
                    "dataset": dname,
                    "preproc": pp_name,
                    "r_used": r_use,
                    "evr_raw": evr_r,
                    "evr_pre": evr_p,
                    "grassmann": grassmann_dist,
                    "cka": self._cka(Zr_, Zp_),
                    "rv": self._rv(Zr_, Zp_),
                    "procrustes": self._procrustes(Zr_, Zp_),
                    "trustworthiness": self._trust(Zr_, Zp_, k=self.knn),
                })
                # cache full PCA scores for visualization
                self.cache_[(dname, pp_name)] = (Zr_, Zp_)
                # Store preprocessed PCA for cross-dataset analysis
                self.pp_pcas_[(dname, pp_name)] = (Zp, Up, evr_p)

        self.df_ = pd.DataFrame(rows)

        # Compute inter-dataset distances
        self._compute_cross_dataset_distances(raw_data, pp_data)

        return self

    def _ensure_pp_structure(self, pp_data, raw_data):
        """
        Ensure pp_data has structure {preproc: {dataset: X}}.
        If it's {dataset: {preproc: X}}, pivot it.
        """
        if not pp_data:
            return pp_data

        # Check first key to determine structure
        first_key = next(iter(pp_data.keys()))
        first_val = pp_data[first_key]

        # If first value is a dict and its keys match raw_data keys, it's {preproc: {dataset: X}}
        if isinstance(first_val, dict):
            first_inner_key = next(iter(first_val.keys()))
            if first_inner_key in raw_data:
                # Already correct structure {preproc: {dataset: X}}
                return pp_data
            elif first_key in raw_data:
                # Wrong structure {dataset: {preproc: X}}, need to pivot
                pivoted = {}
                for dataset_name, preproc_map in pp_data.items():
                    for preproc_name, X in preproc_map.items():
                        if preproc_name not in pivoted:
                            pivoted[preproc_name] = {}
                        pivoted[preproc_name][dataset_name] = X
                return pivoted

        return pp_data

    # ---------------- cross-dataset analysis ----------------
    def _compute_cross_dataset_distances(self, raw_data, pp_data):
        """
        Compute pairwise distances between different datasets in PCA space,
        both for raw and preprocessed data. This helps assess if preprocessing
        brings datasets (e.g., from different machines) closer together.
        """
        dataset_names = list(raw_data.keys())
        if len(dataset_names) < 2:
            self.cross_dataset_df_ = pd.DataFrame()
            return

        preproc_names = list(pp_data.keys())
        rows = []

        # Compute distances for all dataset pairs
        for i, ds1 in enumerate(dataset_names):
            for j in range(i + 1, len(dataset_names)):
                ds2 = dataset_names[j]

                # Raw data distances
                if ds1 in self.raw_pcas_ and ds2 in self.raw_pcas_:
                    Z1_raw, U1_raw, _ = self.raw_pcas_[ds1]
                    Z2_raw, U2_raw, _ = self.raw_pcas_[ds2]

                    # Use minimum components available
                    r_use = min(Z1_raw.shape[1], Z2_raw.shape[1])
                    Z1_raw = Z1_raw[:, :r_use]
                    Z2_raw = Z2_raw[:, :r_use]

                    # Compute centroid distance (how far apart are the dataset centers?)
                    centroid_dist_raw = np.linalg.norm(Z1_raw.mean(axis=0) - Z2_raw.mean(axis=0))

                    # Compute spread overlap (how much do distributions overlap?)
                    # Using Wasserstein/Earth Mover's distance approximation
                    spread_dist_raw = self._compute_spread_distance(Z1_raw, Z2_raw)

                    # Subspace angle (how aligned are the PCA subspaces?)
                    r_subspace = min(U1_raw.shape[1], U2_raw.shape[1], U1_raw.shape[0], U2_raw.shape[0])
                    if r_subspace > 0 and U1_raw.shape[0] == U2_raw.shape[0]:
                        subspace_angle_raw = self._grassmann(U1_raw[:, :r_subspace], U2_raw[:, :r_subspace])
                    else:
                        subspace_angle_raw = np.nan

                    # For each preprocessing method
                    for pp_name in preproc_names:
                        if (ds1, pp_name) in self.pp_pcas_ and (ds2, pp_name) in self.pp_pcas_:
                            Z1_pp, U1_pp, _ = self.pp_pcas_[(ds1, pp_name)]
                            Z2_pp, U2_pp, _ = self.pp_pcas_[(ds2, pp_name)]

                            r_use_pp = min(Z1_pp.shape[1], Z2_pp.shape[1])
                            Z1_pp = Z1_pp[:, :r_use_pp]
                            Z2_pp = Z2_pp[:, :r_use_pp]

                            centroid_dist_pp = np.linalg.norm(Z1_pp.mean(axis=0) - Z2_pp.mean(axis=0))
                            spread_dist_pp = self._compute_spread_distance(Z1_pp, Z2_pp)

                            r_subspace_pp = min(U1_pp.shape[1], U2_pp.shape[1], U1_pp.shape[0], U2_pp.shape[0])
                            if r_subspace_pp > 0 and U1_pp.shape[0] == U2_pp.shape[0]:
                                subspace_angle_pp = self._grassmann(U1_pp[:, :r_subspace_pp], U2_pp[:, :r_subspace_pp])
                            else:
                                subspace_angle_pp = np.nan

                            # Compute improvement (negative = datasets got closer)
                            centroid_improvement = (centroid_dist_raw - centroid_dist_pp) / (centroid_dist_raw + 1e-10)
                            spread_improvement = (spread_dist_raw - spread_dist_pp) / (spread_dist_raw + 1e-10)

                            rows.append({
                                'dataset_1': ds1,
                                'dataset_2': ds2,
                                'preproc': pp_name,
                                'centroid_dist_raw': centroid_dist_raw,
                                'centroid_dist_pp': centroid_dist_pp,
                                'centroid_improvement': centroid_improvement,
                                'spread_dist_raw': spread_dist_raw,
                                'spread_dist_pp': spread_dist_pp,
                                'spread_improvement': spread_improvement,
                                'subspace_angle_raw': subspace_angle_raw,
                                'subspace_angle_pp': subspace_angle_pp,
                            })

        self.cross_dataset_df_ = pd.DataFrame(rows)

    def _compute_spread_distance(self, Z1, Z2):
        """
        Compute a distance metric between the spreads of two datasets in PCA space.
        Uses a combination of covariance distance and mean pairwise distance.
        """
        # Covariance-based distance (how different are the shapes?)
        cov1 = np.cov(Z1.T)
        cov2 = np.cov(Z2.T)
        cov_dist = np.linalg.norm(cov1 - cov2, 'fro')

        # Sample-wise distance (average minimum distance between samples)
        # Take a sample to avoid O(n^2) computation
        n_samples = min(100, Z1.shape[0], Z2.shape[0])
        idx1 = np.random.choice(Z1.shape[0], n_samples, replace=False)
        idx2 = np.random.choice(Z2.shape[0], n_samples, replace=False)

        Z1_sample = Z1[idx1]
        Z2_sample = Z2[idx2]

        # Compute minimum distances
        dists = cdist(Z1_sample, Z2_sample, metric='euclidean')
        min_dist = np.mean(np.minimum(dists.min(axis=0), dists.min(axis=1)))

        # Combine both metrics
        return float(cov_dist + min_dist)

    def get_cross_dataset_summary(self, metric='centroid_improvement'):
        """
        Get a summary of how preprocessing affects inter-dataset distances.

        Args:
            metric: 'centroid_improvement' or 'spread_improvement'
                   Higher values = preprocessing brought datasets closer

        Returns:
            DataFrame sorted by improvement (best preprocessing first)
        """
        if self.cross_dataset_df_ is None or self.cross_dataset_df_.empty:
            raise ValueError("Run fit() first with multiple datasets.")

        # Aggregate across dataset pairs
        summary = self.cross_dataset_df_.groupby('preproc').agg({
            'centroid_improvement': ['mean', 'std'],
            'spread_improvement': ['mean', 'std'],
            'centroid_dist_pp': 'mean',
            'spread_dist_pp': 'mean',
        }).reset_index()

        summary.columns = ['preproc', 'centroid_improv_mean', 'centroid_improv_std',
                          'spread_improv_mean', 'spread_improv_std',
                          'centroid_dist_pp', 'spread_dist_pp']

        # Sort by the requested metric
        sort_col = metric.replace('improvement', 'improv_mean')
        if sort_col in summary.columns:
            summary = summary.sort_values(sort_col, ascending=False)

        return summary

    def get_quality_metric_convergence(self):
        """
        Analyze how preprocessing affects the similarity of quality metrics across datasets.
        Lower variance = preprocessing makes datasets more homogeneous in quality.

        Returns:
            DataFrame with variance of quality metrics (evr, cka, rv, etc.) across datasets
            for raw vs preprocessed data. Lower values = better convergence.
        """
        if self.df_ is None or self.df_.empty:
            raise ValueError("Run fit() first.")

        quality_metrics = ['evr_pre', 'cka', 'rv', 'procrustes', 'trustworthiness', 'grassmann']

        # For raw data, compute variance across datasets (using evr_raw as proxy)
        datasets = self.df_['dataset'].unique()
        raw_variance = {}
        for metric in quality_metrics:
            if metric == 'evr_pre':
                # For raw, use evr_raw
                raw_vals = [self.df_[self.df_['dataset'] == ds]['evr_raw'].iloc[0]
                           for ds in datasets]
            else:
                # For other metrics, we need to compare raw PCA structures
                # Use average across all preprocessings as approximation
                raw_vals = [self.df_[self.df_['dataset'] == ds][metric].mean()
                           for ds in datasets]

            # Invert distance metrics (grassmann, procrustes) so higher = better quality
            # This way variance measures homogeneity in "goodness" not in "distance values"
            if metric in ['grassmann', 'procrustes']:
                raw_vals = [-v for v in raw_vals]

            raw_variance[metric] = float(np.nanvar(raw_vals))

        # For each preprocessing, compute variance across datasets
        results = []
        for preproc in self.df_['preproc'].unique():
            df_pp = self.df_[self.df_['preproc'] == preproc]

            row = {'preproc': preproc}

            for metric in quality_metrics:
                pp_vals = df_pp[metric].values

                # Invert distance metrics (grassmann, procrustes) so higher = better quality
                if metric in ['grassmann', 'procrustes']:
                    pp_vals = -pp_vals

                pp_variance = float(np.nanvar(pp_vals))

                # Convergence = reduction in variance (positive = better)
                convergence = (raw_variance[metric] - pp_variance) / (raw_variance[metric] + 1e-10)

                row[f'{metric}_var_raw'] = raw_variance[metric]
                row[f'{metric}_var_pp'] = pp_variance
                row[f'{metric}_convergence'] = convergence

            results.append(row)

        return pd.DataFrame(results)

    # ---------------- plots ----------------
    def plot_all_datasets_pca(self, figsize=(16, 12)):
        """
        Plot all datasets together in the same PCA space for raw and each preprocessing.
        Shows how datasets cluster and separate in different preprocessing spaces.
        """
        if self.df_ is None or self.df_.empty:
            raise ValueError("Run fit() first.")

        datasets = list(self.raw_pcas_.keys())
        preprocs = sorted(self.df_['preproc'].unique())

        # More contrastive color palette for datasets
        contrastive_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00',
                             '#ffff33', '#a65628', '#f781bf', '#999999', '#66c2a5']
        dataset_colors = {ds: contrastive_colors[i % len(contrastive_colors)]
                         for i, ds in enumerate(datasets)}

        # Determine grid layout: raw + all preprocessings
        n_plots = 1 + len(preprocs)
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Plot 1: Raw data - all datasets together
        ax = axes[0]
        for ds_idx, dname in enumerate(datasets):
            if dname in self.raw_pcas_:
                Zr, _, evr = self.raw_pcas_[dname]
                ax.scatter(Zr[:, 0], Zr[:, 1], alpha=0.5, s=25,
                          c=dataset_colors[dname], label=dname,
                          edgecolors='white', linewidth=0.3)

        ax.set_xlabel('Principal Component 1 (PC1)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Principal Component 2 (PC2)', fontsize=10, fontweight='bold')
        ax.set_title('RAW DATA\nAll Datasets', fontsize=11, fontweight='bold', pad=10)
        # Legend in top-left corner with 3px offset
        ax.legend(loc='upper left', fontsize=8, framealpha=0.9, bbox_to_anchor=(0.01, 0.99))
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')

        # Plot preprocessings
        for pp_idx, pp_name in enumerate(preprocs):
            ax = axes[pp_idx + 1]

            for dname in datasets:
                if (dname, pp_name) in self.pp_pcas_:
                    Zp, _, _ = self.pp_pcas_[(dname, pp_name)]
                    ax.scatter(Zp[:, 0], Zp[:, 1], alpha=0.5, s=25,
                              c=dataset_colors[dname], label=dname,
                              edgecolors='white', linewidth=0.3)

            # Format preprocessing name
            pp_display = pp_name.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(pp_display) > 30:
                pp_display = pp_display[:22] + '...'

            ax.set_xlabel('Principal Component 1 (PC1)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Principal Component 2 (PC2)', fontsize=10, fontweight='bold')
            ax.set_title(f'{pp_display}', fontsize=9, fontweight='bold', pad=10)
            # No legend on preprocessing plots (only on raw)
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_facecolor('#f8f9fa')

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.suptitle('Dataset Clustering in Different Preprocessing Spaces\n(Closer clusters = better for transfer learning)',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_distance_matrices(self, metric='centroid', figsize=(18, 12)):
        """
        Plot distance matrices showing inter-dataset distances for raw and all preprocessings.
        Shows which preprocessing reduces distances (better for transfer learning).

        Args:
            metric: 'centroid' or 'spread' - which distance metric to display
        """
        if self.cross_dataset_df_ is None or self.cross_dataset_df_.empty:
            print("⚠️  No cross-dataset analysis available. Need multiple datasets.")
            return None

        datasets = sorted(set(self.cross_dataset_df_['dataset_1']).union(
                         set(self.cross_dataset_df_['dataset_2'])))
        preprocs = sorted(self.cross_dataset_df_['preproc'].unique())

        # Select metric columns
        metric_col_raw = f'{metric}_dist_raw'
        metric_col_pp = f'{metric}_dist_pp'

        # Define metric name with computation method
        if metric == 'centroid':
            metric_display = 'Centroid Distance (Euclidean L2-norm)'
        elif metric == 'spread':
            metric_display = 'Spread Distance (Frobenius + Sample-wise)'
        else:
            metric_display = metric.capitalize() + ' Distance'

        n_datasets = len(datasets)
        n_preprocs = len(preprocs)

        # Create matrices for raw and each preprocessing
        n_plots = 1 + n_preprocs
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_plots > 1 else [axes]

        # Plot 1: Raw distances
        ax = axes[0]
        raw_matrix = np.zeros((n_datasets, n_datasets))

        for _, row in self.cross_dataset_df_.iterrows():
            i = datasets.index(row['dataset_1'])
            j = datasets.index(row['dataset_2'])
            val = row[metric_col_raw]
            raw_matrix[i, j] = val
            raw_matrix[j, i] = val

        # Use YlOrRd colormap for better text readability (light to dark)
        im = ax.imshow(raw_matrix, cmap='YlOrRd', aspect='auto')
        ax.set_title(f'RAW DATA\n{metric_display}', fontsize=10, fontweight='bold')
        ax.set_xticks(range(n_datasets))
        ax.set_yticks(range(n_datasets))
        ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(datasets, fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Add text annotations with adaptive color (white for dark cells)
        vmax = raw_matrix.max()
        for i in range(n_datasets):
            for j in range(n_datasets):
                if i != j:
                    val = raw_matrix[i, j]
                    text_color = 'white' if val > 0.8 * vmax else 'black'
                    # Use scientific notation if value is very small
                    val_str = f'{val:.2e}' if val < 0.001 else f'{val:.3f}'
                    ax.text(j, i, val_str,
                           ha="center", va="center", color=text_color, fontsize=7, fontweight='bold')

        # Plot preprocessed distances
        for pp_idx, pp_name in enumerate(preprocs):
            ax = axes[pp_idx + 1]
            pp_matrix = np.zeros((n_datasets, n_datasets))

            df_pp = self.cross_dataset_df_[self.cross_dataset_df_['preproc'] == pp_name]

            for _, row in df_pp.iterrows():
                i = datasets.index(row['dataset_1'])
                j = datasets.index(row['dataset_2'])
                val = row[metric_col_pp]
                pp_matrix[i, j] = val
                pp_matrix[j, i] = val

            # Use same colormap and scale for comparison
            im = ax.imshow(pp_matrix, cmap='YlOrRd', aspect='auto', vmin=raw_matrix.min(), vmax=raw_matrix.max())

            # Format preprocessing name
            pp_display = pp_name.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(pp_display) > 22:
                pp_display = pp_display[:22] + '...'

            ax.set_title(f'{pp_display}', fontsize=9, fontweight='bold')
            ax.set_xticks(range(n_datasets))
            ax.set_yticks(range(n_datasets))
            ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(datasets, fontsize=8)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add text annotations with adaptive color
            vmax = raw_matrix.max()
            for i in range(n_datasets):
                for j in range(n_datasets):
                    if i != j:
                        val = pp_matrix[i, j]
                        text_color = 'white' if val > 0.8 * vmax else 'black'
                        # Use scientific notation if value is very small
                        val_str = f'{val:.2e}' if val < 0.001 else f'{val:.3f}'
                        ax.text(j, i, val_str,
                               ha="center", va="center", color=text_color, fontsize=6, fontweight='bold')

        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Inter-Dataset {metric_display} Matrices\n(Lower values = better transfer learning potential)',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_distance_reduction_ranking(self, metric='centroid', log_scale=False, figsize=(14, 8)):
        """
        Bar chart showing which preprocessing methods best reduce inter-dataset distances.
        Directly answers: "Which preprocessing is best for transfer learning?"

        Args:
            metric: 'centroid' or 'spread' - which distance method to use for ranking
            log_scale: If True, use log scale for the right plot (absolute distances) to handle extreme values
        """
        if self.cross_dataset_df_ is None or self.cross_dataset_df_.empty:
            print("⚠️  No cross-dataset analysis available. Need multiple datasets.")
            return None

        # Compute average distance reduction for each preprocessing
        results = []

        # Select metric columns and define display name with computation method
        metric_col_raw = f'{metric}_dist_raw'
        metric_col_pp = f'{metric}_dist_pp'

        if metric == 'centroid':
            metric_display = 'Centroid Distance'
            metric_method = 'Method: Euclidean L2-norm between PCA centroids'
        elif metric == 'spread':
            metric_display = 'Spread Distance'
            metric_method = 'Method: Frobenius norm (covariance) + sample-wise Euclidean'
        else:
            metric_display = metric.capitalize() + ' Distance'
            metric_method = ''

        # Get raw distances
        raw_dists = self.cross_dataset_df_.groupby(['dataset_1', 'dataset_2'])[metric_col_raw].first()
        avg_raw_dist = raw_dists.mean()

        for pp_name in self.cross_dataset_df_['preproc'].unique():
            df_pp = self.cross_dataset_df_[self.cross_dataset_df_['preproc'] == pp_name]

            avg_pp_dist = df_pp[metric_col_pp].mean()
            reduction = ((avg_raw_dist - avg_pp_dist) / avg_raw_dist) * 100  # Percentage

            results.append({
                'preproc': pp_name,
                'avg_distance': avg_pp_dist,
                'reduction_pct': reduction,
                'raw_distance': avg_raw_dist
            })

        df_results = pd.DataFrame(results).sort_values('reduction_pct', ascending=False)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Plot 1: Distance reduction percentage
        labels = []
        for pp in df_results['preproc'].values:
            formatted = pp.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(formatted) > 30:
                formatted = formatted[:30] + '...'
            labels.append(formatted)

        y_pos = np.arange(len(labels))
        colors = ['green' if x > 0 else 'red' for x in df_results['reduction_pct'].values]

        bars = ax1.barh(y_pos, df_results['reduction_pct'].values, color=colors, alpha=0.7, edgecolor='black', linewidth=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.set_xlabel(f'{metric_display} Reduction (%)', fontsize=11, fontweight='bold')
        ax1.set_title(f'Transfer Learning Potential\n{metric_display} | {metric_method}\n(Higher = Better)',
                     fontsize=11, fontweight='bold', pad=15)
        ax1.axvline(0, color='black', linewidth=1.5, linestyle='--')
        ax1.grid(axis='x', alpha=0.3)
        ax1.set_facecolor('#f8f9fa')

        # Apply log scale to handle extreme negative values
        if log_scale:
            # Use symlog to handle both positive and negative values
            ax1.set_xscale('symlog', linthresh=1.0)
            ax1.set_xlabel(f'{metric_display} Reduction (%, symlog scale)', fontsize=11, fontweight='bold')

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_results['reduction_pct'].values)):
            label_x = val + (2 if val > 0 else -2)
            ha = 'left' if val > 0 else 'right'
            ax1.text(label_x, bar.get_y() + bar.get_height() / 2, f'{val:.1f}%',
                    ha=ha, va='center', fontsize=8, fontweight='bold')

        # Plot 2: Absolute distances
        x = np.arange(len(df_results))
        width = 0.35

        bars1 = ax2.bar(x - width/2, [avg_raw_dist] * len(df_results), width,
                       label='Raw', alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.8)
        bars2 = ax2.bar(x + width/2, df_results['avg_distance'].values, width,
                       label='Preprocessed', alpha=0.8, color='coral', edgecolor='black', linewidth=0.8)

        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax2.set_ylabel(f'Average {metric_display}', fontsize=11, fontweight='bold')
        ax2.set_title(f'Absolute {metric_display} Comparison\n{metric_method}',
                     fontsize=11, fontweight='bold', pad=15)
        ax2.legend(fontsize=10, framealpha=0.9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_facecolor('#f8f9fa')

        # Apply log scale if requested (helps with extreme values)
        if log_scale:
            ax2.set_yscale('log')
            ax2.set_ylabel(f'Average {metric_display} (log scale)', fontsize=11, fontweight='bold')

        plt.suptitle('Preprocessing Ranking for Transfer Learning\n' +
                    f'{metric_display} - Best preprocessing reduces distance between datasets',
                    fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    def plot_quality_metric_convergence(self, figsize=(16, 10)):
        """
        Visualize how preprocessing makes quality metrics more homogeneous across datasets.
        Shows variance reduction in EVR, CKA, RV, Procrustes, Trustworthiness, Grassmann.

        Lower variance after preprocessing = datasets behave more similarly = better for transfer learning.
        """
        if self.df_ is None or self.df_.empty or len(self.df_['dataset'].unique()) < 2:
            print("⚠️  Need at least 2 datasets for quality metric convergence analysis.")
            return None

        convergence_df = self.get_quality_metric_convergence()

        # Get convergence columns
        quality_metrics = ['evr_pre', 'cka', 'rv', 'procrustes', 'trustworthiness', 'grassmann']
        metric_display_names = ['EVR', 'CKA', 'RV', 'Procrustes*', 'Trustworthiness', 'Grassmann*']

        # Sort by average convergence
        avg_convergence = convergence_df[[f'{m}_convergence' for m in quality_metrics]].mean(axis=1)
        convergence_df['avg_convergence'] = avg_convergence
        convergence_df = convergence_df.sort_values('avg_convergence', ascending=False)

        # Format preprocessing names
        labels = []
        for pp in convergence_df['preproc'].values:
            formatted = pp.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(formatted) > 30:
                formatted = formatted[:30] + '...'
            labels.append(formatted)

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        colors_map = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#34495e']

        for idx, (metric, display_name, color) in enumerate(zip(quality_metrics, metric_display_names, colors_map)):
            ax = axes[idx]

            convergence_vals = convergence_df[f'{metric}_convergence'].values
            y_pos = np.arange(len(labels))

            # Color based on positive (green) or negative (red) convergence
            bar_colors = ['green' if v > 0 else 'red' for v in convergence_vals]

            bars = ax.barh(y_pos, convergence_vals, color=bar_colors, alpha=0.7,
                          edgecolor='black', linewidth=0.5)

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            ax.set_xlabel('Variance Reduction', fontsize=9, fontweight='bold')

            # Add note about inversion for distance metrics
            subtitle = '(+ = datasets more similar)'
            if metric in ['procrustes', 'grassmann']:
                subtitle = '(inverted*, + = datasets more similar)'

            ax.set_title(f'{display_name} Convergence\n{subtitle}',
                        fontsize=10, fontweight='bold', pad=10)
            ax.axvline(0, color='black', linewidth=1.5, linestyle='--')
            ax.grid(axis='x', alpha=0.3)
            ax.set_facecolor('#f8f9fa')

            # Add value labels for significant convergence
            for i, (bar, val) in enumerate(zip(bars, convergence_vals)):
                if abs(val) > 0.1:  # Only show if significant
                    label_x = val + (0.02 if val > 0 else -0.02)
                    ha = 'left' if val > 0 else 'right'
                    ax.text(label_x, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                           ha=ha, va='center', fontsize=6, fontweight='bold')

        plt.suptitle('Quality Metric Convergence Across Datasets\n' +
                    'How preprocessing makes datasets more homogeneous in quality characteristics\n' +
                    '(* = inverted for consistency; Positive = variance reduction = better for transfer learning)',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        return fig

    def plot_pair(self, dataset: str, preproc: str, figsize=(10, 5)):
        """Enhanced comparison plot for a specific dataset-preprocessing pair."""
        if (dataset, preproc) not in self.cache_:
            raise ValueError(f"No data for ({dataset}, {preproc}). Run fit() first.")

        Zr, Zp = self.cache_[(dataset, preproc)]
        Ar, Ap, disparity = procrustes(Zr[:, :2], Zp[:, :2])

        # Get metrics
        row = self.df_[(self.df_['dataset'] == dataset) & (self.df_['preproc'] == preproc)].iloc[0]

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Raw PCA
        axes[0].scatter(Ar[:, 0], Ar[:, 1], s=50, alpha=0.6, c='steelblue', edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel('PC1', fontweight='bold')
        axes[0].set_ylabel('PC2', fontweight='bold')
        axes[0].set_title(f'{dataset} - Raw PCA\nEVR: {row["evr_raw"]:.4f}', fontweight='bold')
        axes[0].grid(alpha=0.3, linestyle='--')
        axes[0].set_facecolor('#f8f9fa')
        axes[0].set_aspect('equal', 'box')

        # Preprocessed PCA (Procrustes aligned)
        axes[1].scatter(Ap[:, 0], Ap[:, 1], s=50, alpha=0.6, c='coral', edgecolors='black', linewidth=0.5)
        axes[1].set_xlabel('PC1', fontweight='bold')
        axes[1].set_ylabel('PC2', fontweight='bold')
        # Format preprocessing name for readability
        pp_display = preproc.split('|')[-1].replace('MinMax>', '').replace('>', ' → ')
        axes[1].set_title(f'{pp_display}\nEVR: {row["evr_pre"]:.4f}', fontweight='bold', fontsize=10)
        axes[1].grid(alpha=0.3, linestyle='--')
        axes[1].set_facecolor('#f8f9fa')
        axes[1].set_aspect('equal', 'box')

        # Add metrics text box
        metrics_text = (f'CKA: {row["cka"]:.4f}\n'
                       f'RV: {row["rv"]:.4f}\n'
                       f'Procrustes: {row["procrustes"]:.4f}\n'
                       f'Trust: {row["trustworthiness"]:.4f}')

        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle(f'PCA Comparison: {dataset} / {preproc}',
                    fontsize=13, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0.08, 1, 0.96])
        return fig

    def plot_preservation_summary(self, by="preproc", figsize=(14, 8)):
        """Enhanced summary plot with better styling."""
        if self.df_ is None or self.df_.empty:
            raise ValueError("Run fit() first.")

        agg = self.df_.groupby(by).agg({
            'evr_pre': 'mean', 'grassmann': 'mean', 'cka': 'mean',
            'rv': 'mean', 'procrustes': 'mean', 'trustworthiness': 'mean'
        }).reset_index()

        # flip distances, min-max normalize (handle NaN values from incompatible feature spaces)
        agg['grassmann'] = -agg['grassmann']
        agg['procrustes'] = -agg['procrustes']
        for c in ['evr_pre', 'grassmann', 'cka', 'rv', 'procrustes', 'trustworthiness']:
            v = agg[c].values
            valid_mask = ~np.isnan(v)
            if valid_mask.sum() > 0:
                v_min, v_max = v[valid_mask].min(), v[valid_mask].max()
                rng = v_max - v_min
                if rng > 1e-12:
                    agg[c] = np.where(valid_mask, (v - v_min) / rng, np.nan)
                else:
                    agg[c] = np.where(valid_mask, 0.5, np.nan)

        metrics = ['evr_pre', 'cka', 'rv', 'trustworthiness', 'grassmann', 'procrustes']
        metric_labels = ['EVR', 'CKA', 'RV', 'Trust', 'Grassmann*', 'Procrustes*']
        colors_map = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#34495e']

        x = np.arange(len(agg[by]))
        w = 0.13

        fig, ax = plt.subplots(figsize=figsize)

        for i, (m, label, color) in enumerate(zip(metrics, metric_labels, colors_map)):
            values = agg[m].values
            offset = (i - 2.5) * w
            bars = ax.bar(x + offset, values, w, label=label, color=color,
                         alpha=0.8, edgecolor='black', linewidth=0.5)

            # Add value labels on top of bars (only for non-NaN)
            for j, (bar, val) in enumerate(zip(bars, values)):
                if not np.isnan(val) and val > 0.05:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{val:.2f}', ha='center', va='bottom', fontsize=7, rotation=0)

        # Styling - format preprocessing names for readability
        ax.set_xticks(x)
        labels = []
        for label in agg[by].values:
            # Extract meaningful part and format
            formatted = label.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            # Limit to reasonable length but keep it readable
            if len(formatted) > 25:
                formatted = formatted[:25] + '...'
            labels.append(formatted)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel('Normalized Score (0-1)', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'{by.capitalize()}', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.95, ncol=3)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8f9fa')

        ax.set_title(f'Preprocessing Structure Preservation by {by.capitalize()}\n' +
                    '(* inverted: higher is better)',
                    fontsize=13, fontweight='bold', pad=15)

        plt.tight_layout()
        return fig

    def plot_cross_dataset_distances(self, figsize=(14, 8)):
        """
        Plot how preprocessing affects inter-dataset distances.
        Shows which preprocessing methods bring datasets closer together.
        """
        if self.cross_dataset_df_ is None or self.cross_dataset_df_.empty:
            print("⚠️  No cross-dataset analysis available. Need multiple datasets.")
            return None

        summary = self.get_cross_dataset_summary()

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Format preprocessing names for display
        labels = []
        for pp in summary['preproc'].values:
            formatted = pp.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(formatted) > 30:
                formatted = formatted[:30] + '...'
            labels.append(formatted)

        x = np.arange(len(summary))

        # 1. Centroid improvement
        ax = axes[0, 0]
        colors = ['green' if v > 0 else 'red' for v in summary['centroid_improv_mean']]
        bars = ax.barh(x, summary['centroid_improv_mean'], xerr=summary['centroid_improv_std'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Centroid Improvement', fontweight='bold')
        ax.set_title('Dataset Centroid Distance Change\n(+: datasets closer, -: datasets farther)',
                    fontsize=10, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # 2. Spread improvement
        ax = axes[0, 1]
        colors = ['green' if v > 0 else 'red' for v in summary['spread_improv_mean']]
        bars = ax.barh(x, summary['spread_improv_mean'], xerr=summary['spread_improv_std'],
                       color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Spread Improvement', fontweight='bold')
        ax.set_title('Dataset Distribution Overlap Change\n(+: distributions closer, -: distributions farther)',
                    fontsize=10, fontweight='bold')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.grid(axis='x', alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # 3. Absolute centroid distances
        ax = axes[1, 0]
        raw_dists = self.cross_dataset_df_.groupby('preproc')['centroid_dist_raw'].mean()
        pp_dists = summary['centroid_dist_pp'].values

        width = 0.35
        x_pos = np.arange(len(summary))
        ax.bar(x_pos - width/2, raw_dists.values, width, label='Raw',
               alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + width/2, pp_dists, width, label='Preprocessed',
               alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Distance', fontweight='bold')
        ax.set_title('Absolute Centroid Distances', fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        # 4. Absolute spread distances
        ax = axes[1, 1]
        raw_spread = self.cross_dataset_df_.groupby('preproc')['spread_dist_raw'].mean()
        pp_spread = summary['spread_dist_pp'].values

        ax.bar(x_pos - width/2, raw_spread.values, width, label='Raw',
               alpha=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.bar(x_pos + width/2, pp_spread, width, label='Preprocessed',
               alpha=0.8, color='coral', edgecolor='black', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Distance', fontweight='bold')
        ax.set_title('Absolute Spread Distances', fontsize=10, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.set_facecolor('#f8f9fa')

        plt.suptitle('Cross-Dataset Distance Analysis\n' +
                    'Evaluating Preprocessing Impact on Multi-Machine Compatibility',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_cross_dataset_heatmap(self, metric='centroid_improvement', figsize=(12, 10)):
        """
        Create a heatmap showing pairwise dataset distances for each preprocessing.

        Args:
            metric: 'centroid_improvement', 'centroid_dist_pp', 'spread_improvement', or 'spread_dist_pp'
        """
        if self.cross_dataset_df_ is None or self.cross_dataset_df_.empty:
            print("⚠️  No cross-dataset analysis available. Need multiple datasets.")
            return None

        # Get unique datasets and preprocessings
        datasets = sorted(set(self.cross_dataset_df_['dataset_1']).union(
                         set(self.cross_dataset_df_['dataset_2'])))
        preprocs = sorted(self.cross_dataset_df_['preproc'].unique())

        n_preprocs = len(preprocs)
        n_datasets = len(datasets)

        # Determine subplot layout
        n_cols = min(3, n_preprocs)
        n_rows = (n_preprocs + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_preprocs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Create heatmap for each preprocessing
        for idx, pp_name in enumerate(preprocs):
            ax = axes[idx]

            # Create distance matrix
            dist_matrix = np.zeros((n_datasets, n_datasets))

            df_pp = self.cross_dataset_df_[self.cross_dataset_df_['preproc'] == pp_name]

            for _, row in df_pp.iterrows():
                i = datasets.index(row['dataset_1'])
                j = datasets.index(row['dataset_2'])
                val = row[metric]
                dist_matrix[i, j] = val
                dist_matrix[j, i] = val  # Symmetric

            # Choose colormap based on metric
            if 'improvement' in metric:
                cmap = 'RdYlGn'  # Red (worse) to Green (better)
                vmin, vmax = -1, 1
            else:
                cmap = 'YlOrRd'  # Yellow (close) to Red (far)
                vmin, vmax = None, None

            im = ax.imshow(dist_matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

            # Format preprocessing name
            pp_display = pp_name.split('|')[-1].replace('MinMax>', '').replace('>', '→')
            if len(pp_display) > 40:
                pp_display = pp_display[:40] + '...'

            ax.set_title(pp_display, fontsize=8, fontweight='bold')
            ax.set_xticks(range(n_datasets))
            ax.set_yticks(range(n_datasets))
            ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=7)
            ax.set_yticklabels(datasets, fontsize=7)

            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Add text annotations
            for i in range(n_datasets):
                for j in range(n_datasets):
                    if i != j and dist_matrix[i, j] != 0:
                        text = ax.text(j, i, f'{dist_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontsize=6)

        # Hide unused subplots
        for idx in range(n_preprocs, len(axes)):
            axes[idx].axis('off')

        metric_display = metric.replace('_', ' ').title()
        plt.suptitle(f'Cross-Dataset {metric_display} Heatmaps\n' +
                    'Comparing Dataset Compatibility Across Preprocessing Methods',
                    fontsize=13, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_all(self, show=True):
        """Generate all visualization plots."""
        figs = []

        # # 1. Summary comparison
        # print("📊 Generating summary comparison...")
        # figs.append(self.plot_summary(by="preproc"))

        # # 2. PCA scatter plots
        # print("📈 Generating PCA scatter plots...")
        # figs.append(self.plot_pca_scatter())

        # # 3. Distance network
        # print("🕸️  Generating similarity network...")
        # figs.append(self.plot_distance_network(metric='cka'))

        if show:
            plt.show()

        return figs
