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
import networkx as nx

class PreprocPCAEvaluator:
    def __init__(self, r_components=10, knn=10):
        self.r = r_components
        self.knn = knn
        self.df_ = None
        self.cache_ = {}
        self.raw_pcas_ = {}  # Store raw PCA results for visualization

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

        self.df_ = pd.DataFrame(rows)
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

    # ---------------- plots ----------------
    def plot_pca_scatter(self, dataset=None, n_components=2, figsize=(14, 10)):
        """
        Plot PCA scatter plots for raw and preprocessed data.
        If dataset is None, plot all datasets in subplots.
        """
        if self.df_ is None or self.df_.empty:
            raise ValueError("Run fit() first.")

        datasets = [dataset] if dataset else self.df_['dataset'].unique()
        preprocs = self.df_['preproc'].unique()[:9]  # Limit to 9 for visibility

        n_datasets = len(datasets)
        fig = plt.figure(figsize=figsize)

        # Color palette
        colors = cm.get_cmap('tab10', len(preprocs))

        for idx, dname in enumerate(datasets):
            # Plot raw PCA
            ax_raw = plt.subplot(n_datasets, 2, idx * 2 + 1)
            if dname in self.raw_pcas_:
                Zr, _, evr = self.raw_pcas_[dname]
                ax_raw.scatter(Zr[:, 0], Zr[:, 1], alpha=0.6, s=40, c='gray', edgecolors='black', linewidth=0.5)
                ax_raw.set_xlabel(f'PC1', fontsize=10, fontweight='bold')
                ax_raw.set_ylabel(f'PC2', fontsize=10, fontweight='bold')
                ax_raw.set_title(f'{dname} - Raw Data (EVR: {evr:.3f})', fontsize=11, fontweight='bold', pad=10)
                ax_raw.grid(alpha=0.3, linestyle='--')
                ax_raw.set_facecolor('#f8f9fa')

            # Plot preprocessed PCAs
            ax_pp = plt.subplot(n_datasets, 2, idx * 2 + 2)
            for pp_idx, pp_name in enumerate(preprocs):
                if (dname, pp_name) in self.cache_:
                    _, Zp = self.cache_[(dname, pp_name)]
                    # Extract readable label from full preprocessing name
                    label = pp_name.split('|')[-1].replace('MinMax>', '').replace('>', ' → ')[:30]
                    ax_pp.scatter(Zp[:, 0], Zp[:, 1], alpha=0.5, s=30, c=[colors(pp_idx)],
                                label=label, edgecolors='white', linewidth=0.3)

            ax_pp.set_xlabel(f'PC1', fontsize=10, fontweight='bold')
            ax_pp.set_ylabel(f'PC2', fontsize=10, fontweight='bold')
            ax_pp.set_title(f'{dname} - Preprocessed Overlays', fontsize=11, fontweight='bold', pad=10)
            ax_pp.legend(loc='best', fontsize=7, framealpha=0.9, ncol=1, bbox_to_anchor=(1.05, 1), borderaxespad=0)
            ax_pp.grid(alpha=0.3, linestyle='--')
            ax_pp.set_facecolor('#f8f9fa')

        plt.suptitle('PCA Projection Comparison: Raw vs Preprocessed',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig

    def plot_distance_network(self, metric='cka', threshold=None, figsize=(12, 10)):
        """
        Plot a network graph showing similarities/distances between preprocessing methods.

        Args:
            metric: 'cka', 'rv', 'procrustes', or 'trustworthiness'
            threshold: Only show edges above/below this value. Auto-computed if None.
            figsize: Figure size
        """
        if self.df_ is None or self.df_.empty:
            raise ValueError("Run fit() first.")

        # Aggregate metric across datasets
        agg = self.df_.groupby('preproc')[metric].mean().reset_index()

        # Create similarity matrix between preprocessings
        preprocs = agg['preproc'].values
        n = len(preprocs)

        if n < 2:
            print(f"⚠️  Need at least 2 preprocessing methods for network plot. Found {n}.")
            return None

        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                # Compute pairwise distance based on metric difference
                rows_i = self.df_[self.df_['preproc'] == preprocs[i]]
                rows_j = self.df_[self.df_['preproc'] == preprocs[j]]

                # Match datasets and compute distance
                common_datasets = set(rows_i['dataset']).intersection(set(rows_j['dataset']))
                if len(common_datasets) > 0:
                    vals_i = np.array([rows_i[rows_i['dataset'] == d][metric].values[0]
                                     for d in common_datasets])
                    vals_j = np.array([rows_j[rows_j['dataset'] == d][metric].values[0]
                                     for d in common_datasets])
                    # Similarity as 1 / (1 + distance)
                    distance = np.mean(np.abs(vals_i - vals_j))
                    sim = 1.0 / (1.0 + distance)
                    similarity_matrix[i, j] = similarity_matrix[j, i] = sim if not np.isnan(sim) else 0

        # Auto threshold
        positive_sims = similarity_matrix[similarity_matrix > 0]
        if len(positive_sims) == 0:
            print("⚠️  No positive similarities found. Cannot create network.")
            return None

        if threshold is None:
            threshold = np.percentile(positive_sims, 60)

        # Create graph
        G = nx.Graph()
        for i, pp in enumerate(preprocs):
            score = agg[agg['preproc'] == pp][metric].values[0]
            G.add_node(pp, score=score)

        edge_count = 0
        for i in range(n):
            for j in range(i+1, n):
                if similarity_matrix[i, j] > threshold:
                    G.add_edge(preprocs[i], preprocs[j], weight=similarity_matrix[i, j])
                    edge_count += 1

        if edge_count == 0:
            print(f"⚠️  No edges above threshold {threshold:.3f}. Lowering threshold...")
            threshold = np.percentile(positive_sims, 30)
            for i in range(n):
                for j in range(i+1, n):
                    if similarity_matrix[i, j] > threshold:
                        G.add_edge(preprocs[i], preprocs[j], weight=similarity_matrix[i, j])

        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Draw
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_facecolor('#f0f0f0')

        # Node colors based on metric score
        node_scores = [G.nodes[node]['score'] for node in G.nodes()]
        vmin, vmax = min(node_scores), max(node_scores)

        # Draw edges
        edges = G.edges()
        if len(edges) > 0:
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=[w*3 for w in weights], alpha=0.4,
                                  edge_color='gray', ax=ax)

        # Draw nodes
        nodes = nx.draw_networkx_nodes(G, pos, node_color=node_scores, node_size=800,
                                       cmap='RdYlGn', vmin=vmin, vmax=vmax,
                                       edgecolors='black', linewidths=2, ax=ax)

        # Draw labels with readable preprocessing names
        labels = {}
        for node in G.nodes():
            # Extract preprocessing steps and format nicely
            steps = node.split('|')[-1].replace('MinMax>', '').split('>')
            if len(steps) <= 2:
                labels[node] = '\n'.join(steps)
            else:
                labels[node] = steps[0] + '→' + steps[1] + '...'
        nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight='bold', ax=ax)

        # Colorbar
        sm = cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'{metric.upper()} Score', fontsize=11, fontweight='bold')

        ax.set_title(f'Preprocessing Similarity Network\n' +
                    f'(Based on {metric.upper()} metric, {len(edges)} edges shown)',
                    fontsize=13, fontweight='bold', pad=20)
        ax.axis('off')
        plt.tight_layout()
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

    def plot_summary(self, by="preproc", figsize=(14, 8)):
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

        ax.set_title(f'Preprocessing Quality Comparison by {by.capitalize()}\n' +
                    '(* inverted: higher is better)',
                    fontsize=13, fontweight='bold', pad=15)

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
