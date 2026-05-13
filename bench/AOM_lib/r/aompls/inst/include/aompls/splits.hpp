// SPDX-License-Identifier: CeCILL-2.1
// Cross-validation splitters: own-RNG KFold, SPXYFold, and EXTERNAL adapter.
//
// Important: own-RNG KFold is NOT bit-compatible with sklearn's MT19937 KFold.
// For parity against the Python reference, pass `cv_mode = EXTERNAL` and
// provide the fold partitions in AOMConfig::external_folds.
//
// SPXYFold mirrors nirs4all/operators/splitters/splitters.py:670-910 — joint
// X+y distance, alternating max-min assignment with `max_size = floor(n/K) +
// (1 if n%K else 0)` cap (every fold can reach max_size).

#pragma once

#include "aompls/eigen_alias.hpp"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace aompls {

using FoldPartition = std::vector<std::vector<int>>;  // outer = folds, inner = TEST indices

// ---------------------------------------------------------------------------
// PCG-style LCG for deterministic shuffles. We document this as not matching
// numpy/sklearn — parity tests use EXTERNAL folds.
// ---------------------------------------------------------------------------
class LcgRng {
   public:
    explicit LcgRng(std::uint64_t seed) : state_(seed * 2862933555777941757ULL + 3037000493ULL) {}
    std::uint64_t next() {
        state_ = state_ * 6364136223846793005ULL + 1442695040888963407ULL;
        return state_;
    }
    int randint(int n) {
        // Bias-light bounded sampling (acceptable for our small n).
        return static_cast<int>(next() % static_cast<std::uint64_t>(n));
    }

   private:
    std::uint64_t state_;
};

inline FoldPartition kfold(int n, int n_splits, std::uint64_t seed, bool shuffle = true) {
    if (n_splits < 2) throw std::invalid_argument("n_splits must be >= 2");
    if (n < n_splits) throw std::invalid_argument("n must be >= n_splits");
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    if (shuffle) {
        LcgRng rng(seed + 1);
        for (int i = n - 1; i > 0; --i) {
            int j = rng.randint(i + 1);
            std::swap(indices[i], indices[j]);
        }
    }
    FoldPartition folds(n_splits);
    const int base = n / n_splits;
    const int rem = n % n_splits;
    int cursor = 0;
    for (int f = 0; f < n_splits; ++f) {
        const int size = base + (f < rem ? 1 : 0);
        folds[f].assign(indices.begin() + cursor, indices.begin() + cursor + size);
        cursor += size;
    }
    return folds;
}

// ---------------------------------------------------------------------------
// SPXYFold (mirrors splitters.py:670-910).
// ---------------------------------------------------------------------------
inline Mat pairwise_euclidean(const Eigen::Ref<const Mat>& M) {
    // Pairwise euclidean distance matrix (n, n).
    const Idx n = M.rows();
    Vec sq = M.rowwise().squaredNorm();           // (n,)
    Mat D2 = sq.replicate(1, n) + sq.transpose().replicate(n, 1) - 2.0 * M * M.transpose();
    D2 = D2.cwiseMax(0.0);
    return D2.array().sqrt();
}

inline FoldPartition spxy_fold(const Eigen::Ref<const Mat>& X, const Vec& y, int n_splits) {
    if (n_splits < 2) throw std::invalid_argument("n_splits must be >= 2");
    const Idx n = X.rows();
    if (y.size() != n) throw std::invalid_argument("y length must match X rows");
    if (n_splits > n) throw std::invalid_argument("n_splits must be <= n");

    Mat D_X = pairwise_euclidean(X);
    double maxX = D_X.maxCoeff();
    if (maxX > 0) D_X /= maxX;
    Mat y_col(n, 1); y_col.col(0) = y;
    Mat D_Y = pairwise_euclidean(y_col);
    double maxY = D_Y.maxCoeff();
    if (maxY > 0) D_Y /= maxY;
    Mat D = D_X + D_Y;  // (n, n) combined distance

    // Initialisation: pick `n_splits` samples farthest from centroid (row-mean of D).
    Vec centroid_dist = D.rowwise().mean();
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return centroid_dist(a) < centroid_dist(b); });
    // Take the last `n_splits` entries (largest centroid distances) but preserve
    // their natural index order to match the Python reference:
    //   init_indices = np.argsort(centroid_distances)[-n_splits:]
    // → sorted ascending; we keep the same ordering when assigning to folds.
    std::vector<int> init_indices(order.end() - n_splits, order.end());

    std::vector<int> fold_assignment(n, -1);
    std::vector<std::vector<int>> fold_members(n_splits);
    std::vector<int> fold_sizes(n_splits, 1);
    for (int f = 0; f < n_splits; ++f) {
        fold_assignment[init_indices[f]] = f;
        fold_members[f].push_back(init_indices[f]);
    }

    const int base = static_cast<int>(n) / n_splits;
    const int rem = static_cast<int>(n) % n_splits;
    const int max_size = base + (rem > 0 ? 1 : 0);

    // `remaining` preserves insertion order; we mirror the reference's iteration
    // by using a contiguous vector of unassigned indices.
    std::vector<int> remaining;
    remaining.reserve(n - n_splits);
    for (int i = 0; i < n; ++i)
        if (fold_assignment[i] < 0) remaining.push_back(i);

    while (!remaining.empty()) {
        bool advanced = false;
        for (int f = 0; f < n_splits && !remaining.empty(); ++f) {
            if (fold_sizes[f] >= max_size) continue;
            // For each candidate r in remaining, compute min distance to fold members.
            double best_score = -1.0;
            int best_pos = -1;  // position in `remaining`
            int best_idx = -1;
            for (std::size_t pi = 0; pi < remaining.size(); ++pi) {
                const int r = remaining[pi];
                double mind = std::numeric_limits<double>::infinity();
                for (int m : fold_members[f]) {
                    const double d = D(r, m);
                    if (d < mind) mind = d;
                }
                if (mind > best_score) {
                    best_score = mind;
                    best_pos = static_cast<int>(pi);
                    best_idx = r;
                }
            }
            if (best_idx < 0) continue;
            fold_assignment[best_idx] = f;
            fold_members[f].push_back(best_idx);
            ++fold_sizes[f];
            remaining.erase(remaining.begin() + best_pos);
            advanced = true;
        }
        if (!advanced) break;
    }

    FoldPartition folds(n_splits);
    for (int i = 0; i < n; ++i) {
        const int f = fold_assignment[i];
        if (f >= 0) folds[f].push_back(i);
    }
    return folds;
}

// Helper: derive train indices (complement of test set).
inline std::vector<int> complement(int n, const std::vector<int>& test_idx) {
    std::vector<char> mask(n, 0);
    for (int t : test_idx) mask[t] = 1;
    std::vector<int> train;
    train.reserve(n - static_cast<int>(test_idx.size()));
    for (int i = 0; i < n; ++i)
        if (!mask[i]) train.push_back(i);
    return train;
}

}  // namespace aompls
