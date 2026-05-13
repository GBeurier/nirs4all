// SPDX-License-Identifier: CeCILL-2.1
// AOM global operator selection with auto-prefix scoring and one-SE rule.
//
// Reference: bench/AOM_v0/aompls/selection.py:236-282 (_cv_score_per_prefix)
//          and bench/AOM_v0/aompls/selection.py:322-449 (select_global).
//
// The auto-prefix optimisation fits SIMPLS once per (operator, fold) at
// K=max_components and then evaluates every prefix via SimplsResult::coef_prefix(k).
//
// The one-SE rule mirrors the Python reference exactly: SE is computed from
// the variability of the *winning operator's RMSE curve across prefixes*
// (std/sqrt(K)), not from fold-level variance. Among all (b, k) within
// best_score + SE, the smallest k (then smallest b) wins.

#pragma once

#include "aompls/eigen_alias.hpp"
#include "aompls/operators.hpp"
#include "aompls/simpls.hpp"
#include "aompls/splits.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace aompls {

struct SelectionDiagnostics {
    int selected_operator_index = 0;
    int selected_n_components = 0;
    double best_score = std::numeric_limits<double>::infinity();
    bool one_se_applied = false;
    std::vector<std::vector<double>> rmse_curves;  // (n_ops, max_components)
};

// Score per-prefix CV RMSE for every operator in `bank` using `fold_partition`
// (test indices per fold). Xc / yc must already be centered for the global fit;
// per-fold re-centering is performed internally to match the reference scorer.
inline std::vector<std::vector<double>> score_operator_curves(
    const Eigen::Ref<const Mat>& X,            // RAW (uncentered)
    const Vec& y,                              // RAW
    const std::vector<OperatorPtr>& bank,
    const FoldPartition& fold_partition,
    int max_components) {
    const int n_ops = static_cast<int>(bank.size());
    const int K = max_components;
    const int n_folds = static_cast<int>(fold_partition.size());
    const Idx n = X.rows();
    const double kInf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> curves(n_ops, std::vector<double>(K, kInf));
    if (n_folds == 0) return curves;
    // Per-(op, fold, k) RMSE — initialise to +inf to match Python's np.full(..., np.inf).
    std::vector<std::vector<std::vector<double>>> per_fold(
        n_ops, std::vector<std::vector<double>>(n_folds, std::vector<double>(K, kInf)));

    for (int f = 0; f < n_folds; ++f) {
        const auto& test_idx = fold_partition[f];
        std::vector<int> train_idx = complement(static_cast<int>(n), test_idx);
        const int n_tr = static_cast<int>(train_idx.size());
        const int n_va = static_cast<int>(test_idx.size());
        Mat Xtr(n_tr, X.cols()); Vec ytr(n_tr);
        for (int i = 0; i < n_tr; ++i) { Xtr.row(i) = X.row(train_idx[i]); ytr(i) = y(train_idx[i]); }
        Mat Xva(n_va, X.cols()); Vec yva(n_va);
        for (int i = 0; i < n_va; ++i) { Xva.row(i) = X.row(test_idx[i]); yva(i) = y(test_idx[i]); }
        Vec x_mean = Xtr.colwise().mean();
        double y_mean = ytr.mean();
        Mat Xc = Xtr.rowwise() - x_mean.transpose();
        Vec yc = ytr.array() - y_mean;
        Mat Xva_c = Xva.rowwise() - x_mean.transpose();
        for (int b = 0; b < n_ops; ++b) {
            SimplsResult res = simpls_materialized_global(Xc, yc, *bank[b], K);
            for (int k = 1; k <= res.n_components; ++k) {
                Vec coef_k = res.coef_prefix(k);
                Vec pred = Xva_c * coef_k;
                pred.array() += y_mean;
                Vec err = pred - yva;
                double rmse = std::sqrt(err.squaredNorm() / static_cast<double>(n_va));
                if (std::isfinite(rmse)) per_fold[b][f][k - 1] = rmse;
            }
        }
    }
    // np.mean semantics: if any fold is +inf at prefix k, the mean is +inf.
    for (int b = 0; b < n_ops; ++b) {
        for (int k = 0; k < K; ++k) {
            double acc = 0.0;
            bool any_inf = false;
            for (int f = 0; f < n_folds; ++f) {
                const double v = per_fold[b][f][k];
                if (!std::isfinite(v)) { any_inf = true; break; }
                acc += v;
            }
            curves[b][k] = any_inf ? kInf : acc / static_cast<double>(n_folds);
        }
    }
    return curves;
}

// Pick (operator, n_components) by argmin over the score surface, then optionally
// apply the one-SE rule using the winning curve's prefix-std as SE.
inline SelectionDiagnostics select_global_from_curves(
    const std::vector<std::vector<double>>& curves, int max_components, bool one_se_rule) {
    SelectionDiagnostics diag;
    diag.rmse_curves = curves;
    const int n_ops = static_cast<int>(curves.size());
    const int K = max_components;
    int best_b = 0, best_k = 1;
    double best_score = std::numeric_limits<double>::infinity();
    for (int b = 0; b < n_ops; ++b) {
        for (int k = 1; k <= K; ++k) {
            const double sc = curves[b][k - 1];
            if (std::isfinite(sc) && sc < best_score) {
                best_score = sc;
                best_b = b;
                best_k = k;
            }
        }
    }
    diag.selected_operator_index = best_b;
    diag.selected_n_components = best_k;
    diag.best_score = best_score;

    if (one_se_rule && std::isfinite(best_score)) {
        // SE = std(winning_curve, ddof=1) / sqrt(K_finite). Threshold = best + SE.
        const std::vector<double>& curve_b = curves[best_b];
        std::vector<double> finite;
        finite.reserve(curve_b.size());
        for (double v : curve_b) if (std::isfinite(v)) finite.push_back(v);
        if (finite.size() >= 2) {
            double mean = 0.0;
            for (double v : finite) mean += v;
            mean /= static_cast<double>(finite.size());
            double var = 0.0;
            for (double v : finite) var += (v - mean) * (v - mean);
            var /= static_cast<double>(finite.size() - 1);  // ddof=1
            const double se = std::sqrt(var) / std::sqrt(static_cast<double>(finite.size()));
            const double threshold = best_score + se;
            // Find smallest k, then smallest b within threshold.
            int simple_b = best_b;
            int simple_k = best_k;
            for (int b = 0; b < n_ops; ++b) {
                for (int k = 1; k <= best_k; ++k) {  // only k <= current best_k
                    const double sc = curves[b][k - 1];
                    if (!std::isfinite(sc) || sc > threshold) continue;
                    if (k < simple_k || (k == simple_k && b < simple_b)) {
                        simple_b = b;
                        simple_k = k;
                    }
                }
            }
            if (simple_k < best_k || simple_b != best_b) {
                diag.one_se_applied = true;
                diag.selected_operator_index = simple_b;
                diag.selected_n_components = simple_k;
                diag.best_score = curves[simple_b][simple_k - 1];
            }
        }
    }
    return diag;
}

}  // namespace aompls
