// SPDX-License-Identifier: CeCILL-2.1
// Top-level AOM-PLS (compact, PLS1) API.
//
// fit() orchestrates: optional preprocessing (fit + transform) → centering →
// AOM global selection over the compact bank → refit on the full training data
// with the chosen operator and k → store coef + intercept + preproc state.
//
// predict() replays preprocessing → centering → linear projection.

#pragma once

#include "aompls/eigen_alias.hpp"
#include "aompls/operators.hpp"
#include "aompls/preproc.hpp"
#include "aompls/selection.hpp"
#include "aompls/simpls.hpp"
#include "aompls/splits.hpp"

#include <chrono>
#include <cstring>
#include <string>
#include <vector>

namespace aompls {

enum class CVMode : int { KFOLD = 0, SPXY = 1, HOLDOUT = 2, EXTERNAL = 3 };

struct AOMConfig {
    int max_components = 15;
    int n_folds = 5;
    CVMode cv_mode = CVMode::KFOLD;
    bool one_se_rule = false;
    bool center = true;
    std::uint64_t random_state = 0;
    Preproc preproc = Preproc::NONE;
    int osc_n_components = 1;
    AslsParams asls;                       // only when preproc includes ASLS
    FoldPartition external_folds;          // used iff cv_mode == EXTERNAL
};

struct AOMResult {
    AOMConfig config_used;
    int n_features = 0;
    int n_components_selected = 0;
    int selected_operator_index = 0;
    std::string selected_operator_name;
    std::vector<std::string> bank_names;
    Vec coef;                              // (p,) for PLS1
    double intercept = 0.0;
    Vec x_mean;                            // (p,)
    double y_mean = 0.0;
    PreprocState preproc;
    std::vector<std::vector<double>> rmse_curves;  // (n_ops, max_components)
    FoldPartition fold_indices;            // actual partition used
    bool one_se_applied = false;
    double fit_time_s = 0.0;
};

namespace detail {

inline FoldPartition build_fold_partition(const Mat& X_pre, const Vec& y, const AOMConfig& cfg) {
    switch (cfg.cv_mode) {
        case CVMode::KFOLD: return kfold(static_cast<int>(X_pre.rows()), cfg.n_folds, cfg.random_state);
        case CVMode::SPXY:  return spxy_fold(X_pre, y, cfg.n_folds);
        case CVMode::HOLDOUT: {
            const int n = static_cast<int>(X_pre.rows());
            const int n_val = std::max(3, n / 5);
            LcgRng rng(cfg.random_state);
            std::vector<int> perm(n);
            for (int i = 0; i < n; ++i) perm[i] = i;
            for (int i = n - 1; i > 0; --i) std::swap(perm[i], perm[rng.randint(i + 1)]);
            FoldPartition fp(1);
            fp[0].assign(perm.begin(), perm.begin() + n_val);
            return fp;
        }
        case CVMode::EXTERNAL:
            if (cfg.external_folds.empty())
                throw std::invalid_argument("cv_mode=EXTERNAL but external_folds is empty");
            return cfg.external_folds;
    }
    throw std::logic_error("unknown CVMode");
}

}  // namespace detail

inline AOMResult fit(const double* X_ptr, std::size_t n, std::size_t p,
                     const double* y_ptr, const AOMConfig& cfg) {
    auto t0 = std::chrono::steady_clock::now();
    if (n == 0 || p == 0) throw std::invalid_argument("X must be non-empty");
    // The public API documents row-major input (X[i*p + j] = X(i, j)). Mat is
    // column-major internally, so we map the raw pointer as row-major and copy
    // into a column-major Mat (Eigen handles the layout conversion).
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatRM> X_map(X_ptr, static_cast<Idx>(n), static_cast<Idx>(p));
    Eigen::Map<const Vec> y_map(y_ptr, static_cast<Idx>(n));
    Mat X = X_map;     // copy with row-major → column-major conversion
    Vec y = y_map;

    // ---- preprocessing fit/transform on full training data ----
    PreprocState ps;
    fit_preproc(X, y, cfg.preproc, cfg.osc_n_components, cfg.asls, ps);
    apply_preproc(X, ps);

    // ---- compact bank ----
    std::vector<OperatorPtr> bank = make_compact_bank();
    std::vector<std::string> names = compact_bank_names();

    // ---- fold partition ----
    FoldPartition folds = detail::build_fold_partition(X, y, cfg);

    // ---- per-operator per-prefix CV RMSE on (X_pre, y) ----
    std::vector<std::vector<double>> curves =
        score_operator_curves(X, y, bank, folds, cfg.max_components);

    SelectionDiagnostics diag = select_global_from_curves(curves, cfg.max_components, cfg.one_se_rule);

    // ---- refit on full data with selected operator, take prefix k ----
    Vec x_mean = cfg.center ? Vec(X.colwise().mean().transpose()) : Vec::Zero(X.cols());
    double y_mean = cfg.center ? y.mean() : 0.0;
    Mat Xc = X.rowwise() - x_mean.transpose();
    Vec yc = y.array() - y_mean;
    // Match Python AOMPLSRegressor: refit with K=best_k (not max_components).
    // This is equivalent to running with max_components and taking coef_prefix(best_k)
    // for the covariance engine, but for the materialized engine the truncation
    // policy on collapse matters — refit at K=best_k stays in line with the reference.
    SimplsResult res = simpls_materialized_global(
        Xc, yc, *bank[diag.selected_operator_index], diag.selected_n_components);
    int k = std::min(diag.selected_n_components, res.n_components);
    if (k < 1) k = std::max(1, res.n_components);
    Vec coef = (k > 0) ? res.coef_prefix(k) : Vec::Zero(X.cols());
    double intercept = y_mean - x_mean.dot(coef);

    AOMResult out;
    out.config_used = cfg;
    out.n_features = static_cast<int>(p);
    out.n_components_selected = k;
    out.selected_operator_index = diag.selected_operator_index;
    out.selected_operator_name = names[diag.selected_operator_index];
    out.bank_names = names;
    out.coef = coef;
    out.intercept = intercept;
    out.x_mean = x_mean;
    out.y_mean = y_mean;
    out.preproc = ps;
    out.rmse_curves = curves;
    out.fold_indices = folds;
    out.one_se_applied = diag.one_se_applied;
    auto t1 = std::chrono::steady_clock::now();
    out.fit_time_s = std::chrono::duration<double>(t1 - t0).count();
    return out;
}

inline void predict(const AOMResult& model, const double* X_ptr, std::size_t n,
                    double* out_ptr) {
    using MatRM = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    Eigen::Map<const MatRM> X_map(X_ptr, static_cast<Idx>(n), static_cast<Idx>(model.n_features));
    Mat X = X_map;
    apply_preproc(X, model.preproc);
    Eigen::Map<Vec> out_vec(out_ptr, static_cast<Idx>(n));
    out_vec = X * model.coef;
    out_vec.array() += model.intercept;
}

}  // namespace aompls
