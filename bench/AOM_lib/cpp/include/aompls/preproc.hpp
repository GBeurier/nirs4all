// SPDX-License-Identifier: CeCILL-2.1
// One-shot preprocessing wrappers applied BEFORE the AOM operator bank.
//
// - SNV   : per-row (centered + std-normalised). Stateless.
// - MSC   : Multiplicative Scatter Correction. State: training mean spectrum.
// - OSC   : Wold-1998 Orthogonal Signal Correction. State: W, P (p x k).
// - ASLS  : Asymmetric Least Squares baseline correction (Eilers & Boelens 2005).
//           Per-spectrum baseline subtraction; stateless except for hyperparams.
//
// Combination policy: SNV+OSC (snv first, then osc fit on snv-transformed) and
// ASLS+OSC are supported.

#pragma once

#include "aompls/eigen_alias.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace aompls {

enum class Preproc : int {
    NONE = 0,
    SNV = 1,
    MSC = 2,
    OSC = 3,
    ASLS = 4,
    SNV_OSC = 5,
    ASLS_OSC = 6,
};

struct AslsParams {
    double lam = 1e5;
    double p = 0.01;
    int n_iter = 10;
};

struct PreprocState {
    Preproc kind = Preproc::NONE;
    Vec msc_reference;          // (p,) when MSC active
    Mat osc_W;                  // (p, k) when OSC active
    Mat osc_P;                  // (p, k) when OSC active
    int osc_k = 0;
    AslsParams asls_params;     // hyperparameters echoed back when ASLS active
    bool has_snv = false;
    bool has_asls = false;
};

namespace detail {

inline void apply_snv_inplace(Mat& X) {
    for (Idx i = 0; i < X.rows(); ++i) {
        double mean = X.row(i).mean();
        Vec centered = X.row(i).transpose().array() - mean;
        double std = std::sqrt(centered.squaredNorm() / static_cast<double>(X.cols()));
        if (std < 1e-12) std = 1.0;
        X.row(i) = centered.transpose() / std;
    }
}

inline void apply_msc_inplace(Mat& X, const Vec& reference) {
    // For each row x, regress x on reference: x ≈ a*ref + b*1.
    // Replace x with (x - b) / a.
    const Idx p = X.cols();
    const double ref_mean = reference.mean();
    const Vec ref_centered = reference.array() - ref_mean;
    const double ref_sq = ref_centered.squaredNorm();
    if (ref_sq < 1e-12) return;  // degenerate reference
    for (Idx i = 0; i < X.rows(); ++i) {
        double x_mean = X.row(i).mean();
        Vec x_centered = X.row(i).transpose().array() - x_mean;
        double a = x_centered.dot(ref_centered) / ref_sq;
        if (std::abs(a) < 1e-12) a = 1.0;
        double b = x_mean - a * ref_mean;
        X.row(i) = ((X.row(i).array() - b) / a).matrix();
    }
}

inline void apply_osc_inplace(Mat& X, const Mat& W, const Mat& P) {
    // X <- X - X W (P^T W)^{-1} P^T
    Mat XW = X * W;                       // (n, k)
    Mat PtW = P.transpose() * W;          // (k, k)
    Mat inv = PtW.completeOrthogonalDecomposition().pseudoInverse();
    X.noalias() -= XW * inv * P.transpose();
}

inline Vec asls_baseline(const Vec& x, double lam, double p, int n_iter) {
    // Solves iteratively (W + lam D^T D) z = W x with
    //   w_i = p   if x_i > z_i, else 1 - p
    // D is the (m-2) x m second-difference matrix.
    const Idx m = x.size();
    if (m < 3) return x;
    Vec w = Vec::Ones(m);
    Vec z = x;
    // Build D^T D once: D[i, i:i+3] = [1, -2, 1] for i in [0, m-3]. D^T D is
    // a pentadiagonal symmetric positive-semi-definite matrix.
    // Diagonals of D^T D (size m):
    //   main diag:  [1, 5, 6, 6, ..., 6, 5, 1]
    //   off-diag 1: [-2, -4, -4, ..., -4, -2]
    //   off-diag 2: [1, 1, ..., 1]
    Vec d0 = Vec::Constant(m, 6.0);
    d0(0) = 1.0; d0(1) = 5.0;
    d0(m - 1) = 1.0; d0(m - 2) = 5.0;
    Vec d1 = Vec::Constant(m - 1, -4.0);
    d1(0) = -2.0; d1(m - 2) = -2.0;
    Vec d2 = Vec::Constant(m - 2, 1.0);

    for (int it = 0; it < n_iter; ++it) {
        // Build banded pentadiagonal A = diag(w) + lam * D^T D as dense (m x m).
        // For m up to a few thousand this is OK; switch to a banded solver if too slow.
        Mat A = Mat::Zero(m, m);
        for (Idx i = 0; i < m; ++i) {
            A(i, i) = w(i) + lam * d0(i);
        }
        for (Idx i = 0; i < m - 1; ++i) {
            const double v = lam * d1(i);
            A(i, i + 1) = v;
            A(i + 1, i) = v;
        }
        for (Idx i = 0; i < m - 2; ++i) {
            const double v = lam * d2(i);
            A(i, i + 2) = v;
            A(i + 2, i) = v;
        }
        Vec rhs = w.array() * x.array();
        Eigen::LLT<Mat> llt(A);
        if (llt.info() != Eigen::Success) {
            // Fall back to LDLT for ill-conditioned cases.
            Eigen::LDLT<Mat> ldlt(A);
            z = ldlt.solve(rhs);
        } else {
            z = llt.solve(rhs);
        }
        // Update weights.
        for (Idx i = 0; i < m; ++i) w(i) = x(i) > z(i) ? p : (1.0 - p);
    }
    return z;
}

inline void apply_asls_inplace(Mat& X, const AslsParams& cfg) {
    for (Idx i = 0; i < X.rows(); ++i) {
        Vec row = X.row(i).transpose();
        Vec base = asls_baseline(row, cfg.lam, cfg.p, cfg.n_iter);
        X.row(i) = (row - base).transpose();
    }
}

// Fit Wold-1998 OSC (k components).
inline void fit_osc(const Mat& X_in, const Vec& y, int k, Mat& W_out, Mat& P_out) {
    Mat X = X_in;
    const Idx p = X.cols();
    W_out = Mat::Zero(p, k);
    P_out = Mat::Zero(p, k);
    for (int a = 0; a < k; ++a) {
        // 1) PCA: pick the dominant right singular vector t of X.
        Eigen::JacobiSVD<Mat> svd(X, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Vec t = svd.matrixU().col(0) * svd.singularValues()(0);
        // 2) Orthogonalise t against y: t* = t - (y^T t / y^T y) y.
        const double yty = y.squaredNorm();
        if (yty < 1e-12) break;
        t = t - (y.dot(t) / yty) * y;
        const double t_norm = t.norm();
        if (t_norm < 1e-12) break;
        t /= t_norm;
        // 3) w = X^T t / (t^T t)
        Vec w = X.transpose() * t;
        const double w_norm = w.norm();
        if (w_norm < 1e-12) break;
        w /= w_norm;
        // 4) Recompute t = X w; p_load = X^T t / (t^T t)
        Vec t2 = X * w;
        const double tt = t2.squaredNorm();
        if (tt < 1e-12) break;
        Vec p_load = X.transpose() * t2 / tt;
        W_out.col(a) = w;
        P_out.col(a) = p_load;
        // Deflate X.
        X.noalias() -= t2 * p_load.transpose();
    }
}

}  // namespace detail

// Fit preprocessing in place: populates `state` from (X, y). Does NOT mutate X.
inline void fit_preproc(const Mat& X, const Vec& y, Preproc kind, int osc_k,
                        const AslsParams& asls, PreprocState& state) {
    state = PreprocState{};
    state.kind = kind;
    state.asls_params = asls;
    switch (kind) {
        case Preproc::NONE: return;
        case Preproc::SNV: state.has_snv = true; return;
        case Preproc::MSC: state.msc_reference = X.colwise().mean().transpose(); return;
        case Preproc::OSC: {
            state.osc_k = std::max(1, osc_k);
            detail::fit_osc(X, y, state.osc_k, state.osc_W, state.osc_P);
            return;
        }
        case Preproc::ASLS: state.has_asls = true; return;
        case Preproc::SNV_OSC: {
            state.has_snv = true;
            Mat Xtmp = X; detail::apply_snv_inplace(Xtmp);
            state.osc_k = std::max(1, osc_k);
            detail::fit_osc(Xtmp, y, state.osc_k, state.osc_W, state.osc_P);
            return;
        }
        case Preproc::ASLS_OSC: {
            state.has_asls = true;
            Mat Xtmp = X; detail::apply_asls_inplace(Xtmp, asls);
            state.osc_k = std::max(1, osc_k);
            detail::fit_osc(Xtmp, y, state.osc_k, state.osc_W, state.osc_P);
            return;
        }
    }
}

// Replay preprocessing on a (possibly new) X in place.
inline void apply_preproc(Mat& X, const PreprocState& state) {
    switch (state.kind) {
        case Preproc::NONE: return;
        case Preproc::SNV: detail::apply_snv_inplace(X); return;
        case Preproc::MSC: detail::apply_msc_inplace(X, state.msc_reference); return;
        case Preproc::OSC: detail::apply_osc_inplace(X, state.osc_W, state.osc_P); return;
        case Preproc::ASLS: detail::apply_asls_inplace(X, state.asls_params); return;
        case Preproc::SNV_OSC:
            detail::apply_snv_inplace(X);
            detail::apply_osc_inplace(X, state.osc_W, state.osc_P);
            return;
        case Preproc::ASLS_OSC:
            detail::apply_asls_inplace(X, state.asls_params);
            detail::apply_osc_inplace(X, state.osc_W, state.osc_P);
            return;
    }
}

}  // namespace aompls
