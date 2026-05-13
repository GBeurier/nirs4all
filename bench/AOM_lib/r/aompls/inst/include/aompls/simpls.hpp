// SPDX-License-Identifier: CeCILL-2.1
// Covariance-space SIMPLS for AOM-PLS (PLS1 / univariate y).
//
// Reference: bench/AOM_v0/aompls/simpls.py:308-405 (simpls_covariance).
// We implement the single-fixed-operator path (AOM global): all `K` components
// share the same operator. The coefficient formula
//     B_k = Z[:, :k] (P[:, :k]^T Z[:, :k])^{-1} q[:k]^T
// is exposed via SimplsResult::coef_prefix(k) for the auto-prefix scorer.

#pragma once

#include "aompls/eigen_alias.hpp"
#include "aompls/operators.hpp"

#include <cmath>

namespace aompls {

struct SimplsResult {
    int n_features = 0;
    int n_components_max = 0;
    int n_components = 0;  // requested K (matches Python's NIPALSResult.n_components = Z.shape[1])
    Mat Z;                 // (p, K) original-space effective weights (zero columns for collapsed components)
    Mat P;                 // (p, K) X-loadings
    Vec q;                 // (K,) y-loadings (PLS1)
    Mat T;                 // (n, K) X-scores (kept for diagnostics)
    Mat V;                 // (p, K) orthonormal basis used for covariance deflation

    // Coefficient vector with k components: shape (p,).
    // Mirrors Python's NIPALSResult.coef_prefix(k): uses Moore-Penrose pinv
    // when P^T Z is singular (e.g. when some components collapsed to zero).
    Vec coef_prefix(int k) const {
        if (k <= 0 || n_components == 0) return Vec::Zero(n_features);
        const int kk = std::min(k, n_components);
        const Mat Zk = Z.leftCols(kk);
        const Mat Pk = P.leftCols(kk);
        const Vec qk = q.head(kk);
        Mat PtZ = Pk.transpose() * Zk;  // (kk, kk)
        // Match np.linalg.inv → pinv fallback semantics (operators.py uses pinv only on
        // LinAlgError, so we try a full-pivot LU first).
        Eigen::FullPivLU<Mat> lu(PtZ);
        Mat inv;
        if (lu.isInvertible()) {
            inv = lu.inverse();
        } else {
            Eigen::JacobiSVD<Mat> svd(PtZ, Eigen::ComputeFullU | Eigen::ComputeFullV);
            const double tol = svd.singularValues()(0) * static_cast<double>(kk) *
                               std::numeric_limits<double>::epsilon();
            Vec sv = svd.singularValues();
            for (Idx i = 0; i < sv.size(); ++i) sv(i) = (sv(i) > tol) ? 1.0 / sv(i) : 0.0;
            inv = svd.matrixV() * sv.asDiagonal() * svd.matrixU().transpose();
        }
        return Zk * inv * qk;
    }
};

// Standard SIMPLS on a (possibly transformed) matrix X (n, p) and centered y (n,).
// Mirrors bench/AOM_v0/aompls/simpls.py:53-122 (simpls_standard) including the
// `break` policy on deflation collapse and the truncation to actual_K components.
inline SimplsResult simpls_standard(const Eigen::Ref<const Mat>& X, const Vec& y, int n_components) {
    const Idx n = X.rows();
    const Idx p = X.cols();
    if (y.size() != n) throw std::invalid_argument("y length must match X rows");
    const int K = n_components;
    SimplsResult res;
    res.n_features = static_cast<int>(p);
    res.n_components_max = K;
    res.Z = Mat::Zero(p, K);
    res.P = Mat::Zero(p, K);
    res.q = Vec::Zero(K);
    res.T = Mat::Zero(n, K);
    res.V = Mat::Zero(p, K);

    Mat S(p, 1);
    S.col(0) = X.transpose() * y;
    constexpr double kEps = 1e-14;
    int actual_k = 0;
    for (int a = 0; a < K; ++a) {
        Vec r = S.col(0);                 // PLS1 dominant direction
        Vec t = X * r;                    // (n,)
        const double t_norm = t.norm();
        if (t_norm < kEps) break;
        t /= t_norm;
        r /= t_norm;                      // post-normalisation matches simpls_standard
        Vec p_load = X.transpose() * t;   // (p,)
        const double q_load = y.dot(t);

        Vec v = p_load;
        if (a > 0) {
            Mat Va = res.V.leftCols(a);
            Vec coeffs = Va.transpose() * v;
            v.noalias() -= Va * coeffs;
        }
        const double v_norm = v.norm();
        if (v_norm < kEps) break;
        v /= v_norm;

        S.noalias() -= v * (v.transpose() * S);
        res.Z.col(a) = r;
        res.P.col(a) = p_load;
        res.q(a) = q_load;
        res.T.col(a) = t;
        res.V.col(a) = v;
        ++actual_k;
    }
    // Truncate to actual_k (matches simpls_standard return Z[:, :actual_K]).
    res.n_components = actual_k;
    if (actual_k < K) {
        res.Z.conservativeResize(p, actual_k);
        res.P.conservativeResize(p, actual_k);
        res.q.conservativeResize(actual_k);
        res.T.conservativeResize(n, actual_k);
        res.V.conservativeResize(p, actual_k);
    }
    return res;
}

// Materialized SIMPLS through a single fixed operator with TRANSFORMED-space
// orthogonalisation. Mirrors simpls_materialized_fixed in simpls.py:125-195:
//
//   X_b = X * A^T (via op.transform), run standard SIMPLS on (X_b, y) → res_b,
//   map weights back: Z[:, a] = A^T r_a (op.adjoint_vec); then recompute
//   loadings P, q, scores T, and the orthonormal basis V in the ORIGINAL space.
//
// This is the engine the Python AOMPLSRegressor uses for `selection="global"`
// (orthogonalization auto-resolves to "transformed", which delegates to this).
inline SimplsResult simpls_materialized_global(const Eigen::Ref<const Mat>& Xc,
                                               const Vec& yc,
                                               const LinearSpectralOperator& op,
                                               int n_components_max) {
    const Idx n = Xc.rows();
    const Idx p = Xc.cols();
    // Step 1: materialize the operator on row spectra.
    Mat Xb = op.transform(Xc);                   // (n, p)
    // Step 2: standard SIMPLS on (Xb, yc).
    SimplsResult res_b = simpls_standard(Xb, yc, n_components_max);
    const int K = res_b.n_components;             // actual_K (may be < requested)
    SimplsResult out;
    out.n_features = static_cast<int>(p);
    out.n_components_max = n_components_max;
    out.n_components = K;
    if (K == 0) {
        out.Z = Mat::Zero(p, 0);
        out.P = Mat::Zero(p, 0);
        out.q = Vec::Zero(0);
        out.T = Mat::Zero(n, 0);
        out.V = Mat::Zero(p, 0);
        return out;
    }
    out.Z = Mat::Zero(p, K);
    out.P = Mat::Zero(p, K);
    out.q = Vec::Zero(K);
    out.T = Mat::Zero(n, K);
    out.V = Mat::Zero(p, K);

    // Map transformed-space weights back to original space.
    for (int a = 0; a < K; ++a) {
        out.Z.col(a) = op.adjoint_vec(res_b.Z.col(a));
    }
    // Recompute scores, loadings, and Gram-Schmidt basis V in the ORIGINAL space
    // using SIMPLS-style updates (matches simpls.py:151-185).
    constexpr double kEps = 1e-14;
    for (int a = 0; a < K; ++a) {
        Vec z = out.Z.col(a);
        Vec t = Xc * z;
        const double t_norm = t.norm();
        if (t_norm < kEps) {
            out.P.col(a).setZero();
            out.q(a) = 0.0;
            out.T.col(a).setZero();
            continue;
        }
        t /= t_norm;
        z /= t_norm;
        out.Z.col(a) = z;
        Vec p_load = Xc.transpose() * t;
        const double q_load = yc.dot(t);
        Vec v = p_load;
        if (a > 0) {
            Mat Va = out.V.leftCols(a);
            Vec coeffs = Va.transpose() * v;
            v.noalias() -= Va * coeffs;
        }
        const double v_norm = v.norm();
        if (v_norm < kEps) {
            out.P.col(a).setZero();
            out.q(a) = 0.0;
            out.T.col(a).setZero();
            continue;
        }
        out.V.col(a) = v / v_norm;
        out.T.col(a) = t;
        out.P.col(a) = p_load;
        out.q(a) = q_load;
    }
    return out;
}

// Run covariance-space SIMPLS with a single fixed operator across all components.
// `Xc` must already be centered; `yc` must already be centered (mean removed).
// Returns a result with up to `n_components_max` components (fewer if deflation
// norms collapse below machine epsilon).
inline SimplsResult simpls_global_fixed(const Eigen::Ref<const Mat>& Xc,
                                        const Vec& yc,
                                        const LinearSpectralOperator& op,
                                        int n_components_max) {
    const Idx n = Xc.rows();
    const Idx p = Xc.cols();
    if (yc.size() != n) throw std::invalid_argument("y length must match X rows");
    const int K = n_components_max;
    SimplsResult res;
    res.n_features = static_cast<int>(p);
    res.n_components_max = K;
    res.n_components = K;          // matches Python's NIPALSResult.n_components = Z.shape[1]
    res.Z = Mat::Zero(p, K);
    res.P = Mat::Zero(p, K);
    res.q = Vec::Zero(K);
    res.T = Mat::Zero(n, K);
    res.V = Mat::Zero(p, K);

    // S = X^T y (PLS1: single column).
    Mat S(p, 1);
    S.col(0) = Xc.transpose() * yc;

    constexpr double kEps = 1e-14;
    for (int a = 0; a < K; ++a) {
        // Apply A to the covariance matrix.
        Mat S_b = op.apply_cov(S);                // (p, 1)
        Vec r = S_b.col(0);
        const double r_norm = r.norm();
        if (r_norm < kEps) continue;              // matches Python: skip component but keep K total
        r /= r_norm;

        Vec z = op.adjoint_vec(r);                // (p,)
        Vec t = Xc * z;                           // (n,)
        const double t_norm = t.norm();
        if (t_norm < kEps) continue;
        t /= t_norm;
        z /= t_norm;

        Vec p_load = Xc.transpose() * t;          // (p,)
        const double q_load = yc.dot(t);          // scalar (PLS1)

        // Gram-Schmidt orthogonalisation against prior V[:, :a].
        Vec v = p_load;
        if (a > 0) {
            Mat Va = res.V.leftCols(a);          // (p, a)
            Vec coeffs = Va.transpose() * v;      // (a,)
            v.noalias() -= Va * coeffs;
        }
        const double v_norm = v.norm();
        if (v_norm < kEps) continue;
        v /= v_norm;

        res.V.col(a) = v;
        res.Z.col(a) = z;
        res.P.col(a) = p_load;
        res.q(a) = q_load;
        res.T.col(a) = t;
        // Deflate covariance: S <- (I - v v^T) S
        S.noalias() -= v * (v.transpose() * S);
    }
    return res;
}

}  // namespace aompls
