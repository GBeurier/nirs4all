// SPDX-License-Identifier: CeCILL-2.1
// AOM-PLS compact operator bank (9 strict-linear spectral operators).
// Reference: bench/AOM_v0/aompls/operators.py and banks.py:24-41.
#pragma once

#include "aompls/eigen_alias.hpp"

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace aompls {

// ---------------------------------------------------------------------------
// Zero-padded same-length cross-correlation along columns.
//
// Convention (matches operators.py:_xcorr_zero_pad):
//   out[i, t] = sum_{l=0}^{K-1} kernel[l] * X[i, t + l - half]
// where half = (K-1) / 2, and X[i, j] = 0 for j outside [0, p).
//
// Equivalent to applying the Toeplitz matrix `M` with
//   M[t, j] = kernel[j - t + half]  (and 0 outside the band).
// ---------------------------------------------------------------------------
inline Mat xcorr_zero_pad_rows(const Eigen::Ref<const Mat>& X, const Vec& kernel) {
    const Idx K = kernel.size();
    const Idx p = X.cols();
    const Idx n = X.rows();
    if (K % 2 == 0) throw std::invalid_argument("kernel length must be odd");
    const Idx half = (K - 1) / 2;
    Mat out = Mat::Zero(n, p);
    for (Idx l = 0; l < K; ++l) {
        const Idx shift = l - half;
        const double k = kernel(l);
        if (shift == 0) {
            out.noalias() += k * X;
        } else if (shift > 0) {
            // out[:, 0 : p-shift] += k * X[:, shift : p]
            const Idx len = p - shift;
            if (len > 0) {
                out.block(0, 0, n, len).noalias() += k * X.block(0, shift, n, len);
            }
        } else {  // shift < 0
            const Idx s = -shift;
            const Idx len = p - s;
            if (len > 0) {
                out.block(0, s, n, len).noalias() += k * X.block(0, 0, n, len);
            }
        }
    }
    return out;
}

// Same operator applied along columns of S (S has shape (p, q)):
//   (A S)[:, j] = sum_l kernel[l] * S[t + l - half, j]
// Apply along rows of S^T then transpose back.
inline Mat xcorr_zero_pad_cols(const Eigen::Ref<const Mat>& S, const Vec& kernel) {
    return xcorr_zero_pad_rows(S.transpose(), kernel).transpose();
}

inline Vec xcorr_zero_pad_vec(const Vec& v, const Vec& kernel) {
    Mat M(1, v.size());
    M.row(0) = v.transpose();
    return xcorr_zero_pad_rows(M, kernel).row(0).transpose();
}

// ---------------------------------------------------------------------------
// Operator interface
// ---------------------------------------------------------------------------

struct LinearSpectralOperator {
    virtual ~LinearSpectralOperator() = default;

    // Short identifier, persisted in AOMResult::bank_names.
    virtual std::string name() const = 0;

    // Apply A to row spectra: returns X * A^T (shape (n, p)).
    virtual Mat transform(const Eigen::Ref<const Mat>& X) const = 0;

    // Apply A to a covariance matrix: returns A * S (shape (p, q)).
    virtual Mat apply_cov(const Eigen::Ref<const Mat>& S) const = 0;

    // Apply A^T to a vector: returns A^T * v (shape (p,)).
    virtual Vec adjoint_vec(const Vec& v) const = 0;

    // Build the explicit p x p matrix. Used for parity testing only.
    virtual Mat matrix(Idx p) const = 0;
};

using OperatorPtr = std::unique_ptr<LinearSpectralOperator>;

// ---------------------------------------------------------------------------
// Savitzky-Golay coefficients (reference: operators.py:_sg_coefficients).
//
// Returns the `deriv`-th row of pinv(Vandermonde(j, polyorder+1)) scaled by
// factorial(deriv). The Vandermonde uses increasing powers with grid
// j = (-half ... +half).
// ---------------------------------------------------------------------------
inline Vec sg_coefficients(int window_length, int polyorder, int deriv) {
    if (window_length < 3 || (window_length % 2) == 0)
        throw std::invalid_argument("window_length must be an odd integer >= 3");
    if (polyorder < 0 || polyorder >= window_length)
        throw std::invalid_argument("polyorder must be in [0, window_length)");
    if (deriv < 0 || deriv > polyorder)
        throw std::invalid_argument("deriv must be in [0, polyorder]");
    const int half = (window_length - 1) / 2;

    Mat A(window_length, polyorder + 1);
    for (int i = 0; i < window_length; ++i) {
        const double j = static_cast<double>(i - half);
        A(i, 0) = 1.0;
        for (int k = 1; k <= polyorder; ++k) A(i, k) = A(i, k - 1) * j;
    }
    // pinv via BDCSVD (numerically stable, matches np.linalg.pinv to ~1e-12).
    Eigen::BDCSVD<Mat> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const double tol = static_cast<double>(std::max<Idx>(A.rows(), A.cols())) *
                       svd.singularValues()(0) * std::numeric_limits<double>::epsilon();
    Vec sing_inv = svd.singularValues();
    for (Idx i = 0; i < sing_inv.size(); ++i) sing_inv(i) = (sing_inv(i) > tol) ? 1.0 / sing_inv(i) : 0.0;
    Mat pinv = svd.matrixV() * sing_inv.asDiagonal() * svd.matrixU().transpose();
    // Scale row `deriv` by factorial(deriv).
    double fact = 1.0;
    for (int i = 2; i <= deriv; ++i) fact *= static_cast<double>(i);
    return pinv.row(deriv).transpose() * fact;
}

// ---------------------------------------------------------------------------
// IdentityOperator
// ---------------------------------------------------------------------------
class IdentityOperator final : public LinearSpectralOperator {
   public:
    std::string name() const override { return "identity"; }
    Mat transform(const Eigen::Ref<const Mat>& X) const override { return X; }
    Mat apply_cov(const Eigen::Ref<const Mat>& S) const override { return S; }
    Vec adjoint_vec(const Vec& v) const override { return v; }
    Mat matrix(Idx p) const override { return Mat::Identity(p, p); }
};

// ---------------------------------------------------------------------------
// SavitzkyGolayOperator
// ---------------------------------------------------------------------------
class SavitzkyGolayOperator final : public LinearSpectralOperator {
   public:
    SavitzkyGolayOperator(int window_length, int polyorder, int deriv)
        : window_length_(window_length), polyorder_(polyorder), deriv_(deriv) {
        kernel_ = sg_coefficients(window_length, polyorder, deriv);
        kernel_rev_ = kernel_.reverse();
        if (deriv == 0)
            name_ = "sg_smooth_w" + std::to_string(window_length) + "_p" + std::to_string(polyorder);
        else
            name_ = "sg_d" + std::to_string(deriv) + "_w" + std::to_string(window_length) + "_p" +
                    std::to_string(polyorder);
    }
    std::string name() const override { return name_; }
    Mat transform(const Eigen::Ref<const Mat>& X) const override {
        return xcorr_zero_pad_rows(X, kernel_);
    }
    Mat apply_cov(const Eigen::Ref<const Mat>& S) const override {
        return xcorr_zero_pad_cols(S, kernel_);
    }
    Vec adjoint_vec(const Vec& v) const override { return xcorr_zero_pad_vec(v, kernel_rev_); }
    Mat matrix(Idx p) const override {
        // A is the column-action matrix. A = apply_cov(I_p).
        return apply_cov(Mat::Identity(p, p));
    }

   private:
    int window_length_, polyorder_, deriv_;
    Vec kernel_;
    Vec kernel_rev_;
    std::string name_;
};

// ---------------------------------------------------------------------------
// DetrendProjectionOperator: A = I - Q Q^T where Q is the QR-orthonormal
// basis of [1, t, t^2, ..., t^degree] on t = linspace(-1, 1, p).
// Symmetric, so A^T = A.
// ---------------------------------------------------------------------------
class DetrendProjectionOperator final : public LinearSpectralOperator {
   public:
    explicit DetrendProjectionOperator(int degree) : degree_(degree) {
        if (degree < 0) throw std::invalid_argument("degree must be >= 0");
        name_ = "detrend_d" + std::to_string(degree);
    }
    std::string name() const override { return name_; }
    Mat transform(const Eigen::Ref<const Mat>& X) const override {
        // X * A^T = X * A (symmetric).
        ensure_basis(X.cols());
        // X * A = X * (I - Q Q^T) = X - (X * Q) * Q^T
        Mat XQ = X * Q_cached_;
        return X - XQ * Q_cached_.transpose();
    }
    Mat apply_cov(const Eigen::Ref<const Mat>& S) const override {
        ensure_basis(S.rows());
        // A * S = (I - Q Q^T) S = S - Q (Q^T S)
        Mat QtS = Q_cached_.transpose() * S;
        return S - Q_cached_ * QtS;
    }
    Vec adjoint_vec(const Vec& v) const override {
        ensure_basis(v.size());
        Vec Qtv = Q_cached_.transpose() * v;
        return v - Q_cached_ * Qtv;
    }
    Mat matrix(Idx p) const override {
        ensure_basis(p);
        return Mat::Identity(p, p) - Q_cached_ * Q_cached_.transpose();
    }
    int degree() const { return degree_; }

   private:
    void ensure_basis(Idx p) const {
        if (cached_p_ == p && Q_cached_.cols() == degree_ + 1) return;
        if (p < degree_ + 1)
            throw std::invalid_argument(
                "DetrendProjection requires p >= degree + 1 (degree=" + std::to_string(degree_) + ")");
        Mat V(p, degree_ + 1);
        Vec t = Vec::LinSpaced(p, -1.0, 1.0);
        V.col(0).setOnes();
        for (int k = 1; k <= degree_; ++k) V.col(k) = V.col(k - 1).cwiseProduct(t);
        Eigen::HouseholderQR<Mat> qr(V);
        // Thin Q has shape (p, degree+1).
        Mat I_thin = Mat::Identity(p, degree_ + 1);
        Q_cached_ = qr.householderQ() * I_thin;
        cached_p_ = p;
    }
    int degree_;
    mutable Idx cached_p_ = -1;
    mutable Mat Q_cached_;
    std::string name_;
};

// ---------------------------------------------------------------------------
// FiniteDifferenceOperator (centered, order 1: kernel [-0.5, 0, +0.5]).
// ---------------------------------------------------------------------------
class FiniteDifferenceOperator final : public LinearSpectralOperator {
   public:
    explicit FiniteDifferenceOperator(int order) : order_(order) {
        if (order != 1 && order != 2) throw std::invalid_argument("order must be 1 or 2");
        kernel_ = Vec(3);
        if (order == 1) {
            kernel_ << -0.5, 0.0, 0.5;
        } else {
            kernel_ << 1.0, -2.0, 1.0;
        }
        kernel_rev_ = kernel_.reverse();
        name_ = "fd_d" + std::to_string(order);
    }
    std::string name() const override { return name_; }
    Mat transform(const Eigen::Ref<const Mat>& X) const override {
        return xcorr_zero_pad_rows(X, kernel_);
    }
    Mat apply_cov(const Eigen::Ref<const Mat>& S) const override {
        return xcorr_zero_pad_cols(S, kernel_);
    }
    Vec adjoint_vec(const Vec& v) const override { return xcorr_zero_pad_vec(v, kernel_rev_); }
    Mat matrix(Idx p) const override { return apply_cov(Mat::Identity(p, p)); }

   private:
    int order_;
    Vec kernel_;
    Vec kernel_rev_;
    std::string name_;
};

// ---------------------------------------------------------------------------
// Compact bank (9 operators) — order matches banks.py:24-41.
// ---------------------------------------------------------------------------
inline std::vector<OperatorPtr> make_compact_bank() {
    std::vector<OperatorPtr> bank;
    bank.reserve(9);
    bank.emplace_back(std::make_unique<IdentityOperator>());
    bank.emplace_back(std::make_unique<SavitzkyGolayOperator>(11, 2, 0));
    bank.emplace_back(std::make_unique<SavitzkyGolayOperator>(21, 3, 0));
    bank.emplace_back(std::make_unique<SavitzkyGolayOperator>(11, 2, 1));
    bank.emplace_back(std::make_unique<SavitzkyGolayOperator>(21, 3, 1));
    bank.emplace_back(std::make_unique<SavitzkyGolayOperator>(11, 2, 2));
    bank.emplace_back(std::make_unique<DetrendProjectionOperator>(1));
    bank.emplace_back(std::make_unique<DetrendProjectionOperator>(2));
    bank.emplace_back(std::make_unique<FiniteDifferenceOperator>(1));
    return bank;
}

inline std::vector<std::string> compact_bank_names() {
    return {"identity",       "sg_smooth_w11_p2", "sg_smooth_w21_p3",
            "sg_d1_w11_p2",   "sg_d1_w21_p3",     "sg_d2_w11_p2",
            "detrend_d1",     "detrend_d2",       "fd_d1"};
}

}  // namespace aompls
