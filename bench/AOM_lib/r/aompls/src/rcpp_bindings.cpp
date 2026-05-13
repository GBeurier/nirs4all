// SPDX-License-Identifier: CeCILL-2.1
// Rcpp binding for aompls::fit / predict (PLS1).

#include <Rcpp.h>

#include "aompls/aom_pls.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace {

aompls::CVMode parse_cv_mode(const std::string& name) {
    if (name == "kfold")    return aompls::CVMode::KFOLD;
    if (name == "spxy")     return aompls::CVMode::SPXY;
    if (name == "holdout")  return aompls::CVMode::HOLDOUT;
    if (name == "external") return aompls::CVMode::EXTERNAL;
    Rcpp::stop("unknown cv_mode: " + name);
    return aompls::CVMode::KFOLD;
}

aompls::Preproc parse_preproc(const std::string& name) {
    if (name == "none")      return aompls::Preproc::NONE;
    if (name == "snv")       return aompls::Preproc::SNV;
    if (name == "msc")       return aompls::Preproc::MSC;
    if (name == "osc")       return aompls::Preproc::OSC;
    if (name == "asls")      return aompls::Preproc::ASLS;
    if (name == "snv+osc")   return aompls::Preproc::SNV_OSC;
    if (name == "asls+osc")  return aompls::Preproc::ASLS_OSC;
    Rcpp::stop("unknown preproc: " + name);
    return aompls::Preproc::NONE;
}

std::vector<double> matrix_to_row_major(const Rcpp::NumericMatrix& X) {
    const int n = X.nrow();
    const int p = X.ncol();
    std::vector<double> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) out[static_cast<std::size_t>(i) * p + j] = X(i, j);
    return out;
}

}  // namespace

// [[Rcpp::export]]
Rcpp::List aompls_fit_cpp(Rcpp::NumericMatrix X,
                          Rcpp::NumericVector y,
                          int max_components,
                          int n_folds,
                          std::string cv_mode,
                          bool one_se_rule,
                          bool center,
                          double random_state,
                          std::string preproc,
                          int osc_n_components,
                          double asls_lam,
                          double asls_p,
                          int asls_n_iter,
                          Rcpp::Nullable<Rcpp::List> external_folds) {
    const int n = X.nrow();
    const int p = X.ncol();
    if (y.size() != n) Rcpp::stop("y length must match X rows");
    std::vector<double> Xrm = matrix_to_row_major(X);
    std::vector<double> yvec(y.begin(), y.end());

    aompls::AOMConfig cfg;
    cfg.max_components = max_components;
    cfg.n_folds = n_folds;
    cfg.cv_mode = parse_cv_mode(cv_mode);
    cfg.one_se_rule = one_se_rule;
    cfg.center = center;
    cfg.random_state = static_cast<std::uint64_t>(random_state);
    cfg.preproc = parse_preproc(preproc);
    cfg.osc_n_components = osc_n_components;
    cfg.asls = aompls::AslsParams{asls_lam, asls_p, asls_n_iter};
    if (cfg.cv_mode == aompls::CVMode::EXTERNAL) {
        if (external_folds.isNull())
            Rcpp::stop("cv_mode='external' requires external_folds");
        Rcpp::List folds_list(external_folds);
        cfg.external_folds.reserve(folds_list.size());
        for (R_xlen_t k = 0; k < folds_list.size(); ++k) {
            Rcpp::IntegerVector iv(folds_list[k]);
            cfg.external_folds.emplace_back(iv.begin(), iv.end());
        }
        cfg.n_folds = static_cast<int>(cfg.external_folds.size());
    }
    aompls::AOMResult res = aompls::fit(Xrm.data(),
                                        static_cast<std::size_t>(n),
                                        static_cast<std::size_t>(p),
                                        yvec.data(), cfg);

    Rcpp::List out;
    out["n_features"] = res.n_features;
    out["n_components_selected"] = res.n_components_selected;
    out["selected_operator_index"] = res.selected_operator_index;
    out["selected_operator_name"] = res.selected_operator_name;
    out["bank_names"] = Rcpp::wrap(res.bank_names);
    out["coef"] = Rcpp::NumericVector(res.coef.data(), res.coef.data() + res.coef.size());
    out["intercept"] = res.intercept;
    out["x_mean"] = Rcpp::NumericVector(res.x_mean.data(), res.x_mean.data() + res.x_mean.size());
    out["y_mean"] = res.y_mean;
    out["preproc_kind"] = static_cast<int>(res.preproc.kind);
    if (res.preproc.kind == aompls::Preproc::MSC) {
        out["msc_reference"] = Rcpp::NumericVector(res.preproc.msc_reference.data(),
                                                   res.preproc.msc_reference.data() + res.preproc.msc_reference.size());
    }
    if (res.preproc.osc_k > 0) {
        const int rows = static_cast<int>(res.preproc.osc_W.rows());
        const int cols = res.preproc.osc_k;
        Rcpp::NumericMatrix W(rows, cols), P(rows, cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                W(i, j) = res.preproc.osc_W(i, j);
                P(i, j) = res.preproc.osc_P(i, j);
            }
        out["osc_W"] = W;
        out["osc_P"] = P;
        out["osc_k"] = cols;
    }
    if (!res.rmse_curves.empty()) {
        const int nb = static_cast<int>(res.rmse_curves.size());
        const int K = static_cast<int>(res.rmse_curves[0].size());
        Rcpp::NumericMatrix curves(nb, K);
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < K; ++j) curves(i, j) = res.rmse_curves[i][j];
        out["rmse_curves"] = curves;
    }
    Rcpp::List folds_out(res.fold_indices.size());
    for (std::size_t k = 0; k < res.fold_indices.size(); ++k)
        folds_out[k] = Rcpp::IntegerVector(res.fold_indices[k].begin(), res.fold_indices[k].end());
    out["fold_indices"] = folds_out;
    out["one_se_applied"] = res.one_se_applied;
    out["fit_time_s"] = res.fit_time_s;
    return out;
}

// [[Rcpp::export]]
Rcpp::NumericVector aompls_predict_cpp(Rcpp::List model, Rcpp::NumericMatrix X) {
    const int n = X.nrow();
    const int p = X.ncol();
    const int model_p = Rcpp::as<int>(model["n_features"]);
    if (p != model_p) Rcpp::stop("X.ncol() mismatches model$n_features");
    std::vector<double> Xrm = matrix_to_row_major(X);

    aompls::AOMResult m;
    m.n_features = model_p;
    m.n_components_selected = Rcpp::as<int>(model["n_components_selected"]);
    m.selected_operator_index = Rcpp::as<int>(model["selected_operator_index"]);
    m.selected_operator_name = Rcpp::as<std::string>(model["selected_operator_name"]);
    m.bank_names = Rcpp::as<std::vector<std::string>>(model["bank_names"]);
    Rcpp::NumericVector coef_r(model["coef"]);
    m.coef = aompls::Vec(coef_r.size());
    for (R_xlen_t i = 0; i < coef_r.size(); ++i) m.coef(i) = coef_r[i];
    m.intercept = Rcpp::as<double>(model["intercept"]);
    Rcpp::NumericVector xm_r(model["x_mean"]);
    m.x_mean = aompls::Vec(xm_r.size());
    for (R_xlen_t i = 0; i < xm_r.size(); ++i) m.x_mean(i) = xm_r[i];
    m.y_mean = Rcpp::as<double>(model["y_mean"]);
    m.preproc.kind = static_cast<aompls::Preproc>(Rcpp::as<int>(model["preproc_kind"]));
    if (model.containsElementNamed("msc_reference")) {
        Rcpp::NumericVector mr(model["msc_reference"]);
        m.preproc.msc_reference = aompls::Vec(mr.size());
        for (R_xlen_t i = 0; i < mr.size(); ++i) m.preproc.msc_reference(i) = mr[i];
    }
    if (model.containsElementNamed("osc_W")) {
        Rcpp::NumericMatrix W(model["osc_W"]);
        Rcpp::NumericMatrix Pmat(model["osc_P"]);
        const int rows = W.nrow();
        const int cols = W.ncol();
        m.preproc.osc_W = aompls::Mat(rows, cols);
        m.preproc.osc_P = aompls::Mat(rows, cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                m.preproc.osc_W(i, j) = W(i, j);
                m.preproc.osc_P(i, j) = Pmat(i, j);
            }
        m.preproc.osc_k = cols;
    }

    std::vector<double> out_buf(static_cast<std::size_t>(n));
    aompls::predict(m, Xrm.data(), static_cast<std::size_t>(n), out_buf.data());
    return Rcpp::NumericVector(out_buf.begin(), out_buf.end());
}
