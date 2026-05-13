// SPDX-License-Identifier: CeCILL-2.1
// MATLAB MEX wrapper for aompls::fit / predict.
//
// Build (from this directory):
//     mex -I../cpp/include CXXFLAGS='$CXXFLAGS -std=c++17 -O3 -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE' aompls_mex.cpp
//
// MATLAB usage:
//     model = aompls_mex('fit', X, y, opts)
//     pred  = aompls_mex('predict', model, Xnew)
//
//   X    : numeric matrix (n x p), MATLAB column-major; we transpose for the
//          row-major C++ API.
//   y    : numeric column vector (n x 1).
//   opts : struct with optional fields max_components, n_folds, cv_mode,
//          one_se_rule, center, random_state, preproc, osc_n_components,
//          asls (struct lam/p/n_iter), external_folds (cell of int vectors).
//   model: struct with fields coef, intercept, x_mean, y_mean, n_features,
//          n_components_selected, selected_operator_index, selected_operator_name,
//          bank_names (cellstr), rmse_curves (n_ops x K), preproc_kind,
//          msc_reference / osc_W / osc_P / osc_k (when applicable).

#include "mex.h"
#include "matrix.h"

#include "aompls/aom_pls.hpp"

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

aompls::CVMode parse_cv_mode(const std::string& s) {
    if (s == "kfold")    return aompls::CVMode::KFOLD;
    if (s == "spxy")     return aompls::CVMode::SPXY;
    if (s == "holdout")  return aompls::CVMode::HOLDOUT;
    if (s == "external") return aompls::CVMode::EXTERNAL;
    mexErrMsgIdAndTxt("aompls:cv_mode", ("unknown cv_mode: " + s).c_str());
    return aompls::CVMode::KFOLD;
}

aompls::Preproc parse_preproc(const std::string& s) {
    if (s == "none")     return aompls::Preproc::NONE;
    if (s == "snv")      return aompls::Preproc::SNV;
    if (s == "msc")      return aompls::Preproc::MSC;
    if (s == "osc")      return aompls::Preproc::OSC;
    if (s == "asls")     return aompls::Preproc::ASLS;
    if (s == "snv+osc")  return aompls::Preproc::SNV_OSC;
    if (s == "asls+osc") return aompls::Preproc::ASLS_OSC;
    mexErrMsgIdAndTxt("aompls:preproc", ("unknown preproc: " + s).c_str());
    return aompls::Preproc::NONE;
}

std::string get_string(const mxArray* a) {
    if (!a || !mxIsChar(a)) mexErrMsgIdAndTxt("aompls:type", "expected string");
    char buf[256];
    mxGetString(a, buf, sizeof(buf));
    return std::string(buf);
}

double get_double_field(const mxArray* opts, const char* name, double dflt) {
    if (!opts || !mxIsStruct(opts)) return dflt;
    const mxArray* f = mxGetField(opts, 0, name);
    if (!f) return dflt;
    return mxGetScalar(f);
}

int get_int_field(const mxArray* opts, const char* name, int dflt) {
    return static_cast<int>(get_double_field(opts, name, static_cast<double>(dflt)));
}

bool get_bool_field(const mxArray* opts, const char* name, bool dflt) {
    if (!opts || !mxIsStruct(opts)) return dflt;
    const mxArray* f = mxGetField(opts, 0, name);
    if (!f) return dflt;
    return mxGetScalar(f) != 0.0;
}

std::string get_string_field(const mxArray* opts, const char* name, const std::string& dflt) {
    if (!opts || !mxIsStruct(opts)) return dflt;
    const mxArray* f = mxGetField(opts, 0, name);
    if (!f) return dflt;
    return get_string(f);
}

// MATLAB stores matrices column-major. We transpose into a row-major buffer.
std::vector<double> matlab_to_row_major(const mxArray* X) {
    if (!mxIsDouble(X) || mxGetNumberOfDimensions(X) != 2)
        mexErrMsgIdAndTxt("aompls:type", "X must be a 2D double matrix");
    const mwSize n = mxGetM(X);
    const mwSize p = mxGetN(X);
    const double* src = mxGetPr(X);
    std::vector<double> out(static_cast<std::size_t>(n) * static_cast<std::size_t>(p));
    for (mwSize j = 0; j < p; ++j)
        for (mwSize i = 0; i < n; ++i)
            out[i * p + j] = src[j * n + i];
    return out;
}

mxArray* vec_to_matlab_column(const aompls::Vec& v) {
    mxArray* out = mxCreateDoubleMatrix(v.size(), 1, mxREAL);
    std::memcpy(mxGetPr(out), v.data(), sizeof(double) * static_cast<std::size_t>(v.size()));
    return out;
}

mxArray* mat_to_matlab(const aompls::Mat& m) {
    mxArray* out = mxCreateDoubleMatrix(m.rows(), m.cols(), mxREAL);
    double* dst = mxGetPr(out);
    for (Eigen::Index j = 0; j < m.cols(); ++j)
        for (Eigen::Index i = 0; i < m.rows(); ++i)
            dst[j * m.rows() + i] = m(i, j);
    return out;
}

mxArray* result_to_struct(const aompls::AOMResult& res) {
    const char* fields[] = {"n_features", "n_components_selected",
                            "selected_operator_index", "selected_operator_name",
                            "bank_names", "coef", "intercept", "x_mean", "y_mean",
                            "preproc_kind", "msc_reference",
                            "osc_W", "osc_P", "osc_k",
                            "rmse_curves", "fold_indices",
                            "one_se_applied", "fit_time_s"};
    mxArray* s = mxCreateStructMatrix(1, 1, sizeof(fields) / sizeof(*fields), fields);
    mxSetField(s, 0, "n_features", mxCreateDoubleScalar(res.n_features));
    mxSetField(s, 0, "n_components_selected", mxCreateDoubleScalar(res.n_components_selected));
    mxSetField(s, 0, "selected_operator_index", mxCreateDoubleScalar(res.selected_operator_index));
    mxSetField(s, 0, "selected_operator_name", mxCreateString(res.selected_operator_name.c_str()));
    mxArray* names = mxCreateCellMatrix(1, res.bank_names.size());
    for (std::size_t i = 0; i < res.bank_names.size(); ++i)
        mxSetCell(names, i, mxCreateString(res.bank_names[i].c_str()));
    mxSetField(s, 0, "bank_names", names);
    mxSetField(s, 0, "coef", vec_to_matlab_column(res.coef));
    mxSetField(s, 0, "intercept", mxCreateDoubleScalar(res.intercept));
    mxSetField(s, 0, "x_mean", vec_to_matlab_column(res.x_mean));
    mxSetField(s, 0, "y_mean", mxCreateDoubleScalar(res.y_mean));
    mxSetField(s, 0, "preproc_kind", mxCreateDoubleScalar(static_cast<int>(res.preproc.kind)));
    if (res.preproc.msc_reference.size() > 0)
        mxSetField(s, 0, "msc_reference", vec_to_matlab_column(res.preproc.msc_reference));
    if (res.preproc.osc_k > 0) {
        mxSetField(s, 0, "osc_W", mat_to_matlab(res.preproc.osc_W));
        mxSetField(s, 0, "osc_P", mat_to_matlab(res.preproc.osc_P));
        mxSetField(s, 0, "osc_k", mxCreateDoubleScalar(res.preproc.osc_k));
    }
    if (!res.rmse_curves.empty()) {
        const mwSize nb = res.rmse_curves.size();
        const mwSize K = res.rmse_curves[0].size();
        mxArray* curves = mxCreateDoubleMatrix(nb, K, mxREAL);
        double* dst = mxGetPr(curves);
        for (mwSize j = 0; j < K; ++j)
            for (mwSize i = 0; i < nb; ++i)
                dst[j * nb + i] = res.rmse_curves[i][j];
        mxSetField(s, 0, "rmse_curves", curves);
    }
    mxArray* folds = mxCreateCellMatrix(1, res.fold_indices.size());
    for (std::size_t k = 0; k < res.fold_indices.size(); ++k) {
        mxArray* iv = mxCreateDoubleMatrix(1, res.fold_indices[k].size(), mxREAL);
        double* dst = mxGetPr(iv);
        for (std::size_t i = 0; i < res.fold_indices[k].size(); ++i)
            dst[i] = static_cast<double>(res.fold_indices[k][i]);
        mxSetCell(folds, k, iv);
    }
    mxSetField(s, 0, "fold_indices", folds);
    mxSetField(s, 0, "one_se_applied", mxCreateLogicalScalar(res.one_se_applied));
    mxSetField(s, 0, "fit_time_s", mxCreateDoubleScalar(res.fit_time_s));
    return s;
}

aompls::AOMResult struct_to_result(const mxArray* model_struct) {
    aompls::AOMResult m;
    m.n_features = static_cast<int>(get_double_field(model_struct, "n_features", 0));
    m.n_components_selected = static_cast<int>(get_double_field(model_struct, "n_components_selected", 0));
    m.selected_operator_index = static_cast<int>(get_double_field(model_struct, "selected_operator_index", 0));
    m.selected_operator_name = get_string_field(model_struct, "selected_operator_name", "identity");
    const mxArray* coef = mxGetField(model_struct, 0, "coef");
    if (!coef) mexErrMsgIdAndTxt("aompls:model", "model missing coef");
    m.coef = aompls::Vec(mxGetNumberOfElements(coef));
    std::memcpy(m.coef.data(), mxGetPr(coef), sizeof(double) * mxGetNumberOfElements(coef));
    m.intercept = get_double_field(model_struct, "intercept", 0.0);
    const mxArray* xm = mxGetField(model_struct, 0, "x_mean");
    if (!xm) mexErrMsgIdAndTxt("aompls:model", "model missing x_mean");
    m.x_mean = aompls::Vec(mxGetNumberOfElements(xm));
    std::memcpy(m.x_mean.data(), mxGetPr(xm), sizeof(double) * mxGetNumberOfElements(xm));
    m.y_mean = get_double_field(model_struct, "y_mean", 0.0);
    m.preproc.kind = static_cast<aompls::Preproc>(get_int_field(model_struct, "preproc_kind", 0));
    const mxArray* ref = mxGetField(model_struct, 0, "msc_reference");
    if (ref) {
        m.preproc.msc_reference = aompls::Vec(mxGetNumberOfElements(ref));
        std::memcpy(m.preproc.msc_reference.data(), mxGetPr(ref),
                    sizeof(double) * mxGetNumberOfElements(ref));
    }
    const mxArray* W = mxGetField(model_struct, 0, "osc_W");
    const mxArray* Pmat = mxGetField(model_struct, 0, "osc_P");
    if (W && Pmat) {
        const mwSize rows = mxGetM(W);
        const mwSize cols = mxGetN(W);
        m.preproc.osc_W = aompls::Mat(rows, cols);
        m.preproc.osc_P = aompls::Mat(rows, cols);
        const double* sW = mxGetPr(W);
        const double* sP = mxGetPr(Pmat);
        for (mwSize j = 0; j < cols; ++j)
            for (mwSize i = 0; i < rows; ++i) {
                m.preproc.osc_W(i, j) = sW[j * rows + i];
                m.preproc.osc_P(i, j) = sP[j * rows + i];
            }
        m.preproc.osc_k = static_cast<int>(cols);
    }
    return m;
}

void do_fit(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 3) mexErrMsgIdAndTxt("aompls:fit", "fit needs (X, y, [opts])");
    const mxArray* X = prhs[1];
    const mxArray* y = prhs[2];
    const mxArray* opts = (nrhs > 3) ? prhs[3] : nullptr;
    if (!mxIsDouble(X) || mxGetNumberOfDimensions(X) != 2)
        mexErrMsgIdAndTxt("aompls:fit", "X must be a 2D double matrix");
    if (!mxIsDouble(y) || mxGetNumberOfElements(y) != mxGetM(X))
        mexErrMsgIdAndTxt("aompls:fit", "y length must match number of rows in X");
    const std::size_t n = mxGetM(X);
    const std::size_t p = mxGetN(X);
    std::vector<double> Xrm = matlab_to_row_major(X);
    std::vector<double> yvec(mxGetPr(y), mxGetPr(y) + n);

    aompls::AOMConfig cfg;
    cfg.max_components = get_int_field(opts, "max_components", 15);
    cfg.n_folds = get_int_field(opts, "n_folds", 5);
    cfg.cv_mode = parse_cv_mode(get_string_field(opts, "cv_mode", "kfold"));
    cfg.one_se_rule = get_bool_field(opts, "one_se_rule", false);
    cfg.center = get_bool_field(opts, "center", true);
    cfg.random_state = static_cast<std::uint64_t>(get_double_field(opts, "random_state", 0.0));
    cfg.preproc = parse_preproc(get_string_field(opts, "preproc", "none"));
    cfg.osc_n_components = get_int_field(opts, "osc_n_components", 1);
    if (opts) {
        const mxArray* asls = mxGetField(opts, 0, "asls");
        if (asls && mxIsStruct(asls)) {
            cfg.asls.lam = get_double_field(asls, "lam", 1e5);
            cfg.asls.p = get_double_field(asls, "p", 0.01);
            cfg.asls.n_iter = get_int_field(asls, "n_iter", 10);
        }
    }
    if (cfg.cv_mode == aompls::CVMode::EXTERNAL) {
        if (!opts) mexErrMsgIdAndTxt("aompls:fit", "external folds required for cv_mode='external'");
        const mxArray* folds = mxGetField(opts, 0, "external_folds");
        if (!folds || !mxIsCell(folds))
            mexErrMsgIdAndTxt("aompls:fit", "external_folds must be a cell array of int vectors");
        const mwSize nf = mxGetNumberOfElements(folds);
        cfg.external_folds.reserve(nf);
        for (mwSize k = 0; k < nf; ++k) {
            const mxArray* iv = mxGetCell(folds, k);
            if (!mxIsDouble(iv))
                mexErrMsgIdAndTxt("aompls:fit", "external_folds entries must be double vectors");
            const mwSize sz = mxGetNumberOfElements(iv);
            const double* src = mxGetPr(iv);
            std::vector<int> fold(sz);
            for (mwSize i = 0; i < sz; ++i) fold[i] = static_cast<int>(src[i]);
            cfg.external_folds.push_back(std::move(fold));
        }
        cfg.n_folds = static_cast<int>(cfg.external_folds.size());
    }
    try {
        aompls::AOMResult res = aompls::fit(Xrm.data(), n, p, yvec.data(), cfg);
        plhs[0] = result_to_struct(res);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("aompls:fit", e.what());
    }
}

void do_predict(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 3) mexErrMsgIdAndTxt("aompls:predict", "predict needs (model, X)");
    const mxArray* model = prhs[1];
    const mxArray* X = prhs[2];
    if (!mxIsStruct(model)) mexErrMsgIdAndTxt("aompls:predict", "model must be a struct");
    if (!mxIsDouble(X) || mxGetNumberOfDimensions(X) != 2)
        mexErrMsgIdAndTxt("aompls:predict", "X must be a 2D double matrix");
    aompls::AOMResult m = struct_to_result(model);
    const std::size_t n = mxGetM(X);
    const std::size_t p = mxGetN(X);
    if (static_cast<int>(p) != m.n_features)
        mexErrMsgIdAndTxt("aompls:predict", "X.cols mismatches model.n_features");
    std::vector<double> Xrm = matlab_to_row_major(X);
    plhs[0] = mxCreateDoubleMatrix(n, 1, mxREAL);
    aompls::predict(m, Xrm.data(), n, mxGetPr(plhs[0]));
}

}  // namespace

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if (nrhs < 1 || !mxIsChar(prhs[0]))
        mexErrMsgIdAndTxt("aompls:usage", "first argument must be 'fit' or 'predict'");
    const std::string cmd = get_string(prhs[0]);
    if (cmd == "fit") {
        do_fit(nlhs, plhs, nrhs, prhs);
    } else if (cmd == "predict") {
        do_predict(nlhs, plhs, nrhs, prhs);
    } else {
        mexErrMsgIdAndTxt("aompls:usage", ("unknown command: " + cmd).c_str());
    }
}
