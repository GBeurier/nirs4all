// SPDX-License-Identifier: CeCILL-2.1
// C ABI implementation. Wraps aompls::fit / predict and exposes a flat C API
// for ccall-style consumers (Julia, MATLAB loadlibrary, etc.).
//
// Build as a shared library:
//   g++ -std=c++17 -O3 -fPIC -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE \
//       -I ../include -shared src/c_api.cpp -o libaompls.so

#include "aompls/c_api.h"
#include "aompls/aom_pls.hpp"

#include <cstdlib>
#include <cstring>
#include <new>
#include <string>
#include <vector>

struct aompls_model {
    aompls::AOMResult res;
};

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.data(), s.size());
    out[s.size()] = '\0';
    return out;
}

aompls::AOMConfig from_c_config(const aompls_config_t& c) {
    aompls::AOMConfig cfg;
    cfg.max_components = c.max_components;
    cfg.n_folds = c.n_folds;
    cfg.cv_mode = static_cast<aompls::CVMode>(c.cv_mode);
    cfg.one_se_rule = c.one_se_rule != 0;
    cfg.center = c.center != 0;
    cfg.random_state = c.random_state;
    cfg.preproc = static_cast<aompls::Preproc>(c.preproc);
    cfg.osc_n_components = c.osc_n_components;
    cfg.asls = aompls::AslsParams{c.asls_lam, c.asls_p, c.asls_n_iter};
    if (cfg.cv_mode == aompls::CVMode::EXTERNAL) {
        cfg.external_folds.reserve(static_cast<std::size_t>(c.n_external_folds));
        const int* cursor = c.external_folds_flat;
        for (int f = 0; f < c.n_external_folds; ++f) {
            const int sz = c.external_fold_sizes[f];
            cfg.external_folds.emplace_back(cursor, cursor + sz);
            cursor += sz;
        }
        cfg.n_folds = c.n_external_folds;
    }
    return cfg;
}

}  // namespace

extern "C" {

void aompls_config_init(aompls_config_t* cfg) {
    if (!cfg) return;
    cfg->max_components = 15;
    cfg->n_folds = 5;
    cfg->cv_mode = 0;
    cfg->one_se_rule = 0;
    cfg->center = 1;
    cfg->random_state = 0;
    cfg->preproc = 0;
    cfg->osc_n_components = 1;
    cfg->asls_lam = 1e5;
    cfg->asls_p = 0.01;
    cfg->asls_n_iter = 10;
    cfg->external_folds_flat = nullptr;
    cfg->external_fold_sizes = nullptr;
    cfg->n_external_folds = 0;
}

aompls_model_t* aompls_fit(const double* X, int n, int p,
                           const double* y, const aompls_config_t* c_cfg,
                           char** err_msg) {
    if (err_msg) *err_msg = nullptr;
    try {
        aompls::AOMConfig cfg = from_c_config(*c_cfg);
        auto* m = new aompls_model_t();
        m->res = aompls::fit(X, static_cast<std::size_t>(n), static_cast<std::size_t>(p), y, cfg);
        return m;
    } catch (const std::exception& e) {
        if (err_msg) *err_msg = dup_cstr(e.what());
        return nullptr;
    } catch (...) {
        if (err_msg) *err_msg = dup_cstr("unknown error in aompls_fit");
        return nullptr;
    }
}

int aompls_predict(const aompls_model_t* model, const double* X, int n,
                   double* out, char** err_msg) {
    if (err_msg) *err_msg = nullptr;
    if (!model) {
        if (err_msg) *err_msg = dup_cstr("model is NULL");
        return -1;
    }
    try {
        aompls::predict(model->res, X, static_cast<std::size_t>(n), out);
        return 0;
    } catch (const std::exception& e) {
        if (err_msg) *err_msg = dup_cstr(e.what());
        return -1;
    } catch (...) {
        if (err_msg) *err_msg = dup_cstr("unknown error in aompls_predict");
        return -1;
    }
}

int aompls_n_features(const aompls_model_t* m) { return m ? m->res.n_features : 0; }
int aompls_n_components(const aompls_model_t* m) { return m ? m->res.n_components_selected : 0; }
int aompls_selected_operator_index(const aompls_model_t* m) {
    return m ? m->res.selected_operator_index : -1;
}
const char* aompls_selected_operator_name(const aompls_model_t* m) {
    return m ? m->res.selected_operator_name.c_str() : "";
}
void aompls_get_coef(const aompls_model_t* m, double* out) {
    if (!m || !out) return;
    std::memcpy(out, m->res.coef.data(), sizeof(double) * static_cast<std::size_t>(m->res.coef.size()));
}
double aompls_get_intercept(const aompls_model_t* m) { return m ? m->res.intercept : 0.0; }
void aompls_get_x_mean(const aompls_model_t* m, double* out) {
    if (!m || !out) return;
    std::memcpy(out, m->res.x_mean.data(), sizeof(double) * static_cast<std::size_t>(m->res.x_mean.size()));
}
double aompls_get_y_mean(const aompls_model_t* m) { return m ? m->res.y_mean : 0.0; }

int aompls_get_rmse_curves(const aompls_model_t* m, int* n_ops, int* n_k, double* out) {
    if (!m || m->res.rmse_curves.empty()) {
        if (n_ops) *n_ops = 0;
        if (n_k) *n_k = 0;
        return 0;
    }
    const int rows = static_cast<int>(m->res.rmse_curves.size());
    const int cols = static_cast<int>(m->res.rmse_curves[0].size());
    if (n_ops) *n_ops = rows;
    if (n_k) *n_k = cols;
    if (out) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) out[i * cols + j] = m->res.rmse_curves[i][j];
    }
    return rows;
}

void aompls_free(aompls_model_t* m) { delete m; }
void aompls_free_string(char* s) { std::free(s); }

}  // extern "C"
