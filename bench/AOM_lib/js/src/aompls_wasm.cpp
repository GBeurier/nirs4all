// SPDX-License-Identifier: CeCILL-2.1
// Emscripten / Embind bindings for aompls. Produces a single .wasm + .js bundle
// that runs in both Node.js and the browser.
//
// Build (requires emsdk activated, https://emscripten.org/docs/getting_started/downloads.html):
//   emcc -std=c++17 -O3 -DEIGEN_NO_DEBUG -DEIGEN_DONT_PARALLELIZE \
//        -I ../../cpp/include src/aompls_wasm.cpp -o dist/aompls.mjs \
//        -lembind -sMODULARIZE=1 -sEXPORT_ES6=1 -sENVIRONMENT=node,web \
//        -sALLOW_MEMORY_GROWTH=1

#include <emscripten/bind.h>

#include "aompls/aom_pls.hpp"

#include <stdexcept>
#include <string>
#include <vector>

using namespace emscripten;

namespace {

aompls::CVMode parse_cv_mode(const std::string& s) {
    if (s == "kfold")    return aompls::CVMode::KFOLD;
    if (s == "spxy")     return aompls::CVMode::SPXY;
    if (s == "holdout")  return aompls::CVMode::HOLDOUT;
    if (s == "external") return aompls::CVMode::EXTERNAL;
    throw std::invalid_argument("unknown cv_mode: " + s);
}

aompls::Preproc parse_preproc(const std::string& s) {
    if (s == "none")     return aompls::Preproc::NONE;
    if (s == "snv")      return aompls::Preproc::SNV;
    if (s == "msc")      return aompls::Preproc::MSC;
    if (s == "osc")      return aompls::Preproc::OSC;
    if (s == "asls")     return aompls::Preproc::ASLS;
    if (s == "snv+osc")  return aompls::Preproc::SNV_OSC;
    if (s == "asls+osc") return aompls::Preproc::ASLS_OSC;
    throw std::invalid_argument("unknown preproc: " + s);
}

// JS-friendly config struct (everything as plain values; Embind handles
// JS → struct field assignment).
struct JsConfig {
    int max_components = 15;
    int n_folds = 5;
    std::string cv_mode = "kfold";
    bool one_se_rule = false;
    bool center = true;
    unsigned long long random_state = 0;
    std::string preproc = "none";
    int osc_n_components = 1;
    double asls_lam = 1e5;
    double asls_p = 0.01;
    int asls_n_iter = 10;
    std::vector<std::vector<int>> external_folds;
};

struct JsResult {
    int n_features = 0;
    int n_components_selected = 0;
    int selected_operator_index = 0;
    std::string selected_operator_name;
    std::vector<std::string> bank_names;
    std::vector<double> coef;
    double intercept = 0.0;
    std::vector<double> x_mean;
    double y_mean = 0.0;
    std::string preproc_kind;
    std::vector<double> msc_reference;
    std::vector<double> osc_W_flat;  // row-major (p × k)
    std::vector<double> osc_P_flat;
    int osc_k = 0;
    std::vector<std::vector<double>> rmse_curves;
    std::vector<std::vector<int>> fold_indices;
    bool one_se_applied = false;
    double fit_time_s = 0.0;
};

JsConfig default_config() { return JsConfig{}; }

JsResult result_to_js(const aompls::AOMResult& res) {
    JsResult out;
    out.n_features = res.n_features;
    out.n_components_selected = res.n_components_selected;
    out.selected_operator_index = res.selected_operator_index;
    out.selected_operator_name = res.selected_operator_name;
    out.bank_names = res.bank_names;
    out.coef.assign(res.coef.data(), res.coef.data() + res.coef.size());
    out.intercept = res.intercept;
    out.x_mean.assign(res.x_mean.data(), res.x_mean.data() + res.x_mean.size());
    out.y_mean = res.y_mean;
    switch (res.preproc.kind) {
        case aompls::Preproc::NONE: out.preproc_kind = "none"; break;
        case aompls::Preproc::SNV: out.preproc_kind = "snv"; break;
        case aompls::Preproc::MSC: out.preproc_kind = "msc"; break;
        case aompls::Preproc::OSC: out.preproc_kind = "osc"; break;
        case aompls::Preproc::ASLS: out.preproc_kind = "asls"; break;
        case aompls::Preproc::SNV_OSC: out.preproc_kind = "snv+osc"; break;
        case aompls::Preproc::ASLS_OSC: out.preproc_kind = "asls+osc"; break;
    }
    if (res.preproc.msc_reference.size() > 0)
        out.msc_reference.assign(res.preproc.msc_reference.data(),
                                 res.preproc.msc_reference.data() + res.preproc.msc_reference.size());
    out.osc_k = res.preproc.osc_k;
    if (res.preproc.osc_k > 0) {
        const int rows = static_cast<int>(res.preproc.osc_W.rows());
        const int cols = res.preproc.osc_k;
        out.osc_W_flat.resize(static_cast<std::size_t>(rows) * cols);
        out.osc_P_flat.resize(static_cast<std::size_t>(rows) * cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                out.osc_W_flat[static_cast<std::size_t>(i) * cols + j] = res.preproc.osc_W(i, j);
                out.osc_P_flat[static_cast<std::size_t>(i) * cols + j] = res.preproc.osc_P(i, j);
            }
    }
    out.rmse_curves = res.rmse_curves;
    out.fold_indices = res.fold_indices;
    out.one_se_applied = res.one_se_applied;
    out.fit_time_s = res.fit_time_s;
    return out;
}

aompls::AOMResult js_to_result(const JsResult& js_model) {
    aompls::AOMResult m;
    m.n_features = js_model.n_features;
    m.n_components_selected = js_model.n_components_selected;
    m.selected_operator_index = js_model.selected_operator_index;
    m.selected_operator_name = js_model.selected_operator_name;
    m.bank_names = js_model.bank_names;
    m.coef = aompls::Vec(js_model.coef.size());
    std::copy(js_model.coef.begin(), js_model.coef.end(), m.coef.data());
    m.intercept = js_model.intercept;
    m.x_mean = aompls::Vec(js_model.x_mean.size());
    std::copy(js_model.x_mean.begin(), js_model.x_mean.end(), m.x_mean.data());
    m.y_mean = js_model.y_mean;
    m.preproc.kind = parse_preproc(js_model.preproc_kind);
    if (!js_model.msc_reference.empty()) {
        m.preproc.msc_reference = aompls::Vec(js_model.msc_reference.size());
        std::copy(js_model.msc_reference.begin(), js_model.msc_reference.end(),
                  m.preproc.msc_reference.data());
    }
    if (js_model.osc_k > 0 && !js_model.osc_W_flat.empty()) {
        const int rows = static_cast<int>(m.n_features);
        const int cols = js_model.osc_k;
        m.preproc.osc_W = aompls::Mat(rows, cols);
        m.preproc.osc_P = aompls::Mat(rows, cols);
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) {
                m.preproc.osc_W(i, j) = js_model.osc_W_flat[static_cast<std::size_t>(i) * cols + j];
                m.preproc.osc_P(i, j) = js_model.osc_P_flat[static_cast<std::size_t>(i) * cols + j];
            }
        m.preproc.osc_k = cols;
    }
    return m;
}

JsResult fit(const std::vector<double>& X_flat, int n, int p,
             const std::vector<double>& y, const JsConfig& js_cfg) {
    if (static_cast<int>(X_flat.size()) != n * p)
        throw std::invalid_argument("X length must equal n * p");
    if (static_cast<int>(y.size()) != n)
        throw std::invalid_argument("y length must equal n");
    aompls::AOMConfig cfg;
    cfg.max_components = js_cfg.max_components;
    cfg.n_folds = js_cfg.n_folds;
    cfg.cv_mode = parse_cv_mode(js_cfg.cv_mode);
    cfg.one_se_rule = js_cfg.one_se_rule;
    cfg.center = js_cfg.center;
    cfg.random_state = js_cfg.random_state;
    cfg.preproc = parse_preproc(js_cfg.preproc);
    cfg.osc_n_components = js_cfg.osc_n_components;
    cfg.asls = aompls::AslsParams{js_cfg.asls_lam, js_cfg.asls_p, js_cfg.asls_n_iter};
    if (cfg.cv_mode == aompls::CVMode::EXTERNAL) {
        cfg.external_folds = js_cfg.external_folds;
        cfg.n_folds = static_cast<int>(cfg.external_folds.size());
    }
    aompls::AOMResult res = aompls::fit(X_flat.data(),
                                        static_cast<std::size_t>(n),
                                        static_cast<std::size_t>(p),
                                        y.data(), cfg);
    return result_to_js(res);
}

std::vector<double> predict(const JsResult& js_model, const std::vector<double>& X_flat, int n) {
    aompls::AOMResult m = js_to_result(js_model);
    const int p = m.n_features;
    if (static_cast<int>(X_flat.size()) != n * p)
        throw std::invalid_argument("X length must equal n * model.n_features");
    std::vector<double> out(static_cast<std::size_t>(n));
    aompls::predict(m, X_flat.data(), static_cast<std::size_t>(n), out.data());
    return out;
}

}  // namespace

EMSCRIPTEN_BINDINGS(aompls_module) {
    register_vector<double>("VectorDouble");
    register_vector<int>("VectorInt");
    register_vector<std::string>("VectorString");
    register_vector<std::vector<double>>("VectorVectorDouble");
    register_vector<std::vector<int>>("VectorVectorInt");

    value_object<JsConfig>("AOMConfig")
        .field("max_components", &JsConfig::max_components)
        .field("n_folds", &JsConfig::n_folds)
        .field("cv_mode", &JsConfig::cv_mode)
        .field("one_se_rule", &JsConfig::one_se_rule)
        .field("center", &JsConfig::center)
        .field("random_state", &JsConfig::random_state)
        .field("preproc", &JsConfig::preproc)
        .field("osc_n_components", &JsConfig::osc_n_components)
        .field("asls_lam", &JsConfig::asls_lam)
        .field("asls_p", &JsConfig::asls_p)
        .field("asls_n_iter", &JsConfig::asls_n_iter)
        .field("external_folds", &JsConfig::external_folds);

    value_object<JsResult>("AOMResult")
        .field("n_features", &JsResult::n_features)
        .field("n_components_selected", &JsResult::n_components_selected)
        .field("selected_operator_index", &JsResult::selected_operator_index)
        .field("selected_operator_name", &JsResult::selected_operator_name)
        .field("bank_names", &JsResult::bank_names)
        .field("coef", &JsResult::coef)
        .field("intercept", &JsResult::intercept)
        .field("x_mean", &JsResult::x_mean)
        .field("y_mean", &JsResult::y_mean)
        .field("preproc_kind", &JsResult::preproc_kind)
        .field("msc_reference", &JsResult::msc_reference)
        .field("osc_W_flat", &JsResult::osc_W_flat)
        .field("osc_P_flat", &JsResult::osc_P_flat)
        .field("osc_k", &JsResult::osc_k)
        .field("rmse_curves", &JsResult::rmse_curves)
        .field("fold_indices", &JsResult::fold_indices)
        .field("one_se_applied", &JsResult::one_se_applied)
        .field("fit_time_s", &JsResult::fit_time_s);

    function("defaultConfig", &default_config);
    function("fit", &fit);
    function("predict", &predict);
}
