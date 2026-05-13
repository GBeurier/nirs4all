// SPDX-License-Identifier: CeCILL-2.1
// pybind11 binding for aompls::fit / predict (PLS1).

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "aompls/aom_pls.hpp"

#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

aompls::CVMode parse_cv_mode(const std::string& name) {
    if (name == "kfold")    return aompls::CVMode::KFOLD;
    if (name == "spxy")     return aompls::CVMode::SPXY;
    if (name == "holdout")  return aompls::CVMode::HOLDOUT;
    if (name == "external") return aompls::CVMode::EXTERNAL;
    throw std::invalid_argument("unknown cv_mode: " + name);
}

aompls::Preproc parse_preproc(const std::string& name) {
    if (name == "none")      return aompls::Preproc::NONE;
    if (name == "snv")       return aompls::Preproc::SNV;
    if (name == "msc")       return aompls::Preproc::MSC;
    if (name == "osc")       return aompls::Preproc::OSC;
    if (name == "asls")      return aompls::Preproc::ASLS;
    if (name == "snv+osc")   return aompls::Preproc::SNV_OSC;
    if (name == "asls+osc")  return aompls::Preproc::ASLS_OSC;
    throw std::invalid_argument("unknown preproc: " + name);
}

py::dict result_to_dict(const aompls::AOMResult& res) {
    py::dict out;
    out["n_features"] = res.n_features;
    out["n_components_selected"] = res.n_components_selected;
    out["selected_operator_index"] = res.selected_operator_index;
    out["selected_operator_name"] = res.selected_operator_name;
    out["bank_names"] = res.bank_names;
    py::array_t<double> coef_arr(res.coef.size());
    std::memcpy(coef_arr.mutable_data(), res.coef.data(), sizeof(double) * static_cast<std::size_t>(res.coef.size()));
    out["coef"] = coef_arr;
    out["intercept"] = res.intercept;
    py::array_t<double> x_mean_arr(res.x_mean.size());
    std::memcpy(x_mean_arr.mutable_data(), res.x_mean.data(), sizeof(double) * static_cast<std::size_t>(res.x_mean.size()));
    out["x_mean"] = x_mean_arr;
    out["y_mean"] = res.y_mean;
    out["preproc_kind"] = static_cast<int>(res.preproc.kind);
    if (res.preproc.kind == aompls::Preproc::MSC) {
        py::array_t<double> arr(res.preproc.msc_reference.size());
        std::memcpy(arr.mutable_data(), res.preproc.msc_reference.data(),
                    sizeof(double) * static_cast<std::size_t>(res.preproc.msc_reference.size()));
        out["msc_reference"] = arr;
    }
    if (res.preproc.osc_k > 0) {
        const int p = static_cast<int>(res.preproc.osc_W.rows());
        const int k = res.preproc.osc_k;
        py::array_t<double> W_arr({p, k});
        py::array_t<double> P_arr({p, k});
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < p; ++i) {
                W_arr.mutable_at(i, j) = res.preproc.osc_W(i, j);
                P_arr.mutable_at(i, j) = res.preproc.osc_P(i, j);
            }
        }
        out["osc_W"] = W_arr;
        out["osc_P"] = P_arr;
        out["osc_k"] = k;
    }
    // rmse_curves as a 2D ndarray
    if (!res.rmse_curves.empty()) {
        const int nb = static_cast<int>(res.rmse_curves.size());
        const int K = static_cast<int>(res.rmse_curves[0].size());
        py::array_t<double> curves({nb, K});
        for (int i = 0; i < nb; ++i)
            for (int j = 0; j < K; ++j)
                curves.mutable_at(i, j) = res.rmse_curves[i][j];
        out["rmse_curves"] = curves;
    }
    out["fold_indices"] = res.fold_indices;
    out["one_se_applied"] = res.one_se_applied;
    out["fit_time_s"] = res.fit_time_s;
    return out;
}

py::dict py_fit(py::array_t<double, py::array::c_style | py::array::forcecast> X,
                py::array_t<double, py::array::c_style | py::array::forcecast> y,
                int max_components, int n_folds, const std::string& cv_mode,
                bool one_se_rule, bool center, std::uint64_t random_state,
                const std::string& preproc, int osc_n_components,
                double asls_lam, double asls_p, int asls_n_iter,
                py::object external_folds_obj) {
    if (X.ndim() != 2) throw std::invalid_argument("X must be 2D");
    if (y.ndim() != 1) throw std::invalid_argument("y must be 1D");
    const std::size_t n = static_cast<std::size_t>(X.shape(0));
    const std::size_t p = static_cast<std::size_t>(X.shape(1));
    if (static_cast<std::size_t>(y.shape(0)) != n)
        throw std::invalid_argument("y length must match X rows");

    aompls::AOMConfig cfg;
    cfg.max_components = max_components;
    cfg.n_folds = n_folds;
    cfg.cv_mode = parse_cv_mode(cv_mode);
    cfg.one_se_rule = one_se_rule;
    cfg.center = center;
    cfg.random_state = random_state;
    cfg.preproc = parse_preproc(preproc);
    cfg.osc_n_components = osc_n_components;
    cfg.asls = aompls::AslsParams{asls_lam, asls_p, asls_n_iter};
    if (cfg.cv_mode == aompls::CVMode::EXTERNAL) {
        if (external_folds_obj.is_none())
            throw std::invalid_argument("cv_mode='external' requires external_folds");
        auto outer = external_folds_obj.cast<std::vector<std::vector<int>>>();
        cfg.external_folds = std::move(outer);
        cfg.n_folds = static_cast<int>(cfg.external_folds.size());
    }
    aompls::AOMResult res = aompls::fit(X.data(), n, p, y.data(), cfg);
    return result_to_dict(res);
}

py::array_t<double> py_predict(py::dict model_dict,
                               py::array_t<double, py::array::c_style | py::array::forcecast> X) {
    if (X.ndim() != 2) throw std::invalid_argument("X must be 2D");
    const std::size_t n = static_cast<std::size_t>(X.shape(0));
    const std::size_t p = static_cast<std::size_t>(X.shape(1));

    aompls::AOMResult model;
    model.n_features = model_dict["n_features"].cast<int>();
    if (static_cast<std::size_t>(model.n_features) != p)
        throw std::invalid_argument("X.shape[1] mismatches model.n_features");
    model.n_components_selected = model_dict["n_components_selected"].cast<int>();
    model.selected_operator_index = model_dict["selected_operator_index"].cast<int>();
    model.selected_operator_name = model_dict["selected_operator_name"].cast<std::string>();
    model.bank_names = model_dict["bank_names"].cast<std::vector<std::string>>();

    auto coef_arr = model_dict["coef"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    model.coef = aompls::Vec(coef_arr.shape(0));
    std::memcpy(model.coef.data(), coef_arr.data(), sizeof(double) * static_cast<std::size_t>(coef_arr.shape(0)));
    model.intercept = model_dict["intercept"].cast<double>();
    auto xm_arr = model_dict["x_mean"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    model.x_mean = aompls::Vec(xm_arr.shape(0));
    std::memcpy(model.x_mean.data(), xm_arr.data(), sizeof(double) * static_cast<std::size_t>(xm_arr.shape(0)));
    model.y_mean = model_dict["y_mean"].cast<double>();
    model.preproc.kind = static_cast<aompls::Preproc>(model_dict["preproc_kind"].cast<int>());
    if (model_dict.contains("msc_reference")) {
        auto a = model_dict["msc_reference"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        model.preproc.msc_reference = aompls::Vec(a.shape(0));
        std::memcpy(model.preproc.msc_reference.data(), a.data(), sizeof(double) * static_cast<std::size_t>(a.shape(0)));
    }
    if (model_dict.contains("osc_W")) {
        auto W = model_dict["osc_W"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        auto Pmat = model_dict["osc_P"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
        const int rows = static_cast<int>(W.shape(0));
        const int cols = static_cast<int>(W.shape(1));
        model.preproc.osc_W = aompls::Mat(rows, cols);
        model.preproc.osc_P = aompls::Mat(rows, cols);
        for (int j = 0; j < cols; ++j)
            for (int i = 0; i < rows; ++i) {
                model.preproc.osc_W(i, j) = W.at(i, j);
                model.preproc.osc_P(i, j) = Pmat.at(i, j);
            }
        model.preproc.osc_k = cols;
    }

    py::array_t<double> out(n);
    aompls::predict(model, X.data(), n, out.mutable_data());
    return out;
}

}  // namespace

PYBIND11_MODULE(_binding, m) {
    m.doc() = "AOM-PLS (compact, PLS1) — C++ core via pybind11";
    m.def("fit", &py_fit,
          py::arg("X"), py::arg("y"),
          py::kw_only(),
          py::arg("max_components") = 15,
          py::arg("n_folds") = 5,
          py::arg("cv_mode") = std::string("kfold"),
          py::arg("one_se_rule") = false,
          py::arg("center") = true,
          py::arg("random_state") = static_cast<std::uint64_t>(0),
          py::arg("preproc") = std::string("none"),
          py::arg("osc_n_components") = 1,
          py::arg("asls_lam") = 1e5,
          py::arg("asls_p") = 0.01,
          py::arg("asls_n_iter") = 10,
          py::arg("external_folds") = py::none());
    m.def("predict", &py_predict, py::arg("model"), py::arg("X"));
}
