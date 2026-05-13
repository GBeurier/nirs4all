// SPDX-License-Identifier: CeCILL-2.1
// Phase 4 parity gate (KFold path) — compares the C++ fit/predict against
// the JSON fixtures exported from Python AOM_v0 by scripts/export_reference.py.
//
// Asserts:
//   - same selected operator (name + index)
//   - same n_components_selected
//   - max |coef_cpp - coef_python| <= 1e-8
//   - max |intercept_cpp - intercept_python| <= 1e-8
//   - max |predict(X_train)_cpp - predict_python| <= 1e-8 (relative to y scale)
//   - rmse_curves match to <= 1e-9 where both finite
//
// Does NOT compare Z / P / Q / R — those carry sign ambiguity from the dominant
// singular direction.

#include "aompls/aom_pls.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_fails = 0;

void check(bool cond, const std::string& msg) {
    if (!cond) { std::fprintf(stderr, "FAIL: %s\n", msg.c_str()); ++g_fails; }
}

// Minimal JSON-ish loader: we read the file once into a single std::string and
// extract the leaf primitives we need by key. JSON shape is deterministic
// (single-flat object per cv_label) so we can locate values via key markers.
//
// We rely on the small set of keys we know the exporter emits; we never have
// to recurse into nested structures except for `fold_test_indices`, `coef`,
// `rmse_curves`, `bank_names`, `predictions_train`, and `X` / `y` arrays.
struct JsonDoc {
    std::string text;
    static JsonDoc from_file(const std::string& path) {
        std::ifstream in(path);
        if (!in) throw std::runtime_error("cannot open " + path);
        std::stringstream ss; ss << in.rdbuf();
        return JsonDoc{ss.str()};
    }
};

// Find the start of the value for the *first* occurrence of `"key":` at or
// after `from`. Returns the position just after the colon.
std::size_t value_start(const std::string& s, const std::string& key, std::size_t from = 0) {
    const std::string needle = "\"" + key + "\":";
    std::size_t pos = s.find(needle, from);
    if (pos == std::string::npos) throw std::runtime_error("key not found: " + key);
    return pos + needle.size();
}

// Skip whitespace from position p.
std::size_t skip_ws(const std::string& s, std::size_t p) {
    while (p < s.size() && (s[p] == ' ' || s[p] == '\t' || s[p] == '\n' || s[p] == '\r')) ++p;
    return p;
}

// Parse a JSON number from position p; returns (value, end_pos_just_after).
std::pair<double, std::size_t> parse_number(const std::string& s, std::size_t p) {
    char* endp = nullptr;
    double v = std::strtod(s.c_str() + p, &endp);
    return {v, static_cast<std::size_t>(endp - s.c_str())};
}

// Parse a JSON string at p (expects `"..."`); returns (text, end_pos_after).
std::pair<std::string, std::size_t> parse_string(const std::string& s, std::size_t p) {
    p = skip_ws(s, p);
    if (s[p] != '"') throw std::runtime_error("expected '\"' at pos " + std::to_string(p));
    std::size_t q = p + 1;
    std::string out;
    while (q < s.size() && s[q] != '"') { out.push_back(s[q]); ++q; }
    return {out, q + 1};
}

// Parse a flat JSON array of doubles [v0, v1, ...] starting at `p`.
// Returns the vector and the end position (after the closing bracket).
std::pair<std::vector<double>, std::size_t> parse_array_double(const std::string& s, std::size_t p) {
    p = skip_ws(s, p);
    if (s[p] != '[') throw std::runtime_error("expected '[' at pos " + std::to_string(p));
    ++p;
    std::vector<double> out;
    while (true) {
        p = skip_ws(s, p);
        if (s[p] == ']') { ++p; break; }
        auto [v, q] = parse_number(s, p);
        out.push_back(v);
        p = skip_ws(s, q);
        if (s[p] == ',') ++p;
    }
    return {out, p};
}

// Parse a flat JSON array of ints.
std::pair<std::vector<int>, std::size_t> parse_array_int(const std::string& s, std::size_t p) {
    auto [vd, q] = parse_array_double(s, p);
    std::vector<int> vi(vd.size());
    for (std::size_t i = 0; i < vd.size(); ++i) vi[i] = static_cast<int>(vd[i]);
    return {vi, q};
}

// Parse a nested JSON array of doubles (e.g. X as [[..], [..], ...]).
// Returns flattened row-major vector and the row/col dimensions.
std::tuple<std::vector<double>, std::size_t, std::size_t, std::size_t>
parse_array_2d_double(const std::string& s, std::size_t p) {
    p = skip_ws(s, p);
    if (s[p] != '[') throw std::runtime_error("expected '[' (outer) at pos " + std::to_string(p));
    ++p;
    std::vector<double> flat;
    std::size_t rows = 0, cols = 0;
    while (true) {
        p = skip_ws(s, p);
        if (s[p] == ']') { ++p; break; }
        auto [row, q] = parse_array_double(s, p);
        if (rows == 0) cols = row.size();
        else if (row.size() != cols) throw std::runtime_error("ragged 2D array");
        flat.insert(flat.end(), row.begin(), row.end());
        ++rows;
        p = skip_ws(s, q);
        if (s[p] == ',') ++p;
    }
    return {flat, rows, cols, p};
}

// Parse a nested JSON array of ints.
std::pair<std::vector<std::vector<int>>, std::size_t>
parse_array_2d_int(const std::string& s, std::size_t p) {
    p = skip_ws(s, p);
    if (s[p] != '[') throw std::runtime_error("expected '[' (outer) at pos " + std::to_string(p));
    ++p;
    std::vector<std::vector<int>> out;
    while (true) {
        p = skip_ws(s, p);
        if (s[p] == ']') { ++p; break; }
        auto [row, q] = parse_array_int(s, p);
        out.push_back(std::move(row));
        p = skip_ws(s, q);
        if (s[p] == ',') ++p;
    }
    return {out, p};
}

// Parse a JSON array of strings.
std::pair<std::vector<std::string>, std::size_t>
parse_array_string(const std::string& s, std::size_t p) {
    p = skip_ws(s, p);
    if (s[p] != '[') throw std::runtime_error("expected '[' (strings) at pos " + std::to_string(p));
    ++p;
    std::vector<std::string> out;
    while (true) {
        p = skip_ws(s, p);
        if (s[p] == ']') { ++p; break; }
        auto [str, q] = parse_string(s, p);
        out.push_back(std::move(str));
        p = skip_ws(s, q);
        if (s[p] == ',') ++p;
    }
    return {out, p};
}

struct ReferenceCase {
    int n;
    int p;
    std::vector<double> X;             // row-major (n, p)
    std::vector<double> y;             // (n,)
    std::vector<std::vector<int>> folds;
    std::vector<std::string> bank_names;
    int selected_operator_index;
    std::string selected_operator_name;
    int n_components_selected;
    std::vector<double> coef;          // (p,)
    double intercept;
    std::vector<double> predictions;
    std::vector<std::vector<double>> rmse_curves;  // (n_ops, K)
};

// Locate the start of the `kfold5` object value and parse all required leaves
// from it.
ReferenceCase load_case(const JsonDoc& doc, const std::string& case_name) {
    const std::string& s = doc.text;
    const std::string label = "\"" + case_name + "\":";
    std::size_t case_pos = s.find(label);
    if (case_pos == std::string::npos)
        throw std::runtime_error("case not found: " + case_name);
    ReferenceCase out;
    {
        auto [n, _] = parse_number(s, value_start(s, "n", case_pos));
        out.n = static_cast<int>(n);
    }
    {
        auto [p, _] = parse_number(s, value_start(s, "p", case_pos));
        out.p = static_cast<int>(p);
    }
    {
        auto [flat, rows, cols, _] = parse_array_2d_double(s, value_start(s, "X", case_pos));
        if (static_cast<int>(rows) != out.n || static_cast<int>(cols) != out.p)
            throw std::runtime_error("X shape mismatch");
        out.X = std::move(flat);
    }
    {
        auto [vy, _] = parse_array_double(s, value_start(s, "y", case_pos));
        out.y = std::move(vy);
    }
    {
        auto [folds, _] = parse_array_2d_int(s, value_start(s, "fold_test_indices", case_pos));
        out.folds = std::move(folds);
    }
    {
        auto [bn, _] = parse_array_string(s, value_start(s, "bank_names", case_pos));
        out.bank_names = std::move(bn);
    }
    {
        auto [idx, _] = parse_number(s, value_start(s, "selected_operator_index", case_pos));
        out.selected_operator_index = static_cast<int>(idx);
    }
    {
        auto [name, _] = parse_string(s, value_start(s, "selected_operator_name", case_pos));
        out.selected_operator_name = std::move(name);
    }
    {
        auto [k, _] = parse_number(s, value_start(s, "n_components_selected", case_pos));
        out.n_components_selected = static_cast<int>(k);
    }
    {
        auto [coef, _] = parse_array_double(s, value_start(s, "coef", case_pos));
        out.coef = std::move(coef);
    }
    {
        auto [b, _] = parse_number(s, value_start(s, "intercept", case_pos));
        out.intercept = b;
    }
    {
        auto [preds, _] = parse_array_double(s, value_start(s, "predictions_train", case_pos));
        out.predictions = std::move(preds);
    }
    {
        std::size_t pos = value_start(s, "rmse_curves", case_pos);
        auto [flat, rows, cols, _] = parse_array_2d_double(s, pos);
        out.rmse_curves.assign(rows, std::vector<double>(cols));
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t j = 0; j < cols; ++j)
                out.rmse_curves[i][j] = flat[i * cols + j];
    }
    return out;
}

void run_dataset(const std::string& json_path, const std::string& dataset,
                 const std::string& case_name, aompls::CVMode cv_mode, bool one_se_rule) {
    JsonDoc doc = JsonDoc::from_file(json_path);
    ReferenceCase ref = load_case(doc, case_name);

    aompls::AOMConfig cfg;
    cfg.max_components = 15;
    cfg.cv_mode = cv_mode;
    cfg.one_se_rule = one_se_rule;
    cfg.external_folds = ref.folds;
    cfg.n_folds = static_cast<int>(ref.folds.size());

    aompls::AOMResult res = aompls::fit(ref.X.data(),
                                        static_cast<std::size_t>(ref.n),
                                        static_cast<std::size_t>(ref.p),
                                        ref.y.data(), cfg);

    const std::string tag = "[" + dataset + "/" + case_name + "]";
    check(res.selected_operator_name == ref.selected_operator_name,
          tag + " selected_operator_name (cpp='" + res.selected_operator_name + "', py='" +
              ref.selected_operator_name + "')");
    check(res.selected_operator_index == ref.selected_operator_index,
          tag + " selected_operator_index");
    check(res.n_components_selected == ref.n_components_selected,
          tag + " n_components_selected (cpp=" + std::to_string(res.n_components_selected) +
              ", py=" + std::to_string(ref.n_components_selected) + ")");

    double coef_diff = 0.0;
    for (int i = 0; i < ref.p; ++i) coef_diff = std::max(coef_diff, std::abs(res.coef(i) - ref.coef[i]));
    check(coef_diff < 1e-8, tag + " coef max|Δ| (" + std::to_string(coef_diff) + ")");
    check(std::abs(res.intercept - ref.intercept) < 1e-8,
          tag + " intercept (cpp=" + std::to_string(res.intercept) + ", py=" +
              std::to_string(ref.intercept) + ")");

    std::vector<double> preds_cpp(ref.n);
    aompls::predict(res, ref.X.data(), static_cast<std::size_t>(ref.n), preds_cpp.data());
    double pred_diff = 0.0;
    for (int i = 0; i < ref.n; ++i) pred_diff = std::max(pred_diff, std::abs(preds_cpp[i] - ref.predictions[i]));
    check(pred_diff < 1e-8, tag + " predict(X_train) max|Δ| (" + std::to_string(pred_diff) + ")");

    // RMSE curves: only check rows where both sides are finite.
    double curve_diff = 0.0;
    const std::size_t n_ops = std::min(res.rmse_curves.size(), ref.rmse_curves.size());
    for (std::size_t b = 0; b < n_ops; ++b) {
        for (std::size_t k = 0; k < res.rmse_curves[b].size() && k < ref.rmse_curves[b].size(); ++k) {
            const double a = res.rmse_curves[b][k];
            const double r = ref.rmse_curves[b][k];
            if (std::isfinite(a) && std::isfinite(r)) curve_diff = std::max(curve_diff, std::abs(a - r));
        }
    }
    check(curve_diff < 1e-9, tag + " rmse_curves max|Δ| (" + std::to_string(curve_diff) + ")");

    std::printf("%s OK selected=%s k=%d coef|Δ|=%.3e pred|Δ|=%.3e curves|Δ|=%.3e fit=%.3fs\n",
                tag.c_str(), res.selected_operator_name.c_str(), res.n_components_selected,
                coef_diff, pred_diff, curve_diff, res.fit_time_s);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string default_dir = "tests/reference";
    const std::string dir = (argc > 1) ? argv[1] : default_dir;
    for (const std::string& ds : {"BEER", "CORN", "ALPINE"}) {
        const std::string path = dir + "/" + ds + ".json";
        try {
            run_dataset(path, ds, "kfold5",        aompls::CVMode::EXTERNAL, false);
            run_dataset(path, ds, "kfold5_oneSE",  aompls::CVMode::EXTERNAL, true);
            run_dataset(path, ds, "spxy5",         aompls::CVMode::EXTERNAL, false);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "ERROR [%s]: %s\n", ds.c_str(), e.what());
            ++g_fails;
        }
    }
    if (g_fails == 0) {
        std::printf("test_parity_kfold: all checks passed\n");
        return 0;
    }
    std::fprintf(stderr, "test_parity_kfold: %d FAILURE(S)\n", g_fails);
    return 1;
}
