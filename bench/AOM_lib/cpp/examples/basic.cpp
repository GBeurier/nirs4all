// SPDX-License-Identifier: CeCILL-2.1
// Minimal C++ example: synthetic regression, fit, predict, print metrics.

#include "aompls/aom_pls.hpp"

#include <cstdio>
#include <random>
#include <vector>

int main() {
    const std::size_t n = 200;
    const std::size_t p = 256;

    // Synthetic spectra: simple Gaussian-shaped data + linear y on a few features.
    std::vector<double> X(n * p, 0.0);
    std::vector<double> y(n, 0.0);
    std::mt19937 rng(42);
    std::normal_distribution<double> noise(0.0, 0.02);
    for (std::size_t i = 0; i < n; ++i) {
        const double bump = 0.5 + 0.1 * static_cast<double>(i % 5);
        for (std::size_t j = 0; j < p; ++j) {
            const double t = (j - p / 2.0) / 32.0;
            X[i * p + j] = bump * std::exp(-t * t) + noise(rng);
        }
        y[i] = bump;
    }

    aompls::AOMConfig cfg;
    cfg.max_components = 10;
    cfg.cv_mode = aompls::CVMode::KFOLD;
    cfg.n_folds = 5;
    cfg.preproc = aompls::Preproc::SNV;

    aompls::AOMResult m = aompls::fit(X.data(), n, p, y.data(), cfg);
    std::printf("Selected: %s (idx %d), k=%d\n",
                m.selected_operator_name.c_str(),
                m.selected_operator_index,
                m.n_components_selected);
    std::printf("Fit time: %.3f s\n", m.fit_time_s);

    std::vector<double> pred(n);
    aompls::predict(m, X.data(), n, pred.data());
    double sse = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double d = pred[i] - y[i];
        sse += d * d;
    }
    std::printf("Training RMSE: %.4f\n", std::sqrt(sse / static_cast<double>(n)));
    return 0;
}
