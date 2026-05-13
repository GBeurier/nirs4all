/* SPDX-License-Identifier: CeCILL-2.1
 * Plain C ABI for aompls::fit / predict (PLS1). Designed for Julia ccall,
 * MATLAB loadlibrary, and any other language with a C FFI. Internally calls
 * the header-only C++ implementation in aompls/aom_pls.hpp.
 *
 * Memory model:
 * - `aompls_fit` returns an opaque model handle (NULL on failure, in which case
 *   `err_msg` points to a heap-allocated message that the caller must free via
 *   `aompls_free_string`).
 * - `aompls_predict` writes n predictions into the caller-provided `out` buffer.
 * - All array inputs are row-major double arrays (`X[i*p + j] = X(i, j)`).
 * - The handle is freed via `aompls_free`.
 */
#ifndef AOMPLS_C_API_H
#define AOMPLS_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct aompls_model aompls_model_t;

/* Mirrors aompls::AOMConfig. Use values directly (no enums in C ABI). */
typedef struct aompls_config_t {
    int max_components;        /* default 15 */
    int n_folds;               /* default 5; ignored when cv_mode == 3 (EXTERNAL) */
    int cv_mode;               /* 0=KFOLD, 1=SPXY, 2=HOLDOUT, 3=EXTERNAL */
    int one_se_rule;           /* 0 or 1 */
    int center;                /* 0 or 1, default 1 */
    unsigned long long random_state;
    int preproc;               /* 0=NONE 1=SNV 2=MSC 3=OSC 4=ASLS 5=SNV_OSC 6=ASLS_OSC */
    int osc_n_components;
    double asls_lam;
    double asls_p;
    int asls_n_iter;
    /* External folds: NULL when cv_mode != 3 (EXTERNAL).
     * Flattened test indices for all folds, with per-fold sizes:
     *   external_folds_flat = [f0_idx_0, f0_idx_1, ..., f1_idx_0, ...]
     *   external_fold_sizes = [size(f0), size(f1), ...]
     */
    const int* external_folds_flat;
    const int* external_fold_sizes;
    int n_external_folds;
} aompls_config_t;

/* Initialise an aompls_config_t to the canonical defaults. */
void aompls_config_init(aompls_config_t* cfg);

/* Fit. X is row-major (n, p) doubles. y is length n. On failure returns NULL
 * and sets *err_msg to a malloc'd string (caller must free via
 * aompls_free_string). When cv_mode is EXTERNAL the external_folds_flat and
 * external_fold_sizes pointers must be non-NULL.
 */
aompls_model_t* aompls_fit(const double* X, int n, int p,
                           const double* y,
                           const aompls_config_t* cfg,
                           char** err_msg);

/* Predict. Writes n doubles to out. Returns 0 on success. Caller must size
 * out to at least n elements. The number of features p is implicit (taken
 * from the fitted model).
 */
int aompls_predict(const aompls_model_t* model,
                   const double* X, int n,
                   double* out,
                   char** err_msg);

/* Accessors / diagnostics. */
int   aompls_n_features(const aompls_model_t* model);
int   aompls_n_components(const aompls_model_t* model);
int   aompls_selected_operator_index(const aompls_model_t* model);
const char* aompls_selected_operator_name(const aompls_model_t* model);
void  aompls_get_coef(const aompls_model_t* model, double* out);    /* size n_features */
double aompls_get_intercept(const aompls_model_t* model);
void  aompls_get_x_mean(const aompls_model_t* model, double* out);  /* size n_features */
double aompls_get_y_mean(const aompls_model_t* model);

/* RMSE curves: written into a caller-provided row-major buffer of size
 * (n_ops * max_components). Returns n_ops on success (will be 9 for the
 * compact bank), 0 if unavailable.
 */
int   aompls_get_rmse_curves(const aompls_model_t* model, int* n_ops, int* n_k, double* out);

/* Free a model handle. */
void  aompls_free(aompls_model_t* model);

/* Free an error string allocated by aompls_fit / aompls_predict. */
void  aompls_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif /* AOMPLS_C_API_H */
