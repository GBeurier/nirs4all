## SPDX-License-Identifier: CeCILL-2.1
## Public API: aom_pls(), predict.aom_pls(), aom_pls_tune(), accessors.

#' Fit AOM-PLS (compact, PLS1)
#'
#' Adaptive Operator Mixture PLS with a 9-operator compact bank
#' (Identity, Savitzky-Golay smoothing/derivatives, polynomial detrend,
#' finite difference). Cross-validates each operator over prefix
#' lengths k = 1..max_components and picks the (operator, k) pair
#' minimising mean fold RMSE.
#'
#' @param X Numeric matrix (n_samples x n_features).
#' @param y Numeric vector (n_samples).
#' @param max_components Integer. Maximum number of PLS components. Default 15.
#' @param n_folds Integer. CV folds (ignored when cv_mode = "external"). Default 5.
#' @param cv_mode One of "kfold", "spxy", "holdout", "external".
#' @param one_se_rule Logical. Apply the one-standard-error parsimony rule.
#' @param random_state Integer seed for the internal CV shuffler (own RNG; not
#'   bit-compatible with numpy's MT19937).
#' @param preproc One of "none","snv","msc","osc","asls","snv+osc","asls+osc".
#' @param osc_n_components Integer. Number of OSC components when preproc includes "osc".
#' @param asls Named list with lam, p, n_iter (defaults: list(lam = 1e5, p = 0.01, n_iter = 10L)).
#' @param center Logical. Mean-center X and y before AOM.
#' @param external_folds List of integer vectors of TEST indices per fold;
#'   required when cv_mode = "external".
#' @return Object of class "aom_pls".
#' @export
aom_pls <- function(X, y,
                    max_components = 15L,
                    n_folds = 5L,
                    cv_mode = c("kfold", "spxy", "holdout", "external"),
                    one_se_rule = FALSE,
                    random_state = 0L,
                    preproc = c("none", "snv", "msc", "osc", "asls", "snv+osc", "asls+osc"),
                    osc_n_components = 1L,
                    asls = list(lam = 1e5, p = 0.01, n_iter = 10L),
                    center = TRUE,
                    external_folds = NULL) {
  cv_mode <- match.arg(cv_mode)
  preproc <- match.arg(preproc)
  if (!is.matrix(X)) stop("X must be a matrix", call. = FALSE)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  if (length(y) != nrow(X)) stop("length(y) must equal nrow(X)", call. = FALSE)
  if (is.null(asls$lam)) asls$lam <- 1e5
  if (is.null(asls$p)) asls$p <- 0.01
  if (is.null(asls$n_iter)) asls$n_iter <- 10L
  if (cv_mode == "external" && is.null(external_folds))
    stop("cv_mode='external' requires external_folds", call. = FALSE)
  if (!is.null(external_folds))
    external_folds <- lapply(external_folds, function(v) as.integer(v) - 0L)

  model <- aompls_fit_cpp(
    X, y,
    max_components = as.integer(max_components),
    n_folds = as.integer(n_folds),
    cv_mode = cv_mode,
    one_se_rule = isTRUE(one_se_rule),
    center = isTRUE(center),
    random_state = as.numeric(random_state),
    preproc = preproc,
    osc_n_components = as.integer(osc_n_components),
    asls_lam = as.numeric(asls$lam),
    asls_p = as.numeric(asls$p),
    asls_n_iter = as.integer(asls$n_iter),
    external_folds = external_folds
  )
  model$call <- match.call()
  class(model) <- "aom_pls"
  model
}

#' @export
predict.aom_pls <- function(object, newdata, ...) {
  if (!is.matrix(newdata)) stop("newdata must be a matrix", call. = FALSE)
  storage.mode(newdata) <- "double"
  aompls_predict_cpp(object, newdata)
}

#' @export
coef.aom_pls <- function(object, ...) {
  object$coef
}

#' @export
print.aom_pls <- function(x, ...) {
  cat("AOM-PLS (compact, PLS1)\n")
  cat(sprintf("  selected operator : %s (index %d)\n",
              x$selected_operator_name, x$selected_operator_index))
  cat(sprintf("  n_components      : %d\n", x$n_components_selected))
  cat(sprintf("  n_features        : %d\n", x$n_features))
  cat(sprintf("  one-SE applied    : %s\n", x$one_se_applied))
  cat(sprintf("  fit time (s)      : %.4f\n", x$fit_time_s))
  invisible(x)
}

#' @export
summary.aom_pls <- function(object, ...) {
  print(object)
  cat("\nRMSE curve (selected operator):\n")
  print(object$rmse_curves[object$selected_operator_index + 1L, ])
  invisible(object)
}

#' Outer-K-fold grid HPO for aom_pls
#'
#' For each (max_components, preproc) combination, fits aom_pls on each
#' outer-train fold and evaluates RMSE on the outer-test fold. Returns the best
#' config and a model refit on the full data.
#' @export
aom_pls_tune <- function(X, y,
                         max_components_grid = c(5L, 10L, 15L, 20L, 25L),
                         preproc_grid = c("none", "snv", "osc", "asls", "asls+osc"),
                         n_folds = 5L,
                         cv_mode = "kfold",
                         outer_folds = 5L,
                         one_se_rule = FALSE,
                         random_state = 0L) {
  if (!is.matrix(X)) stop("X must be a matrix", call. = FALSE)
  storage.mode(X) <- "double"
  y <- as.numeric(y)
  n <- nrow(X)
  set.seed(random_state)
  perm <- sample.int(n)
  folds <- split(perm, cut(seq_along(perm), breaks = outer_folds, labels = FALSE))
  grid <- expand.grid(max_components = max_components_grid,
                      preproc = preproc_grid, stringsAsFactors = FALSE)
  scores <- numeric(nrow(grid))
  for (i in seq_len(nrow(grid))) {
    cfg <- grid[i, ]
    rmses <- numeric(outer_folds)
    for (f in seq_len(outer_folds)) {
      test_idx <- folds[[f]]
      train_idx <- setdiff(seq_len(n), test_idx)
      est <- aom_pls(X[train_idx, , drop = FALSE], y[train_idx],
                     max_components = as.integer(cfg$max_components),
                     n_folds = n_folds,
                     cv_mode = cv_mode,
                     one_se_rule = one_se_rule,
                     random_state = random_state,
                     preproc = cfg$preproc)
      pred <- predict(est, X[test_idx, , drop = FALSE])
      rmses[f] <- sqrt(mean((pred - y[test_idx])^2))
    }
    scores[i] <- mean(rmses)
  }
  best <- which.min(scores)
  best_cfg <- list(max_components = as.integer(grid$max_components[best]),
                   preproc = grid$preproc[best])
  refit <- aom_pls(X, y,
                   max_components = best_cfg$max_components,
                   n_folds = n_folds,
                   cv_mode = cv_mode,
                   one_se_rule = one_se_rule,
                   random_state = random_state,
                   preproc = best_cfg$preproc)
  list(best_params = best_cfg, best_score = scores[best],
       all_results = data.frame(grid, rmse = scores), refit_model = refit)
}
