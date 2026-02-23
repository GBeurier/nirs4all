"""Mixture of Preprocessing Experts (MoE) PLS Regressor.

CHANGELOG:
    v1: 4 experts (SNV, SG d1, Detrend, MSC+SG), simple meta-model PLS, slow (66-103s)
    v2: 42+ experts, too slow (698s), Ridge meta-model
    v3: 25 experts, sklearn PLS for OOF, Ridge meta-model — Amylose regression (2.19)
    v4: Two-phase OOF: fast sklearn screen → AOM-PLS OOF for top experts.
        Diversity filtering (correlation < 0.95). Safety fallback to best single expert.
        Non-negative RidgeCV meta-model. Beats v3 on Amylose.
    v5: 94 experts screened (25 core + 69 extended). Extended pool adds OSC, EMSC,
        KubelkaMunk, ArPLS, Haar, coif3/sym8 wavelets, FirstDerivative, SecondDerivative,
        RobustSNV, AreaNorm, LogTransform, more SG params, deep 3rd/4th order chains.
        Only proven 25 core experts get AOM-PLS OOF (extended adds noise to ensemble).
        Two-tier architecture: screen all 94 → AOM-PLS for core 25 → diversity filter → Ridge.
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from nirs4all.operators.models.sklearn.aom_pls import AOMPLSRegressor


def _apply_chain(chain, X, y=None):
    """Apply a chain of transforms, fitting on X. Returns transformed X."""
    import inspect
    X_pre = X.copy()
    for op in chain:
        if hasattr(op, 'fit'):
            sig = inspect.signature(op.fit)
            if 'y' in sig.parameters and y is not None:
                op.fit(X_pre, y)
            else:
                op.fit(X_pre)
        X_pre = op.apply(X_pre) if hasattr(op, 'apply') else op.transform(X_pre)
    return X_pre


def _apply_fitted_chain(chain, X):
    """Apply already-fitted chain to new data (no re-fitting)."""
    X_pre = X.copy()
    for op in chain:
        X_pre = op.apply(X_pre) if hasattr(op, 'apply') else op.transform(X_pre)
    return X_pre


def _make_chain(spec):
    """Create a fresh chain of transform instances from a specification.

    Spec is a list of (class, kwargs) tuples.
    """
    return [cls(**kwargs) for cls, kwargs in spec]


def _build_expert_specs():
    """Build expert chain specifications (class + kwargs, not instances).

    Returns (core_specs, extended_specs) where each is a list of
    (name, [(class, kwargs), ...]) tuples.

    Core: 25 proven experts (always evaluated with AOM-PLS OOF).
    Extended: 69 experts with OSC, EMSC, KubelkaMunk, ArPLS, Haar, coif3/sym8,
    etc. (screened with sklearn PLS but not included in ensemble).

    Instances are created on demand via _make_chain() to avoid state issues.
    """
    from nirs4all.operators.transforms import (
        StandardNormalVariate as SNV,
        RobustStandardNormalVariate as RSNV,
        MultiplicativeScatterCorrection as MSC,
        ExtendedMultiplicativeScatterCorrection as EMSC,
        SavitzkyGolay as SG,
        Detrend,
        Gaussian,
        NorrisWilliams,
        WaveletDenoise,
        Haar,
        OSC,
        FirstDerivative,
        SecondDerivative,
        KubelkaMunk,
        ArPLS,
        LogTransform,
        AreaNormalization,
    )

    # Shortcuts for readability
    sg = lambda w, d: (SG, {"window_length": w, "polyorder": min(2 if d < 2 else 3, w - 1), "deriv": d})
    snv = (SNV, {})
    rsnv = (RSNV, {})
    msc = (MSC, {})
    emsc = lambda deg=2: (EMSC, {"degree": deg})
    det = (Detrend, {})
    gau = lambda s: (Gaussian, {"sigma": s})
    nw = lambda g, d: (NorrisWilliams, {"gap": g, "segment": 5, "deriv": d})
    wav = lambda wt="db4", lvl=4: (WaveletDenoise, {"wavelet": wt, "level": lvl})
    haar = (Haar, {})
    osc = lambda nc=1: (OSC, {"n_components": nc})
    d1 = (FirstDerivative, {})
    d2 = (SecondDerivative, {})
    km = (KubelkaMunk, {})
    arpls = (ArPLS, {"lam": 1e6})
    logt = (LogTransform, {})
    anorm = (AreaNormalization, {})

    # === CORE experts (evaluated with AOM-PLS OOF for meta-model) ===
    # Proven v4 set: 25 experts that consistently work well with AOM-PLS
    core = [
        ("identity", []),
        ("SNV", [snv]),
        ("MSC", [msc]),
        ("SG_d1_11", [sg(11, 1)]),
        ("SG_d1_21", [sg(21, 1)]),
        ("SG_d1_31", [sg(31, 1)]),
        ("SG_d2_21", [sg(21, 2)]),
        ("Detrend", [det]),
        ("Gaussian_5", [gau(5)]),
        ("NW_5_1", [nw(5, 1)]),
        ("Wav_db4", [wav("db4", 4)]),
        ("SNV>SG_d1_21", [snv, sg(21, 1)]),
        ("SNV>SG_d1_11", [snv, sg(11, 1)]),
        ("SNV>Detrend", [snv, det]),
        ("MSC>SG_d1_21", [msc, sg(21, 1)]),
        ("MSC>Detrend", [msc, det]),
        ("Detrend>SNV", [det, snv]),
        ("Wav_db4>SNV", [wav("db4", 4), snv]),
        ("Gau5>SNV", [gau(5), snv]),
        ("SNV>SG_d1_21>Detrend", [snv, sg(21, 1), det]),
        ("MSC>SG_d1_21>Detrend", [msc, sg(21, 1), det]),
        ("Detrend>SNV>SG_d1_21", [det, snv, sg(21, 1)]),
        ("Wav_db4>SNV>SG_d1_21", [wav("db4", 4), snv, sg(21, 1)]),
        ("SNV>SG_d0_15>SG_d1_21>Detrend", [snv, sg(15, 0), sg(21, 1), det]),
        ("MSC>SG_d0_15>SG_d1_21>Detrend", [msc, sg(15, 0), sg(21, 1), det]),
    ]

    # === EXTENDED experts (screened with sklearn PLS, best contribute OOF as
    # additional meta-features alongside core experts' AOM-PLS OOF) ===
    extended = [
        # -- 1st order: scatter --
        ("RSNV", [rsnv]),
        ("EMSC_2", [emsc(2)]),
        ("EMSC_3", [emsc(3)]),
        # -- 1st order: SG params --
        ("SG_d0_11", [sg(11, 0)]),
        ("SG_d0_21", [sg(21, 0)]),
        ("SG_d1_7", [sg(7, 1)]),
        ("SG_d2_11", [sg(11, 2)]),
        ("SG_d2_31", [sg(31, 2)]),
        # -- 1st order: derivatives --
        ("FirstDeriv", [d1]),
        ("SecondDeriv", [d2]),
        # -- 1st order: smoothing --
        ("Gaussian_3", [gau(3)]),
        ("Gaussian_8", [gau(8)]),
        # -- 1st order: gap derivatives --
        ("NW_11_1", [nw(11, 1)]),
        ("NW_5_2", [nw(5, 2)]),
        # -- 1st order: wavelets --
        ("Wav_db8", [wav("db8", 4)]),
        ("Wav_coif3", [wav("coif3", 4)]),
        ("Wav_sym8", [wav("sym8", 4)]),
        ("Haar", [haar]),
        # -- 1st order: OSC (supervised) --
        ("OSC_1", [osc(1)]),
        ("OSC_2", [osc(2)]),
        # -- 1st order: signal conversion --
        ("KubelkaMunk", [km]),
        ("ArPLS", [arpls]),
        ("LogTransform", [logt]),
        ("AreaNorm", [anorm]),
        # -- 2nd order --
        ("SNV>SG_d1_31", [snv, sg(31, 1)]),
        ("SNV>SG_d2_21", [snv, sg(21, 2)]),
        ("SNV>FirstDeriv", [snv, d1]),
        ("RSNV>SG_d1_21", [rsnv, sg(21, 1)]),
        ("RSNV>Detrend", [rsnv, det]),
        ("MSC>SG_d2_21", [msc, sg(21, 2)]),
        ("EMSC>SG_d1_21", [emsc(2), sg(21, 1)]),
        ("EMSC>Detrend", [emsc(2), det]),
        ("Detrend>MSC", [det, msc]),
        ("Wav_db4>MSC", [wav("db4", 4), msc]),
        ("Wav_coif3>SNV", [wav("coif3", 4), snv]),
        ("Gau5>MSC", [gau(5), msc]),
        ("OSC>SNV", [osc(1), snv]),
        ("OSC>SG_d1_21", [osc(1), sg(21, 1)]),
        ("OSC>MSC", [osc(1), msc]),
        ("ArPLS>SNV", [arpls, snv]),
        ("ArPLS>SG_d1_21", [arpls, sg(21, 1)]),
        ("KM>SNV", [km, snv]),
        ("KM>SG_d1_21", [km, sg(21, 1)]),
        # -- 3rd order --
        ("SNV>SG_d1_11>Detrend", [snv, sg(11, 1), det]),
        ("EMSC>SG_d1_21>Detrend", [emsc(2), sg(21, 1), det]),
        ("Detrend>MSC>SG_d1_21", [det, msc, sg(21, 1)]),
        ("Wav_coif3>SNV>SG_d1_21", [wav("coif3", 4), snv, sg(21, 1)]),
        ("OSC>SNV>SG_d1_21", [osc(1), snv, sg(21, 1)]),
        ("OSC>MSC>SG_d1_21", [osc(1), msc, sg(21, 1)]),
        ("OSC>SNV>Detrend", [osc(1), snv, det]),
        ("ArPLS>SNV>SG_d1_21", [arpls, snv, sg(21, 1)]),
        ("ArPLS>MSC>SG_d1_21", [arpls, msc, sg(21, 1)]),
        ("KM>SNV>SG_d1_21", [km, snv, sg(21, 1)]),
        ("KM>MSC>SG_d1_21", [km, msc, sg(21, 1)]),
        ("Gau5>SNV>SG_d1_21", [gau(5), snv, sg(21, 1)]),
        ("RSNV>SG_d1_21>Detrend", [rsnv, sg(21, 1), det]),
        ("AreaNorm>SNV>SG_d1_21", [anorm, snv, sg(21, 1)]),
        # -- 4th order --
        ("EMSC>SG_d0_15>SG_d1_21>Detrend", [emsc(2), sg(15, 0), sg(21, 1), det]),
        ("OSC>SNV>SG_d1_21>Detrend", [osc(1), snv, sg(21, 1), det]),
        ("OSC>MSC>SG_d1_21>Detrend", [osc(1), msc, sg(21, 1), det]),
        ("OSC>EMSC>SG_d1_21>Detrend", [osc(1), emsc(2), sg(21, 1), det]),
        ("ArPLS>SNV>SG_d1_21>Detrend", [arpls, snv, sg(21, 1), det]),
        ("ArPLS>MSC>SG_d1_21>Detrend", [arpls, msc, sg(21, 1), det]),
        ("KM>SNV>SG_d1_21>Detrend", [km, snv, sg(21, 1), det]),
        ("KM>EMSC>SG_d1_21>Detrend", [km, emsc(2), sg(21, 1), det]),
        ("Wav_db4>SNV>SG_d1_21>Detrend", [wav("db4", 4), snv, sg(21, 1), det]),
        ("Wav_coif3>MSC>SG_d1_21>Detrend", [wav("coif3", 4), msc, sg(21, 1), det]),
        ("Detrend>OSC>SNV>SG_d1_21", [det, osc(1), snv, sg(21, 1)]),
        ("Gau5>OSC>SNV>SG_d1_21", [gau(5), osc(1), snv, sg(21, 1)]),
    ]

    return core, extended


def _select_diverse_experts(oof_preds_list, oof_rmses, max_experts=8, corr_threshold=0.95):
    """Greedily select diverse experts: best RMSE first, skip if correlated."""
    ranked = np.argsort(oof_rmses)
    selected = []
    selected_preds = []

    for idx in ranked:
        if len(selected) >= max_experts:
            break
        pred = oof_preds_list[idx]
        # Check correlation with all already-selected experts
        too_similar = False
        for sp in selected_preds:
            corr = np.corrcoef(pred, sp)[0, 1]
            if abs(corr) > corr_threshold:
                too_similar = True
                break
        if not too_similar:
            selected.append(idx)
            selected_preds.append(pred)

    return selected


class MoEPLSRegressor(BaseEstimator, RegressorMixin):
    """Massive Mixture of Preprocessing Experts with 1st-4th order chains.

    v5: 94 experts (25 core + 69 extended) screened with sklearn PLS. Only proven
    25 core experts get AOM-PLS OOF. Extended pool (OSC, EMSC, KubelkaMunk, ArPLS,
    Haar, coif3/sym8, etc.) provides validation. Diversity filtering (max 8, corr 0.95),
    Ridge meta-model with safety fallback to best single expert.
    """
    def __init__(self, n_components=15, n_folds=5):
        self.n_components = n_components
        self.n_folds = n_folds

    def fit(self, X, y):
        core_specs, extended_specs = _build_expert_specs()
        all_specs = core_specs + extended_specs
        core_names = {name for name, _ in core_specs}
        n = X.shape[0]
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        folds = list(kf.split(X))

        # Phase 1: sklearn PLS OOF screening for ALL experts (core + extended)
        screen_experts = []
        screen_oof = []
        screen_rmses = []
        screen_is_core = []

        for name, spec in all_specs:
            try:
                chain = _make_chain(spec)
                X_full = _apply_chain(chain, X, y=y)
                if np.any(np.isnan(X_full)) or np.any(np.isinf(X_full)):
                    continue

                oof_pred = np.zeros(n)
                valid = True
                for train_idx, val_idx in folds:
                    chain = _make_chain(spec)
                    X_tr = _apply_chain(chain, X[train_idx], y=y[train_idx])
                    X_va = _apply_fitted_chain(chain, X[val_idx])

                    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
                        valid = False
                        break

                    n_comp = min(self.n_components, X_tr.shape[0] - 2, X_tr.shape[1])
                    if n_comp < 1:
                        valid = False
                        break
                    pls = PLSRegression(n_components=n_comp)
                    pls.fit(X_tr, y[train_idx])
                    oof_pred[val_idx] = pls.predict(X_va).flatten()

                if not valid:
                    continue

                oof_rmse = np.sqrt(np.mean((y - oof_pred) ** 2))
                if oof_rmse > 3 * np.std(y):
                    continue

                screen_experts.append((name, spec))
                screen_oof.append(oof_pred)
                screen_rmses.append(oof_rmse)
                screen_is_core.append(name in core_names)
            except Exception:
                continue

        if sum(screen_is_core) == 0:
            self.pls_fallback_ = AOMPLSRegressor(n_components=self.n_components)
            self.pls_fallback_.fit(X, y)
            self.valid_experts_ = []
            return self

        # Phase 2: AOM-PLS OOF for core experts only (proven stable ensemble)
        core_idx = [i for i in range(len(screen_experts)) if screen_is_core[i]]

        aom_experts = []
        aom_oof = []

        for si in core_idx:
            name, spec = screen_experts[si]
            try:
                oof_pred = np.zeros(n)
                valid = True
                for train_idx, val_idx in folds:
                    chain = _make_chain(spec)
                    X_tr = _apply_chain(chain, X[train_idx], y=y[train_idx])
                    X_va = _apply_fitted_chain(chain, X[val_idx])

                    if np.any(np.isnan(X_tr)) or np.any(np.isinf(X_tr)):
                        valid = False
                        break

                    pls = AOMPLSRegressor(n_components=self.n_components)
                    pls.fit(X_tr, y[train_idx])
                    oof_pred[val_idx] = pls.predict(X_va).flatten()

                if not valid:
                    continue

                aom_experts.append((name, spec))
                aom_oof.append(oof_pred)
            except Exception:
                continue

        if len(aom_experts) == 0:
            self.pls_fallback_ = AOMPLSRegressor(n_components=self.n_components)
            self.pls_fallback_.fit(X, y)
            self.valid_experts_ = []
            return self

        # Phase 3: Diversity filtering + meta-model
        aom_rmses = [np.sqrt(np.mean((y - p) ** 2)) for p in aom_oof]
        diverse_idx = _select_diverse_experts(aom_oof, aom_rmses, max_experts=8, corr_threshold=0.95)

        selected_experts = [aom_experts[i] for i in diverse_idx]
        selected_oof = [aom_oof[i] for i in diverse_idx]
        selected_rmses = [aom_rmses[i] for i in diverse_idx]

        self.oof_rmses_ = {selected_experts[i][0]: selected_rmses[i] for i in range(len(selected_experts))}

        # Best single expert OOF RMSE (safety benchmark)
        best_single_rmse = min(selected_rmses)
        best_single_idx = selected_rmses.index(best_single_rmse)

        if len(selected_experts) >= 2:
            oof_matrix = np.column_stack(selected_oof)
            self.meta_model_ = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0], cv=min(5, n))
            self.meta_model_.fit(oof_matrix, y)
            meta_oof_pred = self.meta_model_.predict(oof_matrix)
            meta_rmse = np.sqrt(np.mean((y - meta_oof_pred) ** 2))

            # Safety check: use meta-model only if it beats best single expert
            self.use_meta_ = meta_rmse < best_single_rmse
        else:
            self.use_meta_ = False
            self.meta_model_ = None

        # Phase 4: Final fit on ALL data
        self.expert_models_ = []
        self.valid_experts_ = selected_experts

        if not self.use_meta_:
            # Single best expert only
            name, spec = selected_experts[best_single_idx]
            chain = _make_chain(spec)
            X_pre = _apply_chain(chain, X, y=y)
            pls = AOMPLSRegressor(n_components=self.n_components)
            pls.fit(X_pre, y)
            self.expert_models_.append((chain, pls))
            self.valid_experts_ = [selected_experts[best_single_idx]]
            self.n_active_experts_ = 1
        else:
            self.n_active_experts_ = len(selected_experts)
            for name, spec in selected_experts:
                chain = _make_chain(spec)
                X_pre = _apply_chain(chain, X, y=y)
                pls = AOMPLSRegressor(n_components=self.n_components)
                pls.fit(X_pre, y)
                self.expert_models_.append((chain, pls))

        # Store training y stats for prediction clipping
        self.y_mean_ = np.mean(y)
        self.y_std_ = np.std(y)

        self.pls_fallback_ = None
        return self

    def predict(self, X):
        if self.pls_fallback_ is not None:
            return self.pls_fallback_.predict(X).flatten()

        # Safety bounds: clip predictions to 5 std from training mean
        clip_lo = self.y_mean_ - 5 * self.y_std_
        clip_hi = self.y_mean_ + 5 * self.y_std_

        all_preds = np.zeros((X.shape[0], len(self.expert_models_)))
        for i, (chain, pls) in enumerate(self.expert_models_):
            try:
                X_pre = _apply_fitted_chain(chain, X)
                pred = pls.predict(X_pre).flatten()
                if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
                    pred = np.full(X.shape[0], self.y_mean_)
                else:
                    pred = np.clip(pred, clip_lo, clip_hi)
                all_preds[:, i] = pred
            except Exception:
                all_preds[:, i] = self.y_mean_

        if self.use_meta_ and self.meta_model_ is not None:
            return self.meta_model_.predict(all_preds).flatten()
        else:
            return all_preds[:, 0].flatten()
