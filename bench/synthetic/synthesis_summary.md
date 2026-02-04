| # | Étape | Short desc | Paramètres / choix clés |
|---|-------|------------|-------------------------|
| 1 | Grille spectrale | Définir start/end/step et n_features | wavelength_start/end/step, n_features, random_state, complexity simple/realistic/complex (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L332-L376) |
| 2 | Bibliothèque de composants | Choisir librairie (eau, protéines, lipides…), nb composantes | components list ou n_components, domaine (food/pharma/etc.), interpolation si needed (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L60-L84) |
| 3 | Sampling des concentrations | Tirage des proportions des composants | Dirichlet, uniform, lognormal, corrélés; bounds par composant (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L70-L96) |
| 4 | Mode de mesure | Géométrie optique | Transmittance (Beer-Lambert, path length), Diffuse reflectance (Kubelka-Munk), Transflectance, ATR (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L84-L110) |
| 5 | Mélange physique | Combinaison spectrale des composants | Beer-Lambert mix, coefficients d’extinction, optional saturation (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L70-L96) |
| 6 | Forme de bande | Profil des pics | Voigt/Gauss/Lorentz, largeur/offset par composant (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L70-L96) |
| 7 | Instrument / détecteur | Archetype capteur et LSF | Si / InGaAs / e-InGaAs / PbS / PbSe / MCT; résolution, LSF convolution, roll-off (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L60-L84) |
| 8 | Stitch multi-capteurs | Fusion de segments | Overlap, normalisation croisées, équilibrage SNR (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L60-L84) |
| 9 | Mode multi-scan | Moyennes / répétitions instrument | n_scans, réduction bruit √n, drift inter-scan (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L60-L84) |
| 10 | Baseline | Fond additif | On/off, ordre polynomial, amplitude, dérive lente (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L112-L128) |
| 11 | Diffusion / scatter | Effets multiplicatifs & courbure | EMSC, Rayleigh/Mie approx, facteurs multiplicatifs, offsets (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L96-L118) |
| 12 | Effets environnement | Température, humidité | Temp shift/broadening, différenciation eau libre/ liée, paramètres de sensibilité (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L96-L118) |
| 13 | Distorsions λ | Shift/stretch et artefacts | Wavelength shift/stretch, edge effects, stray light (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L112-L128) |
| 14 | Bruits | Modèle de bruit | Additif gaussien, multiplicatif, shot, thermique, 1/f; niveau global ou dépendant signal (docs/_internal/synthetic/SYNTHETIC_GENERATOR_ROADMAP.md#L112-L128) |
| 15 | Lissage / dérivées | Prétraitements simulés | Savitzky-Golay (fenêtre, ordre), dérivées 1/2, éventuellement SNV (docs/_internal/synthetic/generator_improvement_methods.md#L112-L130) |
| 16 | Resampling & grille sortie | Adaptation à grille cible | Interpolation, décimation, alignement nm, header_unit nm (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L332-L376) |
| 17 | Batch effects | Variations par lot | n_batches, shifts d’intensité, offsets spectre par lot (docs/source/user_guide/data/synthetic_data.md#L109-L118) |
| 18 | Metadata & partitions | Structure échantillons | n_groups, n_repetitions (min,max), train_ratio, IDs, regroupement (docs/source/user_guide/data/synthetic_data.md#L109-L118) |
| 19 | Export / format | Sortie finale | dataset vs arrays, CSV/zip/gzip, headers, file_layout standard/single/fragmented (docs/_internal/specifications/synthetic/synthetic_generator_specification.md#L380-L430) |
| 20 | Cible de base (régression) | y linéaire corrélé aux concentrations | distribution uniform/normal/lognormal/bimodal, range, component ou weighted mix, correlation, noise (docs/source/user_guide/data/synthetic_data.md#L105-L129) |
| 21 | Multi-target | Plusieurs y issus de composants | target_components / n_targets, gammes séparées (docs/_internal/synthetic/synthetic_final.md#L705-L721) |
| 22 | Classification | Labels séparables | n_classes, class_weights/balance, separation factor/method (component/threshold/cluster) (nirs4all/synthesis/targets.py#L33-L126) |
| 23 | Interactions non linéaires | Complexifier la relation y–C | interactions polynomial/synergistic/antagonistic, interaction_strength, hidden_factors, polynomial_degree (docs/source/user_guide/data/synthetic_data.md#L132-L166) |
| 24 | Confusions / imprédictible | Introduire erreur irréductible | signal_to_confound_ratio, n_confounders, spectral_masking, temporal_drift, heteroscedastic noise (docs/source/user_guide/data/synthetic_data.md#L168-L186; nirs4all/synthesis/targets.py#L646-L704) |
| 25 | Paysages multi-régimes | Relations y différentes par zone | n_regimes, regime_method (concentration/spectral/random), regime_overlap, noise_heteroscedasticity (docs/source/user_guide/data/synthetic_data.md#L188-L210) |



cfg =
{
  "step01_grid_wavelength_start": 1000,
  "step01_grid_wavelength_end": 2500,
  "step01_grid_wavelength_step": 2,
  "step01_grid_n_features": 751,

  "step02_components_library": "food_basic",
  "step02_components_n_components": 6,
  "step02_components_interpolation": "linear",

  "step03_concentrations_dist": "dirichlet",
  "step03_concentrations_params": {"alpha": 1.0},

  "step04_measurement_mode": "transmittance",
  "step04_measurement_path_length": 1.0,

  "step05_mixing_model": "beer_lambert",
  "step05_mixing_saturation": false,

  "step06_bands_profile": "voigt",
  "step06_bands_width": 12.0,
  "step06_bands_center_jitter": 0.5,

  "step07_instrument_detector": "ingaas",
  "step07_instrument_fwhm": 8.0,
  "step07_instrument_rolloff": 0.0,

  "step08_stitch_overlap": 10,
  "step08_stitch_normalization": "snr_balance",

  "step09_multiscan_n_scans": 1,
  "step09_multiscan_drift": 0.0,

  "step10_baseline_enable": true,
  "step10_baseline_order": 2,
  "step10_baseline_amplitude": 0.02,
  "step10_baseline_drift": 0.0,

  "step11_scatter_enable": true,
  "step11_scatter_model": "emsc",
  "step11_scatter_alpha_std": 0.05,
  "step11_scatter_beta_std": 0.01,

  "step12_environment_enable": true,
  "step12_environment_temp_shift": 0.0,
  "step12_environment_temp_broadening": 0.0,
  "step12_environment_water_bound_free_ratio": 0.0,

  "step13_wavelength_distortion_shift_std": 0.5,
  "step13_wavelength_distortion_stretch_std": 0.001,
  "step13_wavelength_distortion_edge_effects": 0.0,

  "step14_noise_enable": true,
  "step14_noise_base": 0.005,
  "step14_noise_signal_dep": 0.01,
  "step14_noise_one_over_f": 0.0,
  "step14_noise_shot": 0.0,

  "step15_preprocess_smooth_enable": false,
  "step15_preprocess_smooth_window": 15,
  "step15_preprocess_smooth_order": 2,
  "step15_preprocess_derivative_order": 0,
  "step15_preprocess_snv_enable": false,

  "step16_resample_enable": true,
  "step16_resample_target_step": 2.0,
  "step16_resample_method": "linear",

  "step17_batch_enable": false,
  "step17_batch_n_batches": 0,
  "step17_batch_gain_std": 0.0,
  "step17_batch_offset_std": 0.0,

  "step18_metadata_groups": 0,
  "step18_metadata_repetitions": [1, 1],
  "step18_metadata_train_ratio": 0.8,

  "step19_export_format": "dataset",
  "step19_export_compression": "gzip",
  "step19_export_file_layout": "standard",

  "step20_target_mode": "regression",
  "step20_target_distribution": "lognormal",
  "step20_target_range": [0, 100],
  "step20_target_component": "protein",
  "step20_target_correlation": 0.9,
  "step20_target_noise": 0.1,

  "step21_multitarget_n_targets": 1,
  "step21_multitarget_components": ["protein"],

  "step22_classification_n_classes": 3,
  "step22_classification_class_weights": [0.34, 0.33, 0.33],
  "step22_classification_separation": 1.5,
  "step22_classification_separation_method": "component",

  "step23_nonlinear_interactions": "polynomial",
  "step23_interaction_strength": 0.6,
  "step23_hidden_factors": 2,
  "step23_polynomial_degree": 2,

  "step24_confounds_signal_to_confound_ratio": 0.7,
  "step24_confounds_n_confounders": 2,
  "step24_confounds_spectral_masking": 0.0,
  "step24_confounds_temporal_drift": true,
  "step24_confounds_noise_heteroscedasticity": 0.3,

  "step25_regimes_n_regimes": 3,
  "step25_regimes_method": "concentration",
  "step25_regimes_overlap": 0.2
}


"param_meta": {
    "step01_grid_wavelength_start": "float>0 (nm)",
    "step01_grid_wavelength_end": "float>start (nm)",
    "step01_grid_wavelength_step": "float>0 (nm)",
    "step01_grid_n_features": "int>0 (if set, overrides step size)",
    "step02_components_library": "string (e.g. food_basic, pharma, minerals)",
    "step02_components_n_components": "int>0",
    "step02_components_interpolation": "linear|cubic|spline",
    "step03_concentrations_dist": "dirichlet|uniform|lognormal|correlated",
    "step03_concentrations_params": "object (depends on dist)",
    "step04_measurement_mode": "transmittance|diffuse_reflectance|transflectance|atr",
    "step04_measurement_path_length": "float>0",
    "step05_mixing_model": "beer_lambert|kubelka_munk",
    "step05_mixing_saturation": "bool",
    "step06_bands_profile": "voigt|gauss|lorentz",
    "step06_bands_width": "float>0",
    "step06_bands_center_jitter": "float>=0",
    "step07_instrument_detector": "si|ingaas|extended_ingaas|pbs|pbse|mct",
    "step07_instrument_fwhm": "float>0 (nm)",
    "step07_instrument_rolloff": "float>=0",
    "step08_stitch_overlap": "int>=0 (nm or index)",
    "step08_stitch_normalization": "snr_balance|mean_match|none",
    "step09_multiscan_n_scans": "int>=1",
    "step09_multiscan_drift": "float>=0",
    "step10_baseline_enable": "bool",
    "step10_baseline_order": "int>=0",
    "step10_baseline_amplitude": "float>=0",
    "step10_baseline_drift": "float>=0",
    "step11_scatter_enable": "bool",
    "step11_scatter_model": "emsc|rayleigh|mie",
    "step11_scatter_alpha_std": "float>=0",
    "step11_scatter_beta_std": "float>=0",
    "step12_environment_enable": "bool",
    "step12_environment_temp_shift": "float (nm or AU)",
    "step12_environment_temp_broadening": "float>=0",
    "step12_environment_water_bound_free_ratio": "float>=0",
    "step13_wavelength_distortion_shift_std": "float>=0 (nm)",
    "step13_wavelength_distortion_stretch_std": "float>=0",
    "step13_wavelength_distortion_edge_effects": "float>=0",
    "step14_noise_enable": "bool",
    "step14_noise_base": "float>=0",
    "step14_noise_signal_dep": "float>=0",
    "step14_noise_one_over_f": "float>=0",
    "step14_noise_shot": "float>=0",
    "step15_preprocess_smooth_enable": "bool",
    "step15_preprocess_smooth_window": "int odd >=3",
    "step15_preprocess_smooth_order": "int>=1",
    "step15_preprocess_derivative_order": "0|1|2",
    "step15_preprocess_snv_enable": "bool",
    "step16_resample_enable": "bool",
    "step16_resample_target_step": "float>0 (nm)",
    "step16_resample_method": "linear|cubic|spline",
    "step17_batch_enable": "bool",
    "step17_batch_n_batches": "int>=0",
    "step17_batch_gain_std": "float>=0",
    "step17_batch_offset_std": "float>=0",
    "step18_metadata_groups": "int>=0",
    "step18_metadata_repetitions": "[min,max] ints >=1",
    "step18_metadata_train_ratio": "float in (0,1)",
    "step19_export_format": "dataset|arrays|csv",
    "step19_export_compression": "none|gzip|zip",
    "step19_export_file_layout": "standard|single|fragmented",
    "step20_target_mode": "regression|classification",
    "step20_target_distribution": "uniform|normal|lognormal|bimodal",
    "step20_target_range": "[min,max]",
    "step20_target_component": "string component name or index",
    "step20_target_correlation": "float in [0,1]",
    "step20_target_noise": "float>=0",
    "step21_multitarget_n_targets": "int>=1",
    "step21_multitarget_components": "list of component names",
    "step22_classification_n_classes": "int>=2",
    "step22_classification_class_weights": "list summing to 1",
    "step22_classification_separation": "float>=0",
    "step22_classification_separation_method": "component|threshold|cluster",
    "step23_nonlinear_interactions": "none|polynomial|synergistic|antagonistic",
    "step23_interaction_strength": "float in [0,1]",
    "step23_hidden_factors": "int>=0",
    "step23_polynomial_degree": "int>=2",
    "step24_confounds_signal_to_confound_ratio": "float in (0,1]",
    "step24_confounds_n_confounders": "int>=0",
    "step24_confounds_spectral_masking": "float>=0",
    "step24_confounds_temporal_drift": "bool",
    "step24_confounds_noise_heteroscedasticity": "float>=0",
    "step25_regimes_n_regimes": "int>=1",
    "step25_regimes_method": "concentration|spectral|random",
    "step25_regimes_overlap": "float in [0,1]"
  }

def generate_from_config(cfg):
    grid = make_grid(
        cfg["step01_grid_wavelength_start"],
        cfg["step01_grid_wavelength_end"],
        cfg["step01_grid_wavelength_step"],
        cfg["step01_grid_n_features"],
    )

    components = load_components(
        library=cfg["step02_components_library"],
        n_components=cfg["step02_components_n_components"],
        interpolation=cfg["step02_components_interpolation"],
    )

    concentrations = sample_concentrations(
        dist=cfg["step03_concentrations_dist"],
        params=cfg["step03_concentrations_params"],
        n_samples=cfg.get("n_samples", 1000),
        n_components=cfg["step02_components_n_components"],
    )

    X = generate_pure_spectra(
        grid=grid,
        components=components,
        concentrations=concentrations,
        measurement_mode=cfg["step04_measurement_mode"],
        path_length=cfg["step04_measurement_path_length"],
        mixing_model=cfg["step05_mixing_model"],
        saturation=cfg["step05_mixing_saturation"],
        band_profile=cfg["step06_bands_profile"],
        band_width=cfg["step06_bands_width"],
        band_center_jitter=cfg["step06_bands_center_jitter"],
    )

    X = apply_instrument(
        X,
        detector=cfg["step07_instrument_detector"],
        fwhm=cfg["step07_instrument_fwhm"],
        rolloff=cfg["step07_instrument_rolloff"],
    )

    X = apply_stitching(
        X,
        overlap=cfg["step08_stitch_overlap"],
        normalization=cfg["step08_stitch_normalization"],
    )

    X = apply_multiscan(
        X,
        n_scans=cfg["step09_multiscan_n_scans"],
        drift=cfg["step09_multiscan_drift"],
    )

    if cfg["step10_baseline_enable"]:
        X = apply_baseline(
            X,
            order=cfg["step10_baseline_order"],
            amplitude=cfg["step10_baseline_amplitude"],
            drift=cfg["step10_baseline_drift"],
        )

    if cfg["step11_scatter_enable"]:
        X = apply_scatter(
            X,
            model=cfg["step11_scatter_model"],
            alpha_std=cfg["step11_scatter_alpha_std"],
            beta_std=cfg["step11_scatter_beta_std"],
        )

    if cfg["step12_environment_enable"]:
        X = apply_environment(
            X,
            temp_shift=cfg["step12_environment_temp_shift"],
            temp_broadening=cfg["step12_environment_temp_broadening"],
            water_ratio=cfg["step12_environment_water_bound_free_ratio"],
        )

    X = apply_wavelength_distortion(
        X,
        shift_std=cfg["step13_wavelength_distortion_shift_std"],
        stretch_std=cfg["step13_wavelength_distortion_stretch_std"],
        edge_effects=cfg["step13_wavelength_distortion_edge_effects"],
    )

    if cfg["step14_noise_enable"]:
        X = apply_noise(
            X,
            base=cfg["step14_noise_base"],
            signal_dep=cfg["step14_noise_signal_dep"],
            one_over_f=cfg["step14_noise_one_over_f"],
            shot=cfg["step14_noise_shot"],
        )

    if cfg["step15_preprocess_smooth_enable"]:
        X = apply_savgol(
            X,
            window=cfg["step15_preprocess_smooth_window"],
            order=cfg["step15_preprocess_smooth_order"],
        )

    if cfg["step15_preprocess_derivative_order"] > 0:
        X = apply_derivative(
            X,
            order=cfg["step15_preprocess_derivative_order"],
        )

    if cfg["step15_preprocess_snv_enable"]:
        X = apply_snv(X)

    if cfg["step16_resample_enable"]:
        X = resample_grid(
            X,
            target_step=cfg["step16_resample_target_step"],
            method=cfg["step16_resample_method"],
        )

    if cfg["step17_batch_enable"]:
        X, metadata = apply_batch_effects(
            X,
            n_batches=cfg["step17_batch_n_batches"],
            gain_std=cfg["step17_batch_gain_std"],
            offset_std=cfg["step17_batch_offset_std"],
        )
    else:
        metadata = {}

    metadata = add_metadata(
        metadata,
        n_groups=cfg["step18_metadata_groups"],
        repetitions=cfg["step18_metadata_repetitions"],
        train_ratio=cfg["step18_metadata_train_ratio"],
    )

    y = generate_targets(
        concentrations=concentrations,
        mode=cfg["step20_target_mode"],
        distribution=cfg["step20_target_distribution"],
        target_range=cfg["step20_target_range"],
        component=cfg["step20_target_component"],
        correlation=cfg["step20_target_correlation"],
        noise=cfg["step20_target_noise"],
        n_targets=cfg["step21_multitarget_n_targets"],
        target_components=cfg["step21_multitarget_components"],
        n_classes=cfg["step22_classification_n_classes"],
        class_weights=cfg["step22_classification_class_weights"],
        separation=cfg["step22_classification_separation"],
        separation_method=cfg["step22_classification_separation_method"],
    )

    y = apply_nonlinear_targets(
        y,
        concentrations=concentrations,
        interactions=cfg["step23_nonlinear_interactions"],
        interaction_strength=cfg["step23_interaction_strength"],
        hidden_factors=cfg["step23_hidden_factors"],
        polynomial_degree=cfg["step23_polynomial_degree"],
    )

    y = apply_target_confounds(
        y,
        concentrations=concentrations,
        signal_to_confound_ratio=cfg["step24_confounds_signal_to_confound_ratio"],
        n_confounders=cfg["step24_confounds_n_confounders"],
        spectral_masking=cfg["step24_confounds_spectral_masking"],
        temporal_drift=cfg["step24_confounds_temporal_drift"],
        noise_heteroscedasticity=cfg["step24_confounds_noise_heteroscedasticity"],
    )

    y = apply_target_regimes(
        y,
        concentrations=concentrations,
        n_regimes=cfg["step25_regimes_n_regimes"],
        method=cfg["step25_regimes_method"],
        overlap=cfg["step25_regimes_overlap"],
    )

    return export_output(
        X, y, metadata,
        format=cfg["step19_export_format"],
        compression=cfg["step19_export_compression"],
        file_layout=cfg["step19_export_file_layout"],
    )



    def suggest_params(trial):
    cfg = {}

    # 1 Grid
    cfg["step01_grid_wavelength_start"] = trial.suggest_int("step01_grid_wavelength_start", 850, 1200)
    cfg["step01_grid_wavelength_end"] = trial.suggest_int("step01_grid_wavelength_end", 2000, 2600)
    cfg["step01_grid_wavelength_step"] = trial.suggest_float("step01_grid_wavelength_step", 1.0, 5.0)
    cfg["step01_grid_n_features"] = trial.suggest_int("step01_grid_n_features", 200, 1500)

    # 2 Components
    cfg["step02_components_library"] = trial.suggest_categorical(
        "step02_components_library",
        ["food_basic", "pharma", "minerals"]
    )
    cfg["step02_components_n_components"] = trial.suggest_int("step02_components_n_components", 2, 12)
    cfg["step02_components_interpolation"] = trial.suggest_categorical(
        "step02_components_interpolation",
        ["linear", "cubic", "spline"]
    )

    # 3 Concentrations
    cfg["step03_concentrations_dist"] = trial.suggest_categorical(
        "step03_concentrations_dist",
        ["dirichlet", "uniform", "lognormal", "correlated"]
    )
    # Exemple: params minimal pour dirichlet; adapte selon dist
    cfg["step03_concentrations_params"] = {
        "alpha": trial.suggest_float("step03_concentrations_params_alpha", 0.2, 3.0)
    }

    # 4 Measurement
    cfg["step04_measurement_mode"] = trial.suggest_categorical(
        "step04_measurement_mode",
        ["transmittance", "diffuse_reflectance", "transflectance", "atr"]
    )
    cfg["step04_measurement_path_length"] = trial.suggest_float(
        "step04_measurement_path_length", 0.1, 5.0
    )

    # 5 Mixing
    cfg["step05_mixing_model"] = trial.suggest_categorical(
        "step05_mixing_model", ["beer_lambert", "kubelka_munk"]
    )
    cfg["step05_mixing_saturation"] = trial.suggest_categorical(
        "step05_mixing_saturation", [False, True]
    )

    # 6 Bands
    cfg["step06_bands_profile"] = trial.suggest_categorical(
        "step06_bands_profile", ["voigt", "gauss", "lorentz"]
    )
    cfg["step06_bands_width"] = trial.suggest_float("step06_bands_width", 2.0, 30.0)
    cfg["step06_bands_center_jitter"] = trial.suggest_float("step06_bands_center_jitter", 0.0, 2.0)

    # 7 Instrument
    cfg["step07_instrument_detector"] = trial.suggest_categorical(
        "step07_instrument_detector", ["si", "ingaas", "extended_ingaas", "pbs", "pbse", "mct"]
    )
    cfg["step07_instrument_fwhm"] = trial.suggest_float("step07_instrument_fwhm", 2.0, 20.0)
    cfg["step07_instrument_rolloff"] = trial.suggest_float("step07_instrument_rolloff", 0.0, 0.2)

    # 8 Stitch
    cfg["step08_stitch_overlap"] = trial.suggest_int("step08_stitch_overlap", 0, 50)
    cfg["step08_stitch_normalization"] = trial.suggest_categorical(
        "step08_stitch_normalization", ["snr_balance", "mean_match", "none"]
    )

    # 9 Multi-scan
    cfg["step09_multiscan_n_scans"] = trial.suggest_int("step09_multiscan_n_scans", 1, 20)
    cfg["step09_multiscan_drift"] = trial.suggest_float("step09_multiscan_drift", 0.0, 0.02)

    # 10 Baseline
    cfg["step10_baseline_enable"] = trial.suggest_categorical("step10_baseline_enable", [False, True])
    cfg["step10_baseline_order"] = trial.suggest_int("step10_baseline_order", 0, 4)
    cfg["step10_baseline_amplitude"] = trial.suggest_float("step10_baseline_amplitude", 0.0, 0.1)
    cfg["step10_baseline_drift"] = trial.suggest_float("step10_baseline_drift", 0.0, 0.05)

    # 11 Scatter
    cfg["step11_scatter_enable"] = trial.suggest_categorical("step11_scatter_enable", [False, True])
    cfg["step11_scatter_model"] = trial.suggest_categorical("step11_scatter_model", ["emsc", "rayleigh", "mie"])
    cfg["step11_scatter_alpha_std"] = trial.suggest_float("step11_scatter_alpha_std", 0.0, 0.2)
    cfg["step11_scatter_beta_std"] = trial.suggest_float("step11_scatter_beta_std", 0.0, 0.05)

    # 12 Environment
    cfg["step12_environment_enable"] = trial.suggest_categorical("step12_environment_enable", [False, True])
    cfg["step12_environment_temp_shift"] = trial.suggest_float("step12_environment_temp_shift", -2.0, 2.0)
    cfg["step12_environment_temp_broadening"] = trial.suggest_float("step12_environment_temp_broadening", 0.0, 2.0)
    cfg["step12_environment_water_bound_free_ratio"] = trial.suggest_float(
        "step12_environment_water_bound_free_ratio", 0.0, 2.0
    )

    # 13 Wavelength distortion
    cfg["step13_wavelength_distortion_shift_std"] = trial.suggest_float(
        "step13_wavelength_distortion_shift_std", 0.0, 2.0
    )
    cfg["step13_wavelength_distortion_stretch_std"] = trial.suggest_float(
        "step13_wavelength_distortion_stretch_std", 0.0, 0.005
    )
    cfg["step13_wavelength_distortion_edge_effects"] = trial.suggest_float(
        "step13_wavelength_distortion_edge_effects", 0.0, 0.2
    )

    # 14 Noise
    cfg["step14_noise_enable"] = trial.suggest_categorical("step14_noise_enable", [False, True])
    cfg["step14_noise_base"] = trial.suggest_float("step14_noise_base", 0.0, 0.02)
    cfg["step14_noise_signal_dep"] = trial.suggest_float("step14_noise_signal_dep", 0.0, 0.05)
    cfg["step14_noise_one_over_f"] = trial.suggest_float("step14_noise_one_over_f", 0.0, 0.02)
    cfg["step14_noise_shot"] = trial.suggest_float("step14_noise_shot", 0.0, 0.02)

    # 15 Preprocess
    cfg["step15_preprocess_smooth_enable"] = trial.suggest_categorical("step15_preprocess_smooth_enable", [False, True])
    cfg["step15_preprocess_smooth_window"] = trial.suggest_int("step15_preprocess_smooth_window", 3, 51, step=2)
    cfg["step15_preprocess_smooth_order"] = trial.suggest_int("step15_preprocess_smooth_order", 1, 5)
    cfg["step15_preprocess_derivative_order"] = trial.suggest_int("step15_preprocess_derivative_order", 0, 2)
    cfg["step15_preprocess_snv_enable"] = trial.suggest_categorical("step15_preprocess_snv_enable", [False, True])

    # 16 Resample
    cfg["step16_resample_enable"] = trial.suggest_categorical("step16_resample_enable", [False, True])
    cfg["step16_resample_target_step"] = trial.suggest_float("step16_resample_target_step", 0.5, 10.0)
    cfg["step16_resample_method"] = trial.suggest_categorical("step16_resample_method", ["linear", "cubic", "spline"])

    # 17 Batch
    cfg["step17_batch_enable"] = trial.suggest_categorical("step17_batch_enable", [False, True])
    cfg["step17_batch_n_batches"] = trial.suggest_int("step17_batch_n_batches", 0, 10)
    cfg["step17_batch_gain_std"] = trial.suggest_float("step17_batch_gain_std", 0.0, 0.1)
    cfg["step17_batch_offset_std"] = trial.suggest_float("step17_batch_offset_std", 0.0, 0.1)

    # 18 Metadata
    cfg["step18_metadata_groups"] = trial.suggest_int("step18_metadata_groups", 0, 10)
    cfg["step18_metadata_repetitions_min"] = trial.suggest_int("step18_metadata_repetitions_min", 1, 5)
    cfg["step18_metadata_repetitions_max"] = trial.suggest_int("step18_metadata_repetitions_max", 1, 10)
    cfg["step18_metadata_train_ratio"] = trial.suggest_float("step18_metadata_train_ratio", 0.5, 0.95)

    # 19 Export
    cfg["step19_export_format"] = trial.suggest_categorical("step19_export_format", ["dataset", "arrays", "csv"])
    cfg["step19_export_compression"] = trial.suggest_categorical("step19_export_compression", ["none", "gzip", "zip"])
    cfg["step19_export_file_layout"] = trial.suggest_categorical(
        "step19_export_file_layout", ["standard", "single", "fragmented"]
    )

    # 20 Targets
    cfg["step20_target_mode"] = trial.suggest_categorical("step20_target_mode", ["regression", "classification"])
    cfg["step20_target_distribution"] = trial.suggest_categorical(
        "step20_target_distribution", ["uniform", "normal", "lognormal", "bimodal"]
    )
    cfg["step20_target_range_min"] = trial.suggest_float("step20_target_range_min", 0.0, 50.0)
    cfg["step20_target_range_max"] = trial.suggest_float("step20_target_range_max", 50.0, 200.0)
    cfg["step20_target_component"] = trial.suggest_categorical(
        "step20_target_component", ["protein", "water", "lipid", "starch"]
    )
    cfg["step20_target_correlation"] = trial.suggest_float("step20_target_correlation", 0.0, 1.0)
    cfg["step20_target_noise"] = trial.suggest_float("step20_target_noise", 0.0, 0.5)

    # 21 Multi-target
    cfg["step21_multitarget_n_targets"] = trial.suggest_int("step21_multitarget_n_targets", 1, 5)
    cfg["step21_multitarget_components"] = ["protein"]  # peut être géré via un sampler custom

    # 22 Classification
    cfg["step22_classification_n_classes"] = trial.suggest_int("step22_classification_n_classes", 2, 6)
    cfg["step22_classification_separation"] = trial.suggest_float("step22_classification_separation", 0.2, 3.0)
    cfg["step22_classification_separation_method"] = trial.suggest_categorical(
        "step22_classification_separation_method", ["component", "threshold", "cluster"]
    )

    # 23 Nonlinear
    cfg["step23_nonlinear_interactions"] = trial.suggest_categorical(
        "step23_nonlinear_interactions", ["none", "polynomial", "synergistic", "antagonistic"]
    )
    cfg["step23_interaction_strength"] = trial.suggest_float("step23_interaction_strength", 0.0, 1.0)
    cfg["step23_hidden_factors"] = trial.suggest_int("step23_hidden_factors", 0, 5)
    cfg["step23_polynomial_degree"] = trial.suggest_int("step23_polynomial_degree", 2, 4)

    # 24 Confounds
    cfg["step24_confounds_signal_to_confound_ratio"] = trial.suggest_float(
        "step24_confounds_signal_to_confound_ratio", 0.4, 1.0
    )
    cfg["step24_confounds_n_confounders"] = trial.suggest_int("step24_confounds_n_confounders", 0, 5)
    cfg["step24_confounds_spectral_masking"] = trial.suggest_float("step24_confounds_spectral_masking", 0.0, 1.0)
    cfg["step24_confounds_temporal_drift"] = trial.suggest_categorical(
        "step24_confounds_temporal_drift", [False, True]
    )
    cfg["step24_confounds_noise_heteroscedasticity"] = trial.suggest_float(
        "step24_confounds_noise_heteroscedasticity", 0.0, 1.0
    )

    # 25 Regimes
    cfg["step25_regimes_n_regimes"] = trial.suggest_int("step25_regimes_n_regimes", 1, 5)
    cfg["step25_regimes_method"] = trial.suggest_categorical(
        "step25_regimes_method", ["concentration", "spectral", "random"]
    )
    cfg["step25_regimes_overlap"] = trial.suggest_float("step25_regimes_overlap", 0.0, 0.8)

    # Post-process ranges & tuples
    cfg["step18_metadata_repetitions"] = [
        min(cfg["step18_metadata_repetitions_min"], cfg["step18_metadata_repetitions_max"]),
        max(cfg["step18_metadata_repetitions_min"], cfg["step18_metadata_repetitions_max"])
    ]
    cfg["step20_target_range"] = [
        min(cfg["step20_target_range_min"], cfg["step20_target_range_max"]),
        max(cfg["step20_target_range_min"], cfg["step20_target_range_max"])
    ]

    return cfg