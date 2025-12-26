# Réorganisation des Exemples nirs4all

## Résumé Exécutif

Ce document propose une réorganisation complète des exemples nirs4all avec les objectifs suivants :
1. **Walkthrough progressif** : Du "Hello World" aux pipelines les plus complexes
2. **Deux parcours** : Utilisateurs et Développeurs
3. **Nouvelle API** : Migration vers `nirs4all.run()`, `.predict()`, etc.
4. **Couverture exhaustive** : Tous les features couverts par au moins un exemple
5. **Tests d'intégration** : Les exemples servent de tests via `run.sh`/`run.ps1`

---

## Table des Matières

1. [Philosophie de la Réorganisation](#1-philosophie-de-la-réorganisation)
2. [Structure Proposée](#2-structure-proposée)
3. [Parcours Utilisateur (U)](#3-parcours-utilisateur-u)
4. [Parcours Développeur (D)](#4-parcours-développeur-d)
5. [Exemples Spéciaux](#5-exemples-spéciaux)
6. [Mapping Ancien → Nouveau](#6-mapping-ancien--nouveau)
7. [Features Coverage Matrix](#7-features-coverage-matrix)
8. [Scripts run.sh / run.ps1](#8-scripts-runsh--runps1)
9. [Intégration RTD](#9-intégration-rtd)
10. [Plan d'Implémentation](#10-plan-dimplémentation)

---

## 1. Philosophie de la Réorganisation

### Types d'Exemples

| Type | Audience | Longueur | Complexité | Objectif |
|------|----------|----------|------------|----------|
| **Tutorial** | Utilisateurs | Court (~100-200 lignes) | 1-4 pipelines | Boilerplate + explication d'UN concept |
| **Feature Guide** | Mixte | Moyen (~200-400 lignes) | 3-8 pipelines | Couverture d'un feature spécifique |
| **Deep Dive** | Développeurs | Long (~400-800 lignes) | Complet | Exploration exhaustive d'une fonctionnalité |
| **Reference** | Développeurs | Variable | N/A | Documentation des syntaxes et APIs |

### Conventions de Nommage

```
U01_hello_world.py          # U = User, 01 = ordre, nom descriptif
U02_basic_regression.py
...
D01_custom_transformers.py  # D = Developer
D02_controller_system.py
...
R01_pipeline_syntax.py      # R = Reference (syntaxe, config)
R02_generator_syntax.py
```

### Principes

1. **Un concept par exemple tutorial** : Pas 10 features dans un seul fichier
2. **Progression naturelle** : Chaque exemple peut référencer les précédents
3. **Autonomie** : Chaque exemple peut s'exécuter indépendamment
4. **Documentation intégrée** : Docstrings détaillées en tête de fichier
5. **Reproductibilité** : Utilisation de `random_state` quand applicable

---

## 2. Structure Proposée

```
examples/
├── README.md                    # Index des exemples avec liens
├── run.sh                       # Script pour tests (Linux/Mac)
├── run.ps1                      # Script pour tests (Windows)
├── sample_data/                 # Données d'exemple
│
├── user/                        # Parcours Utilisateur
│   ├── 01_getting_started/
│   │   ├── U01_hello_world.py
│   │   ├── U02_basic_regression.py
│   │   ├── U03_basic_classification.py
│   │   └── U04_visualization.py
│   │
│   ├── 02_data_handling/
│   │   ├── U05_flexible_inputs.py
│   │   ├── U06_multi_datasets.py
│   │   ├── U07_multi_source.py
│   │   └── U08_wavelength_handling.py
│   │
│   ├── 03_preprocessing/
│   │   ├── U09_preprocessing_basics.py
│   │   ├── U10_feature_augmentation.py
│   │   ├── U11_sample_augmentation.py
│   │   └── U12_signal_conversion.py
│   │
│   ├── 04_models/
│   │   ├── U13_multi_model.py
│   │   ├── U14_hyperparameter_tuning.py
│   │   ├── U15_stacking_ensembles.py
│   │   └── U16_pls_variants.py
│   │
│   ├── 05_cross_validation/
│   │   ├── U17_cv_strategies.py
│   │   ├── U18_group_splitting.py
│   │   ├── U19_sample_filtering.py
│   │   └── U20_aggregation.py
│   │
│   ├── 06_deployment/
│   │   ├── U21_save_load_predict.py
│   │   ├── U22_export_bundle.py
│   │   ├── U23_workspace_management.py
│   │   └── U24_sklearn_integration.py
│   │
│   └── 07_explainability/
│       ├── U25_shap_basics.py
│       ├── U26_shap_sklearn.py
│       └── U27_feature_selection.py
│
├── developer/                   # Parcours Développeur
│   ├── 01_advanced_pipelines/
│   │   ├── D01_branching_basics.py
│   │   ├── D02_branching_advanced.py
│   │   ├── D03_merge_strategies.py
│   │   ├── D04_source_branching.py
│   │   └── D05_meta_stacking.py
│   │
│   ├── 02_generators/
│   │   ├── D06_generator_basics.py
│   │   ├── D07_generator_advanced.py
│   │   ├── D08_generator_nested.py
│   │   └── D09_constraints_presets.py
│   │
│   ├── 03_deep_learning/
│   │   ├── D10_nicon_tensorflow.py
│   │   ├── D11_pytorch_models.py
│   │   ├── D12_jax_models.py
│   │   └── D13_framework_comparison.py
│   │
│   ├── 04_transfer_learning/
│   │   ├── D14_retrain_modes.py
│   │   ├── D15_transfer_analysis.py
│   │   └── D16_domain_adaptation.py
│   │
│   ├── 05_advanced_features/
│   │   ├── D17_outlier_partitioning.py
│   │   ├── D18_metadata_branching.py
│   │   ├── D19_repetition_transform.py
│   │   └── D20_concat_transform.py
│   │
│   └── 06_internals/
│       ├── D21_session_workflow.py
│       └── D22_custom_controllers.py
│
├── reference/                   # Exemples de Référence
│   ├── R01_pipeline_syntax.py       # Toutes les syntaxes de pipeline
│   ├── R02_generator_reference.py   # Syntaxe complète des générateurs
│   ├── R03_all_keywords.py          # Test de TOUS les keywords
│   └── R04_legacy_api.py            # Ancienne API (pour référence)
│
└── benchmarks/                  # Benchmarks (optionnel, hors tests)
    └── baseline_sota.py
```

---

## 3. Parcours Utilisateur (U)

### 01_getting_started/ - Démarrage Rapide

#### U01_hello_world.py
**Objectif** : Premier pipeline en 20 lignes
**Concepts** : `nirs4all.run()`, pipeline minimal, lecture des résultats
**Complexité** : ★☆☆☆☆
**Pipelines** : 1

```python
"""
U01 - Hello World : Votre premier pipeline nirs4all
====================================================
Le pipeline le plus simple possible : MinMaxScaler + PLS.
Apprenez à utiliser nirs4all.run() et à lire les résultats.
"""
import nirs4all
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

result = nirs4all.run(
    pipeline=[
        MinMaxScaler(),
        ShuffleSplit(n_splits=3, test_size=0.25),
        {"model": PLSRegression(n_components=10)}
    ],
    dataset="sample_data/regression",
    name="HelloWorld",
    verbose=1
)

print(f"Best RMSE: {result.best_rmse:.4f}")
print(f"Best R²: {result.best_r2:.4f}")
```

#### U02_basic_regression.py
**Source** : Fusion de Q1_regression.py (simplifié)
**Objectif** : Pipeline de régression avec preprocessing et visualisation
**Concepts** : Preprocessing NIRS (SNV, MSC), PredictionAnalyzer
**Complexité** : ★★☆☆☆
**Pipelines** : 3 (différents n_components)

#### U03_basic_classification.py
**Source** : Adaptation de Q1_classif.py
**Objectif** : Classification avec RandomForest et XGBoost
**Concepts** : Classification, confusion matrix
**Complexité** : ★★☆☆☆
**Pipelines** : 2

#### U04_visualization.py
**Source** : Nouveau (extrait des parties viz des autres exemples)
**Objectif** : Tour d'horizon des visualisations
**Concepts** : PredictionAnalyzer, heatmaps, candlestick, top-k
**Complexité** : ★★☆☆☆
**Pipelines** : 1 (focus sur analyse)

---

### 02_data_handling/ - Gestion des Données

#### U05_flexible_inputs.py
**Source** : Q11_flexible_inputs.py
**Objectif** : Différents formats d'entrée (numpy, path, dict, tuple)
**Concepts** : DatasetConfigs, formats supportés
**Complexité** : ★☆☆☆☆
**Pipelines** : 1

#### U06_multi_datasets.py
**Source** : Q4_multidatasets.py
**Objectif** : Analyser plusieurs datasets en parallèle
**Concepts** : Liste de datasets, comparaison
**Complexité** : ★★☆☆☆
**Pipelines** : 2

#### U07_multi_source.py
**Source** : Q6_multisource.py
**Objectif** : Données multi-sources (NIR + autres capteurs)
**Concepts** : Sources multiples, merge_sources basic
**Complexité** : ★★★☆☆
**Pipelines** : 2

#### U08_wavelength_handling.py
**Source** : Q10_resampler.py + Q13_nm_headers.py
**Objectif** : Gestion des longueurs d'onde (interpolation, unités)
**Concepts** : Resampling, header_unit, conversion nm/cm⁻¹
**Complexité** : ★★☆☆☆
**Pipelines** : 2

---

### 03_preprocessing/ - Prétraitement

#### U09_preprocessing_basics.py
**Source** : Nouveau (consolidation)
**Objectif** : Tour des prétraitements NIRS standards
**Concepts** : SNV, MSC, Detrend, Derivatives, Gaussian, SavitzkyGolay
**Complexité** : ★★☆☆☆
**Pipelines** : 3 (comparaison)

#### U10_feature_augmentation.py
**Source** : Q_feature_augmentation_modes.py
**Objectif** : Modes d'augmentation de features (extend, add, replace)
**Concepts** : feature_augmentation, actions modes
**Complexité** : ★★★☆☆
**Pipelines** : 3

#### U11_sample_augmentation.py
**Source** : Q12_sample_augmentation.py (simplifié)
**Objectif** : Augmentation des échantillons (bruit, drift, etc.)
**Concepts** : sample_augmentation, transformers, balancing
**Complexité** : ★★★☆☆
**Pipelines** : 2

#### U12_signal_conversion.py
**Source** : Q29_signal_conversion.py
**Objectif** : Conversion entre types de signaux
**Concepts** : Absorbance, Reflectance, Transmittance, Kubelka-Munk
**Complexité** : ★★☆☆☆
**Pipelines** : 2

---

### 04_models/ - Modèles

#### U13_multi_model.py
**Source** : Q2_multimodel.py
**Objectif** : Comparer plusieurs modèles
**Concepts** : PLS, RF, Ridge, XGBoost, comparaison
**Complexité** : ★★☆☆☆
**Pipelines** : 4+

#### U14_hyperparameter_tuning.py
**Source** : Q3_finetune.py
**Objectif** : Optimisation avec Optuna
**Concepts** : finetune_params, grid/hyperband, best/mean eval
**Complexité** : ★★★☆☆
**Pipelines** : 2

#### U15_stacking_ensembles.py
**Source** : Q18_stacking.py
**Objectif** : Stacking et Voting ensembles
**Concepts** : StackingRegressor, VotingRegressor
**Complexité** : ★★★☆☆
**Pipelines** : 3

#### U16_pls_variants.py
**Source** : Q19_pls_methods.py (simplifié)
**Objectif** : Variantes PLS courantes
**Concepts** : PLSR, OPLS, SparsePLS, iPLS
**Complexité** : ★★★☆☆
**Pipelines** : 4-6

---

### 05_cross_validation/ - Validation Croisée

#### U17_cv_strategies.py
**Source** : Nouveau (consolidation)
**Objectif** : Stratégies de CV standard
**Concepts** : KFold, ShuffleSplit, RepeatedKFold
**Complexité** : ★★☆☆☆
**Pipelines** : 3

#### U18_group_splitting.py
**Source** : Q2_groupsplit.py + Q2B_force_group.py
**Objectif** : Splitting par groupes
**Concepts** : GroupKFold, force_group, metadata
**Complexité** : ★★★☆☆
**Pipelines** : 2

#### U19_sample_filtering.py
**Source** : Q28_sample_filtering.py (simplifié)
**Objectif** : Filtrage des outliers
**Concepts** : YOutlierFilter, XOutlierFilter, IQR, Z-score
**Complexité** : ★★★☆☆
**Pipelines** : 3

#### U20_aggregation.py
**Source** : Q34_aggregation.py
**Objectif** : Agrégation des répétitions
**Concepts** : aggregate column, métriques raw vs aggregated
**Complexité** : ★★☆☆☆
**Pipelines** : 2

---

### 06_deployment/ - Déploiement

#### U21_save_load_predict.py
**Source** : Q5_predict.py + Q5_predict_NN.py (fusionnés)
**Objectif** : Sauvegarder, charger et prédire
**Concepts** : nirs4all.predict(), 3 méthodes de chargement
**Complexité** : ★★☆☆☆
**Pipelines** : 1 + prédictions

#### U22_export_bundle.py
**Source** : Q32_export_bundle.py
**Objectif** : Export en bundle .n4a
**Concepts** : result.export(), portabilité, scripts auto-générés
**Complexité** : ★★☆☆☆
**Pipelines** : 1 + export

#### U23_workspace_management.py
**Source** : Q14_workspace.py
**Objectif** : Gestion du workspace et artifacts
**Concepts** : workspace_path, library, manifests
**Complexité** : ★★☆☆☆
**Pipelines** : 2

#### U24_sklearn_integration.py
**Source** : Q_sklearn_wrapper.py
**Objectif** : Intégration sklearn avec NIRSPipeline
**Concepts** : NIRSPipeline.from_result(), .from_bundle()
**Complexité** : ★★★☆☆
**Pipelines** : 1 + intégration

---

### 07_explainability/ - Explicabilité

#### U25_shap_basics.py
**Source** : Q8_shap.py (simplifié)
**Objectif** : Introduction à SHAP pour NIRS
**Concepts** : nirs4all.explain(), spectral importance
**Complexité** : ★★★☆☆
**Pipelines** : 1 + SHAP

#### U26_shap_sklearn.py
**Source** : Q41_sklearn_shap.py
**Objectif** : SHAP avec le wrapper sklearn
**Concepts** : NIRSPipeline + KernelExplainer
**Complexité** : ★★★☆☆
**Pipelines** : 1 + SHAP

#### U27_feature_selection.py
**Source** : Q21_feature_selection.py
**Objectif** : Sélection de longueurs d'onde
**Concepts** : CARS, MC-UVE, feature importance
**Complexité** : ★★★☆☆
**Pipelines** : 2

---

## 4. Parcours Développeur (D)

### 01_advanced_pipelines/ - Pipelines Avancés

#### D01_branching_basics.py
**Source** : Q30_branching.py (première moitié)
**Objectif** : Introduction au branching
**Concepts** : branch keyword, syntaxe list/dict, merge basics
**Complexité** : ★★★☆☆
**Pipelines** : 4-5

#### D02_branching_advanced.py
**Source** : Q30_branching.py (seconde moitié) + Q31_outlier_branching.py
**Objectif** : Branching avancé et statistiques
**Concepts** : BranchAnalyzer, comparaisons statistiques, HTML reports
**Complexité** : ★★★★☆
**Pipelines** : 5-6

#### D03_merge_strategies.py
**Source** : Q_merge_branches.py
**Objectif** : Stratégies de merge
**Concepts** : merge features/predictions, per-branch selection, OOF
**Complexité** : ★★★★☆
**Pipelines** : 6+

#### D04_source_branching.py
**Source** : Q_merge_sources.py
**Objectif** : Branching par source de données
**Concepts** : source_branch, merge_sources, concat/stack
**Complexité** : ★★★★☆
**Pipelines** : 4-5

#### D05_meta_stacking.py
**Source** : Q_meta_stacking.py
**Objectif** : Stacking multi-niveaux
**Concepts** : MetaModel, StackingConfig, OOF reconstruction
**Complexité** : ★★★★★
**Pipelines** : 8+

---

### 02_generators/ - Générateurs

#### D06_generator_basics.py
**Source** : Q23b_generator.py
**Objectif** : Introduction aux générateurs
**Concepts** : _or_, pick, count, _range_
**Complexité** : ★★★☆☆
**Pipelines** : 6+

#### D07_generator_advanced.py
**Source** : Q23_generator_syntax.py (première partie)
**Objectif** : Générateurs avancés
**Concepts** : _grid_, _zip_, _chain_, _sample_
**Complexité** : ★★★★☆
**Pipelines** : 10+

#### D08_generator_nested.py
**Source** : Q26_nested_or_preprocessing.py
**Objectif** : Générateurs imbriqués
**Concepts** : _cartesian_, nested _or_, combinatorics
**Complexité** : ★★★★★
**Pipelines** : 20+

#### D09_constraints_presets.py
**Source** : Q24_generator_advanced.py
**Objectif** : Contraintes et presets
**Concepts** : constraints, presets, iterators, export utilities
**Complexité** : ★★★★★
**Pipelines** : Variable

---

### 03_deep_learning/ - Deep Learning

#### D10_nicon_tensorflow.py
**Source** : X3_hiba_full.py (partie TensorFlow)
**Objectif** : Modèles nicon TensorFlow
**Concepts** : nicon, customizable_nicon, architectures CNN
**Complexité** : ★★★★☆
**Pipelines** : 3-4

#### D11_pytorch_models.py
**Source** : Q16_pytorch_models.py
**Objectif** : Modèles PyTorch custom
**Concepts** : @framework decorator, custom layers, hyperband
**Complexité** : ★★★★☆
**Pipelines** : 3-4

#### D12_jax_models.py
**Source** : Q15_jax_models.py
**Objectif** : Modèles JAX/Flax
**Concepts** : JaxMLPRegressor, nicon_jax, finetuning
**Complexité** : ★★★★☆
**Pipelines** : 3-4

#### D13_framework_comparison.py
**Source** : Q17_nicon_comparison.py
**Objectif** : Comparaison TF/PyTorch/JAX
**Concepts** : Framework abstraction, parity testing
**Complexité** : ★★★★★
**Pipelines** : 6+ (2 par framework)

---

### 04_transfer_learning/ - Transfer Learning

#### D14_retrain_modes.py
**Source** : Q33_retrain_transfer.py
**Objectif** : Modes de réentraînement
**Concepts** : full retrain, transfer, finetune, step control
**Complexité** : ★★★★☆
**Pipelines** : 4

#### D15_transfer_analysis.py
**Source** : Q27_transfer_analysis.py
**Objectif** : Analyse de transfert preprocessing
**Concepts** : TransferPreprocessingSelector, domain adaptation
**Complexité** : ★★★★☆
**Pipelines** : 3

#### D16_domain_adaptation.py
**Source** : Q9_acp_spread.py
**Objectif** : Analyse PCA pour transfer
**Concepts** : PreprocPCAEvaluator, geometry preservation
**Complexité** : ★★★★★
**Pipelines** : 1 (focus analyse)

---

### 05_advanced_features/ - Features Avancés

#### D17_outlier_partitioning.py
**Source** : Q31_outlier_branching.py
**Objectif** : Partitionnement par outliers
**Concepts** : sample_partitioner, outlier_excluder, isolation forest
**Complexité** : ★★★★☆
**Pipelines** : 3

#### D18_metadata_branching.py
**Source** : Q35_metadata_branching.py
**Objectif** : Branching par métadonnées
**Concepts** : metadata_partitioner, per-branch CV, value grouping
**Complexité** : ★★★★☆
**Pipelines** : 4

#### D19_repetition_transform.py
**Source** : Q36_repetition_transform.py
**Objectif** : Transformation des répétitions
**Concepts** : rep_to_sources, rep_to_pp, RepetitionConfig
**Complexité** : ★★★★☆
**Pipelines** : N/A (transformation only)

#### D20_concat_transform.py
**Source** : Q22_concat_transform.py
**Objectif** : Concaténation de transformateurs
**Concepts** : concat_transform, PCA+SVD features
**Complexité** : ★★★☆☆
**Pipelines** : 2

---

### 06_internals/ - Internals

#### D21_session_workflow.py
**Source** : Q42_session_workflow.py
**Objectif** : Workflow avec sessions
**Concepts** : nirs4all.session(), resource sharing, batch experiments
**Complexité** : ★★★☆☆
**Pipelines** : 10+ (dans sessions)

#### D22_custom_controllers.py
**Source** : Nouveau
**Objectif** : Créer des controllers custom
**Concepts** : @register_controller, OperatorController, priority
**Complexité** : ★★★★★
**Pipelines** : 1-2 (démo)

---

## 5. Exemples Spéciaux

### Reference/ - Documentation de Référence

#### R01_pipeline_syntax.py
**Source** : X0_pipeline_sample.py (renommé)
**Objectif** : Documentation de TOUTES les syntaxes de pipeline
**Format** : Commentaires détaillés, pas d'exécution requise

#### R02_generator_reference.py
**Source** : Extraction de Q23_generator_syntax.py
**Objectif** : Référence complète syntaxe générateurs
**Format** : Documentation + exemples courts

#### R03_all_keywords.py
**Source** : Q_complex_all_keywords.py
**Objectif** : Test de TOUS les keywords pipeline
**Note** : Garde pour tests d'intégration, mais ne fait pas partie du parcours normal

#### R04_legacy_api.py
**Source** : Q40_new_api.py (partie ancienne API uniquement)
**Objectif** : Référence de l'ancienne API PipelineRunner
**Note** : 1-2 exemples pour référence et migration

---

## 6. Mapping Ancien → Nouveau

| Ancien | Nouveau | Notes |
|--------|---------|-------|
| Q1_classif.py | U03_basic_classification.py | Simplifié, nouvelle API |
| Q1_regression.py | U02_basic_regression.py | Simplifié, nouvelle API |
| Q2_groupsplit.py | U18_group_splitting.py | Fusionné avec Q2B |
| Q2_multimodel.py | U13_multi_model.py | Nouvelle API |
| Q2B_force_group.py | U18_group_splitting.py | Fusionné |
| Q3_finetune.py | U14_hyperparameter_tuning.py | Nouvelle API |
| Q4_multidatasets.py | U06_multi_datasets.py | Nouvelle API |
| Q5_predict.py | U21_save_load_predict.py | Fusionné avec Q5_predict_NN |
| Q5_predict_NN.py | U21_save_load_predict.py | Fusionné |
| Q6_multisource.py | U07_multi_source.py | Nouvelle API |
| Q7_discretization.py | (intégré dans U03) | Fonctionnalité mineure |
| Q8_shap.py | U25_shap_basics.py | Simplifié, nouvelle API |
| Q9_acp_spread.py | D16_domain_adaptation.py | Renommé |
| Q10_resampler.py | U08_wavelength_handling.py | Fusionné avec Q13 |
| Q11_flexible_inputs.py | U05_flexible_inputs.py | Nouvelle API |
| Q12_sample_augmentation.py | U11_sample_augmentation.py | Simplifié |
| Q13_nm_headers.py | U08_wavelength_handling.py | Fusionné |
| Q14_workspace.py | U23_workspace_management.py | Nouvelle API |
| Q15_jax_models.py | D12_jax_models.py | Maintenu |
| Q16_pytorch_models.py | D11_pytorch_models.py | Maintenu |
| Q17_nicon_comparison.py | D13_framework_comparison.py | Maintenu |
| Q18_stacking.py | U15_stacking_ensembles.py | Nouvelle API |
| Q19_pls_methods.py | U16_pls_variants.py | Simplifié |
| Q21_feature_selection.py | U27_feature_selection.py | Nouvelle API |
| Q22_concat_transform.py | D20_concat_transform.py | Maintenu |
| Q23_generator_syntax.py | D07_generator_advanced.py | Réorganisé |
| Q23b_generator.py | D06_generator_basics.py | Renommé |
| Q24_generator_advanced.py | D09_constraints_presets.py | Renommé |
| Q25_complex_pipeline_pls.py | (supprimé) | Redondant |
| Q26_nested_or_preprocessing.py | D08_generator_nested.py | Renommé |
| Q27_transfer_analysis.py | D15_transfer_analysis.py | Maintenu |
| Q28_sample_filtering.py | U19_sample_filtering.py | Simplifié |
| Q29_signal_conversion.py | U12_signal_conversion.py | Nouvelle API |
| Q30_branching.py | D01/D02_branching_*.py | Divisé |
| Q31_outlier_branching.py | D17_outlier_partitioning.py | Renommé |
| Q32_export_bundle.py | U22_export_bundle.py | Nouvelle API |
| Q33_retrain_transfer.py | D14_retrain_modes.py | Renommé |
| Q34_aggregation.py | U20_aggregation.py | Nouvelle API |
| Q35_metadata_branching.py | D18_metadata_branching.py | Maintenu |
| Q36_repetition_transform.py | D19_repetition_transform.py | Maintenu |
| Q40_new_api.py | (supprimé/intégré) | Distribué dans tous |
| Q41_sklearn_shap.py | U26_shap_sklearn.py | Renommé |
| Q42_session_workflow.py | D21_session_workflow.py | Maintenu |
| Q_complex_all_keywords.py | R03_all_keywords.py | Reference |
| Q_feature_augmentation_modes.py | U10_feature_augmentation.py | Renommé |
| Q_merge_branches.py | D03_merge_strategies.py | Renommé |
| Q_merge_sources.py | D04_source_branching.py | Renommé |
| Q_meta_stacking.py | D05_meta_stacking.py | Renommé |
| Q_sklearn_wrapper.py | U24_sklearn_integration.py | Renommé |
| X0_pipeline_sample.py | R01_pipeline_syntax.py | Reference |
| X1_metadata.py | (supprimé) | Intégré ailleurs |
| X2_sample_augmentation.py | (supprimé) | Redondant avec U11 |
| X3_hiba_full.py | D10_nicon_tensorflow.py | Simplifié |
| X4_features.py | (supprimé) | Test interne |
| baseline_sota.py | benchmarks/baseline_sota.py | Déplacé |

---

## 7. Features Coverage Matrix

| Feature | Exemple Principal | Exemples Secondaires |
|---------|------------------|---------------------|
| **Core API** | | |
| `nirs4all.run()` | U01, U02 | Tous les U* |
| `nirs4all.predict()` | U21 | U22 |
| `nirs4all.explain()` | U25 | U26 |
| `nirs4all.retrain()` | D14 | D15 |
| `nirs4all.session()` | D21 | - |
| **Pipeline Syntax** | | |
| preprocessing | U09 | Tous |
| y_processing | U02 | U14 |
| feature_augmentation | U10 | U02, D06 |
| sample_augmentation | U11 | D02 |
| concat_transform | D20 | R03 |
| branch | D01 | D02, D03 |
| merge | D03 | D05, R03 |
| source_branch | D04 | R03 |
| merge_sources | D04 | U07 |
| model | U01 | Tous |
| **Generators** | | |
| _or_, pick, count | D06 | U10 |
| _range_ | D06 | U14 |
| _grid_, _zip_, _chain_ | D07 | - |
| _sample_ | D07 | - |
| _cartesian_ | D08 | - |
| constraints | D09 | - |
| presets | D09 | - |
| **Cross-Validation** | | |
| KFold | U17 | U02 |
| ShuffleSplit | U01 | Nombreux |
| GroupKFold | U18 | - |
| force_group | U18 | - |
| **Data Handling** | | |
| Multi-dataset | U06 | - |
| Multi-source | U07 | D04 |
| Flexible inputs | U05 | - |
| Wavelength resampling | U08 | - |
| Header units | U08 | - |
| Aggregation | U20 | - |
| **Models** | | |
| PLS variants | U16 | U02 |
| sklearn models | U13 | Nombreux |
| Stacking/Voting | U15 | D05 |
| TensorFlow (nicon) | D10 | D13 |
| PyTorch | D11 | D13 |
| JAX | D12 | D13 |
| **Hyperparameter Tuning** | | |
| finetune_params | U14 | D11, D12 |
| Optuna integration | U14 | - |
| **Explainability** | | |
| SHAP | U25 | U26 |
| Feature selection | U27 | - |
| **Deployment** | | |
| Model persistence | U21 | - |
| Bundle export | U22 | - |
| NIRSPipeline wrapper | U24 | U26 |
| Workspace management | U23 | - |
| **Advanced Features** | | |
| Outlier filtering | U19 | D17 |
| Metadata branching | D18 | - |
| Transfer learning | D14, D15 | D16 |
| Repetition transform | D19 | - |
| **Deep Learning** | | |
| Custom architectures | D10, D11 | D12 |
| Framework comparison | D13 | - |

---

## 8. Scripts run.sh / run.ps1

### Nouveau run.sh

```bash
#!/usr/bin/env bash
set -uo pipefail

# Usage: ./run.sh [-i index] [-b begin] [-e end] [-n name] [-c category] [-l] [-p] [-s] [-q]
#   -c category: user, developer, reference, all (default: all)
#   -q: quick mode (skip deep learning examples)

INDEX=0
BEGIN=0
END=0
NAME=""
CATEGORY="all"
LOG=0
PLOT=0
SHOW=0
QUICK=0

while getopts "i:b:e:n:c:lpsq" opt; do
  case "$opt" in
    i) INDEX="$OPTARG" ;;
    b) BEGIN="$OPTARG" ;;
    e) END="$OPTARG" ;;
    n) NAME="$OPTARG" ;;
    c) CATEGORY="$OPTARG" ;;
    l) LOG=1 ;;
    p) PLOT=1 ;;
    s) SHOW=1 ;;
    q) QUICK=1 ;;
    *) echo "Usage: $0 [-i index] [-b begin] [-e end] [-n name] [-c category] [-l] [-p] [-s] [-q]"; exit 1 ;;
  esac
done
shift $((OPTIND -1))

# Define examples by category
user_examples=(
  # 01_getting_started
  "user/01_getting_started/U01_hello_world.py"
  "user/01_getting_started/U02_basic_regression.py"
  "user/01_getting_started/U03_basic_classification.py"
  "user/01_getting_started/U04_visualization.py"
  # 02_data_handling
  "user/02_data_handling/U05_flexible_inputs.py"
  "user/02_data_handling/U06_multi_datasets.py"
  "user/02_data_handling/U07_multi_source.py"
  "user/02_data_handling/U08_wavelength_handling.py"
  # 03_preprocessing
  "user/03_preprocessing/U09_preprocessing_basics.py"
  "user/03_preprocessing/U10_feature_augmentation.py"
  "user/03_preprocessing/U11_sample_augmentation.py"
  "user/03_preprocessing/U12_signal_conversion.py"
  # 04_models
  "user/04_models/U13_multi_model.py"
  "user/04_models/U14_hyperparameter_tuning.py"
  "user/04_models/U15_stacking_ensembles.py"
  "user/04_models/U16_pls_variants.py"
  # 05_cross_validation
  "user/05_cross_validation/U17_cv_strategies.py"
  "user/05_cross_validation/U18_group_splitting.py"
  "user/05_cross_validation/U19_sample_filtering.py"
  "user/05_cross_validation/U20_aggregation.py"
  # 06_deployment
  "user/06_deployment/U21_save_load_predict.py"
  "user/06_deployment/U22_export_bundle.py"
  "user/06_deployment/U23_workspace_management.py"
  "user/06_deployment/U24_sklearn_integration.py"
  # 07_explainability
  "user/07_explainability/U25_shap_basics.py"
  "user/07_explainability/U26_shap_sklearn.py"
  "user/07_explainability/U27_feature_selection.py"
)

developer_examples=(
  # 01_advanced_pipelines
  "developer/01_advanced_pipelines/D01_branching_basics.py"
  "developer/01_advanced_pipelines/D02_branching_advanced.py"
  "developer/01_advanced_pipelines/D03_merge_strategies.py"
  "developer/01_advanced_pipelines/D04_source_branching.py"
  "developer/01_advanced_pipelines/D05_meta_stacking.py"
  # 02_generators
  "developer/02_generators/D06_generator_basics.py"
  "developer/02_generators/D07_generator_advanced.py"
  "developer/02_generators/D08_generator_nested.py"
  "developer/02_generators/D09_constraints_presets.py"
  # 03_deep_learning (skip in quick mode)
  "developer/03_deep_learning/D10_nicon_tensorflow.py"
  "developer/03_deep_learning/D11_pytorch_models.py"
  "developer/03_deep_learning/D12_jax_models.py"
  "developer/03_deep_learning/D13_framework_comparison.py"
  # 04_transfer_learning
  "developer/04_transfer_learning/D14_retrain_modes.py"
  "developer/04_transfer_learning/D15_transfer_analysis.py"
  "developer/04_transfer_learning/D16_domain_adaptation.py"
  # 05_advanced_features
  "developer/05_advanced_features/D17_outlier_partitioning.py"
  "developer/05_advanced_features/D18_metadata_branching.py"
  "developer/05_advanced_features/D19_repetition_transform.py"
  "developer/05_advanced_features/D20_concat_transform.py"
  # 06_internals
  "developer/06_internals/D21_session_workflow.py"
  "developer/06_internals/D22_custom_controllers.py"
)

reference_examples=(
  "reference/R01_pipeline_syntax.py"
  "reference/R02_generator_reference.py"
  "reference/R03_all_keywords.py"
  "reference/R04_legacy_api.py"
)

# Deep learning examples to skip in quick mode
dl_examples=(
  "D10_nicon_tensorflow.py"
  "D11_pytorch_models.py"
  "D12_jax_models.py"
  "D13_framework_comparison.py"
)

# Build selected examples list based on category
case "$CATEGORY" in
  user)
    selectedExamples=("${user_examples[@]}")
    ;;
  developer)
    selectedExamples=("${developer_examples[@]}")
    ;;
  reference)
    selectedExamples=("${reference_examples[@]}")
    ;;
  all)
    selectedExamples=("${user_examples[@]}" "${developer_examples[@]}" "${reference_examples[@]}")
    ;;
  *)
    echo "Error: Unknown category '$CATEGORY'. Use: user, developer, reference, all"
    exit 1
    ;;
esac

# Filter out DL examples in quick mode
if [ "$QUICK" -eq 1 ]; then
  filtered=()
  for ex in "${selectedExamples[@]}"; do
    skip=0
    for dl in "${dl_examples[@]}"; do
      if [[ "$ex" == *"$dl"* ]]; then
        skip=1
        break
      fi
    done
    if [ "$skip" -eq 0 ]; then
      filtered+=("$ex")
    fi
  done
  selectedExamples=("${filtered[@]}")
  echo "Quick mode: Skipping deep learning examples"
fi

# ... (rest of the script similar to current run.sh)
```

### Nouveau Structure des Tests

```
examples/
├── run.sh                    # Script principal
├── run.ps1                   # Version PowerShell
├── conftest.py               # Configuration pytest (optionnel)
└── test_examples.py          # Tests pytest wrapper (optionnel)
```

---

## 9. Intégration RTD

### Structure Documentation

```
docs/
├── source/
│   ├── tutorials/
│   │   ├── index.rst
│   │   ├── getting_started.rst      # Liens vers U01-U04
│   │   ├── data_handling.rst        # Liens vers U05-U08
│   │   ├── preprocessing.rst        # Liens vers U09-U12
│   │   ├── models.rst               # Liens vers U13-U16
│   │   ├── cross_validation.rst     # Liens vers U17-U20
│   │   ├── deployment.rst           # Liens vers U21-U24
│   │   └── explainability.rst       # Liens vers U25-U27
│   │
│   ├── advanced/
│   │   ├── index.rst
│   │   ├── branching.rst            # Liens vers D01-D05
│   │   ├── generators.rst           # Liens vers D06-D09
│   │   ├── deep_learning.rst        # Liens vers D10-D13
│   │   ├── transfer_learning.rst    # Liens vers D14-D16
│   │   └── advanced_features.rst    # Liens vers D17-D22
│   │
│   └── reference/
│       ├── pipeline_syntax.rst      # R01
│       ├── generator_syntax.rst     # R02
│       └── api.rst                  # Auto-generated
```

### Exemple de Page RTD

```rst
Getting Started
===============

This section introduces the basics of nirs4all through hands-on examples.

Hello World
-----------

Your first pipeline in 20 lines of code.

.. literalinclude:: ../../../examples/user/01_getting_started/U01_hello_world.py
   :language: python
   :caption: U01_hello_world.py
   :linenos:

Key Concepts:
- ``nirs4all.run()`` is the main entry point
- Pipeline is a simple list of steps
- Results are accessed via ``result.best_rmse``

.. seealso::
   - :doc:`../api/run` for full API reference
   - :doc:`basic_regression` for the next step
```

### Annotations dans les Exemples

Chaque exemple inclut un header standard pour RTD :

```python
"""
U01 - Hello World : Votre premier pipeline nirs4all
====================================================

.. currentmodule:: nirs4all

Ce tutoriel couvre :

* L'utilisation de :func:`run` pour entraîner un pipeline
* La structure d'un pipeline minimal
* La lecture des résultats avec :class:`RunResult`

Prérequis
---------
Aucun prérequis, c'est le point de départ !

Étapes suivantes
----------------
Après cet exemple, consultez :ref:`U02_basic_regression` pour découvrir
les prétraitements NIRS.
"""
```

---

## 10. Plan d'Implémentation

### Phase 1 : Structure (1-2 jours)
1. Créer l'arborescence de dossiers
2. Créer le README.md index
3. Mettre à jour run.sh et run.ps1

### Phase 2 : Migration Utilisateur (3-5 jours)
1. Convertir les exemples U01-U10 (getting_started, data_handling)
2. Convertir les exemples U11-U20 (preprocessing, models, CV)
3. Convertir les exemples U21-U27 (deployment, explainability)
4. Tests de chaque exemple

### Phase 3 : Migration Développeur (3-5 jours)
1. Convertir les exemples D01-D05 (advanced pipelines)
2. Convertir les exemples D06-D13 (generators, DL)
3. Convertir les exemples D14-D22 (transfer, advanced, internals)
4. Tests de chaque exemple

### Phase 4 : Reference & Documentation (2-3 jours)
1. Créer les exemples R01-R04
2. Intégrer dans RTD
3. Vérifier tous les liens et références

### Phase 5 : Validation Finale (1-2 jours)
1. Run complet via run.sh
2. Vérifier coverage des features
3. Revue documentation

---

## Annexe A : Template d'Exemple

```python
"""
{ID} - {Titre}
{'=' * (len(ID) + 3 + len(Titre))}

{Description courte en une phrase.}

Ce tutoriel couvre :

* Point 1
* Point 2
* Point 3

Prérequis
---------
- Exemple {précédent} (si applicable)
- Package X installé (si applicable)

Étapes suivantes
----------------
Après cet exemple, consultez :ref:`{suivant}`.

Durée estimée : X minutes
Difficulté : ★★☆☆☆
"""

# Standard library imports
import argparse

# Third-party imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import ShuffleSplit

# NIRS4All imports
import nirs4all

# Parse command-line arguments (standard pour tous les exemples)
parser = argparse.ArgumentParser(description='{Titre}')
parser.add_argument('--plots', action='store_true', help='Generate plots')
parser.add_argument('--show', action='store_true', help='Display plots interactively')
args = parser.parse_args()


# =============================================================================
# Section 1 : {Titre de Section}
# =============================================================================
print("\n" + "="*60)
print("Section 1: {Titre}")
print("="*60)

# Code avec commentaires explicatifs
# ...


# =============================================================================
# Section 2 : {Titre de Section}
# =============================================================================
# ...


# =============================================================================
# Résumé
# =============================================================================
print("\n" + "="*60)
print("Résumé")
print("="*60)
print("""
Ce que nous avons appris :
1. ...
2. ...
3. ...

Prochaine étape : {suivant}
""")

if args.show:
    import matplotlib.pyplot as plt
    plt.show()
```

---

## Annexe B : Exemples Supprimés et Justification

| Exemple | Raison de Suppression |
|---------|----------------------|
| Q25_complex_pipeline_pls.py | Redondant avec U16 et D05 |
| X1_metadata.py | Contenu intégré dans U05, U18, D18 |
| X2_sample_augmentation.py | Doublon de Q12/U11 |
| X4_features.py | Test interne, pas de valeur pédagogique |
| Q7_discretization.py | Feature mineur, intégré dans U03 |

---

## Annexe C : Checklist de Validation par Exemple

Pour chaque exemple migré :

- [ ] Utilise la nouvelle API (`nirs4all.run()`, etc.)
- [ ] Docstring complète avec format RST
- [ ] Arguments `--plots` et `--show` supportés
- [ ] Sections clairement délimitées
- [ ] Pas de DummyController triggered
- [ ] Résumé en fin d'exemple
- [ ] Durée d'exécution < 5 min (sauf DL)
- [ ] Liens vers exemples précédent/suivant
- [ ] Testé via run.sh individuellement
