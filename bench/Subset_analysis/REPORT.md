# Rapport - protocole class-balanced et paper-aware

## Correction apportee

La premiere analyse utilisait 128 variantes AOM comme lignes de la matrice principale. Cela surrepresentait une seule famille. Le protocole courant remplace cette matrice par une matrice de **classes de modeles a poids egal** et rend les scores du papier TabPFN explicites.

Un plancher de **10 datasets** est impose : avec 8 classes seulement, les metriques de rang peuvent saturer artificiellement sur des subsets trop petits.

## Scores papier TabPFN

Les scores papier sont exportes dans `tabpfn_paper_scores_long.csv` et `tabpfn_paper_scores_core_pivot.csv`. Les classes visibles sont `Paper_CNN`, `Paper_CatBoost`, `Paper_PLS`, `Paper_Ridge`, `Paper_TabPFN_Raw`, `Paper_TabPFN_Opt`. CatBoost n'est donc plus anonymise.

## Classes dans l'analyse principale

- `AOM_PLS_Best` : 57/57 scores coeur, source `nicon_v2 long_per_dataset::AOM-PLS-best`.
- `AOM_PLS_Standard` : 57/57 scores coeur, source `nicon_v2 long_per_dataset::AOM-PLS PLS-standard`.
- `Paper_CNN` : 51/57 scores coeur, source `TabPFN paper master_pivot`.
- `Paper_CatBoost` : 56/57 scores coeur, source `TabPFN paper master_pivot`.
- `Paper_PLS` : 54/57 scores coeur, source `TabPFN paper master_pivot`.
- `Paper_Ridge` : 54/57 scores coeur, source `TabPFN paper master_pivot`.
- `Paper_TabPFN_Opt` : 57/57 scores coeur, source `TabPFN paper master_pivot`.
- `Paper_TabPFN_Raw` : 57/57 scores coeur, source `TabPFN paper master_pivot`.

Classes exclues de la selection principale faute de couverture suffisante ou parce qu'elles sont diagnostiques :
- `AOM_Ridge_Best` : 39/57, raison `excluded_from_main_low_core_coverage`.
- `Nicon_CNN_V1c` : 39/57, raison `excluded_from_main_low_core_coverage`.
- `Nicon_Internal_PLS` : 38/57, raison `excluded_from_main_low_core_coverage`.
- `Nicon_Internal_Ridge` : 39/57, raison `excluded_from_main_low_core_coverage`.
- `Nicon_Stack_Ridge_PLS_V1c` : 38/57, raison `excluded_from_main_low_core_coverage`.

Les deux voix AOM conservees sont intentionnelles : `AOM_PLS_Standard` represente la version standard, et `AOM_PLS_Best` represente la meilleure variante AOM-PLS agregee. Aucune variante AOM individuelle ne vote dans la selection.


## Methode

1. Coeur 57 reconstruit depuis `tabpfn_comparison_per_dataset.csv`, pas depuis les artefacts legacy.
2. `class_score_matrix.csv` conserve toutes les classes candidates pour audit ; `main_class_score_matrix.csv` garde les 8 classes a poids egal utilisees pour la selection.
3. Transformation `log(RMSEP)` puis z-score par dataset entre classes disponibles dans `class_dataset_zscores.csv`.
4. Recherche gloutonne sur les 8 classes principales, avec penalite de couverture et `agg_mae` dans l'objectif.
5. Baselines aleatoires, plancher de taille n>=10 et test leave-one-class-out pour eviter de surinterpreter les rangs, car 8 classes seulement font saturer Spearman/Kendall.

## Recommandation revisee

- Taille retenue : **10 datasets**.
- Critere principal : `agg_mae` 0.0713, couverture selectionnee 0.9875, min scores par classe 9.
- Diagnostics de rang, a interpreter avec prudence car seulement 8 classes : Spearman 1.0000, Kendall 1.0000, accord 1.0000.
- Datasets retenus :
  - `All_manure_MgO_SPXY_strat_Manure_type`
  - `An_spxyG70_30_byCultivar_NeoSpectra`
  - `TIC_spxy70`
  - `Chla+b_spxyG_species`
  - `ALPINE_P_291_KS`
  - `Beer_OriginalExtract_60_YbaseSplit`
  - `All_manure_Total_N_SPXY_strat_Manure_type`
  - `Chla+b_spxyG_block2deg`
  - `N_woOutlier`
  - `grapevine_chloride_556_KS`

## Alternative conservatrice

Plus petite taille avec `agg_mae <= 0.05` et couverture >= 0.97 : **19 datasets**.

## Limites

- Les rangs sur 8 classes saturent vite ; le critere discriminant devient surtout `agg_mae` et la couverture.
- Les scores papier CNN/PLS/Ridge/CatBoost ont quelques NaN sur les 57 datasets ; ils sont traites explicitement, pas imputes.
- Les sorties legacy sont regroupees dans `legacy_variant_heavy/` pour audit, mais ne portent plus la recommandation principale.

## Reproduction

```
python3 bench/Subset_analysis/analyze_subset.py
python3 bench/Subset_analysis/make_visualizations.py
```
