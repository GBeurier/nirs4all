# bench/Subset_analysis

Analyse paper-aware et class-balanced pour selectionner un sous-ensemble representatif des 57 datasets de regression communs.

## Sorties principales

- `tabpfn_paper_scores_long.csv` / `tabpfn_paper_scores_core_pivot.csv` : scores explicites du papier TabPFN.
- `model_class_mapping.csv` : classes incluses/exclues et couverture.
- `class_score_matrix.csv` : RMSEP par classe candidate x dataset, incluant les classes diagnostiques exclues.
- `main_class_score_matrix.csv` : RMSEP des 8 classes principales utilisees a poids egal.
- `class_dataset_zscores.csv` : log-RMSEP z-score par dataset sur les 8 classes principales.
- `class_balanced_subset_search_results.csv` : recherche gloutonne class-balanced + baselines aleatoires.
- `selected_subset.json` : recommandation principale class-balanced.
- `REPORT.md` : rapport de protocole.
- `SYNTHESE_TECHNIQUE.md` : synthese technique avec figures.
- `make_visualizations.py` : genere les figures dans `figures/`.

Les anciens artefacts variant-heavy sont regroupes dans `legacy_variant_heavy/` ; ils ne doivent plus etre utilises comme recommandation principale.

## Execution

```
python3 bench/Subset_analysis/analyze_subset.py
python3 bench/Subset_analysis/make_visualizations.py
```
