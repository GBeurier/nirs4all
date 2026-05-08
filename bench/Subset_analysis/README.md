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
- `subset_transfer_analysis.py` : teste le transfert selection-sur-subset -> resultats globaux depuis `benchmark_master_results.csv`.
- `SUBSET_TRANSFER_REPORT.md` / `subset_transfer_summary.csv` : diagnostic direct sur les subsets courants, baselines aleatoires, representativite et garde-fous.
- `rethink_subset_selection.py` : selection transfer-first depuis `benchmark_master_results.csv`, en preservant le choix de candidats concrets sur les scopes lineaires, non-lineaires, no-TabPFN, AOM-Ridge/AOM-PLS et multi-kernel.
- `RETHOUGHT_SUBSETS.md` / `rethought_subsets.json` / `rethought_subset_transfer_summary.csv` : nouvelles recommandations `fast12_transfer_core` et `audit20_transfer_core`.

Les anciens artefacts variant-heavy sont regroupes dans `legacy_variant_heavy/` ; ils ne doivent plus etre utilises comme recommandation principale.

## Execution

```
python3 bench/Subset_analysis/analyze_subset.py
python3 bench/Subset_analysis/make_visualizations.py
python3 bench/Subset_analysis/subset_transfer_analysis.py
python3 bench/Subset_analysis/rethink_subset_selection.py
```
