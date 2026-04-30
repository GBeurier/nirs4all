# bench/Subset_analysis

Analyse reproductible pour selectionner un sous-ensemble representatif des 57 datasets de regression communs (papier TabPFN).

L'analyse principale est regression-only : le coeur 57 repose sur des RMSEP comparables. Les resultats classification disponibles ailleurs dans `bench` ont des metriques et une couverture differentes, donc ils ne sont pas melanges a cette selection.

## Fichiers generes

- `all_scores_long.csv` : table longue normalisee (toutes sources).
- `model_coverage.csv` : couverture par modele dans la table AOM.
- `model_dataset_matrix.csv` : matrice RMSEP modele x dataset (coeur 57).
- `model_dataset_zscores.csv` : z-scores par dataset apres log.
- `dataset_coverage.csv` : par dataset, nombre de modeles toutes sources, nombre de sources, appartenance au coeur 57, et `dataset_group`.
- `greedy_progression.csv` : progression de la selection avant gloutonne.
- `random_baselines.csv` : statistiques de sous-ensembles aleatoires.
- `bootstrap_ci.csv` : IC bootstrap (sur les modeles) pour chaque taille.
- `subset_search_results.csv` : table jointe lisible (greedy, bootstrap CI, baselines aleatoires) pour chaque taille >=3.
- `selected_subset.json` : sous-ensemble retenu, metriques, et `conservative_alternative` (plus petite taille satisfaisant aussi `bootstrap kendall_p05 >= 0.95`, ou null).
- `SYNTHESE_TECHNIQUE.md` : synthese explicative des techniques statistiques avec figures illustratives.
- `make_visualizations.py` : genere les figures dans `figures/` a partir des sorties CSV/JSON.
- `REPORT.md` : rapport detaille en francais.

## Execution

```
python3 bench/Subset_analysis/analyze_subset.py
```

Figures et synthese :

```
python3 bench/Subset_analysis/make_visualizations.py
```

Seed deterministe : 1234.

## Note sur la confiance

Le sous-ensemble recommande (11 datasets) satisfait les seuils **directs** (Spearman/Kendall/accord par paires) ainsi que les bornes bootstrap p05 sur Spearman et accord. Le critere **plus strict** ajoutant `bootstrap kendall_p05 >= 0.95` selectionne plutot l'**alternative conservatrice** decrite dans `selected_subset.json` (champ `conservative_alternative`). Nous ne surestimons pas la confiance : sur cette dimension Kendall, la borne basse au p05 reste sous le seuil pour la taille recommandee.
