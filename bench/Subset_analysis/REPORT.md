# Rapport - Selection d'un sous-ensemble representatif (TabPFN-paper, regression)

## Sources de donnees

- `bench/AOM_v0/publication/tables/tabpfn_comparison_per_dataset.csv` (matrice principale, 7298 lignes, regression).
- `bench/AOM_v0/Multi-kernel/publication/tables/tabpfn_comparison_per_dataset.csv` : doublon binaire du fichier ci-dessus, ignore.
- `bench/AOM_v0/publication/tables/master_pivot.csv` et la version Multi-kernel : baselines de l'article (61 lignes, valeurs manquantes), incluses dans `all_scores_long.csv` et la couverture, mais pas dans la matrice de selection.
- `bench/nicon_v2/publication/tables/full_comparison/long_per_dataset.csv` : 61 datasets, couverture incomplete, idem ci-dessus.

## Pourquoi 57 et non 59 / 61 ?

Le coeur d'analyse est l'intersection des datasets couverts par les modeles AOM avec couverture >= 57. Cette intersection compte **57 datasets**. Les datasets supplementaires presents dans `master_pivot.csv` ou `nicon_v2` (jusqu'a 61) ne sont pas couverts par tous les modeles -- les inclure introduirait des NaN ou forcerait l'imputation, ce qui biaiserait les agregats.

## Methode

1. Normalisation longue (`all_scores_long.csv`).
2. Construction de la matrice modele x dataset sur les modeles AOM couvrant les 57 datasets (128 modeles retenus).
3. Transformation `log(RMSEP)` puis z-score par dataset (suppression des differences d'echelle).
4. Selection avant gloutonne (sizes 3..30) avec score composite : 0.4*Spearman + 0.3*Kendall + 0.3*accord par paires (tolerance 1e-3) - 0.05*MAE agrege.
5. Baselines aleatoires (200 tirages par taille) et bootstrap par modeles (300 reechantillonnages) pour les IC.
6. Selection automatique : plus petit sous-ensemble respectant Spearman>=0.98, Kendall>=0.95, accord>=0.97, IC bootstrap p05 Spearman>=0.95 et p05 accord>=0.95. Sinon, plus petit element du plateau a 1% du meilleur composite.

L'analyse principale est volontairement limitee a la regression : le coeur 57 est defini par des scores RMSEP comparables. Les resultats classification disponibles dans certains sous-repertoires utilisent d'autres metriques et une couverture insuffisante, donc ils ne sont pas melanges a cette selection.

## Recommandation

- Taille selectionnee : **11** datasets.
- Spearman = 0.9952, Kendall = 0.9503, accord par paires = 0.9711.
- Bootstrap : Spearman p05 = 0.9907, accord p05 = 0.9637.
- Comparaison aleatoire (meme taille) : Spearman moyen 0.9348, accord moyen 0.8983.
- Datasets retenus :
  - `All_manure_P2O5_SPXY_strat_Manure_type`
  - `Fv_Fm_grp70_30`
  - `ta_groupSampleID_stratDateVar_balRows`
  - `WOOD_N_402_Olale`
  - `An_spxyG70_30_byCultivar_MicroNIR`
  - `All_manure_CaO_SPXY_strat_Manure_type`
  - `DIESEL_bp50_246_b-a`
  - `Malaria_Sporozoite_229_Maia`
  - `Chla+b_spxyG_block2deg`
  - `DIESEL_bp50_246_hla-b`
  - `Beef_Marbling_RandomSplit`

## Confiance et alternative conservatrice

- Justification de la selection : Tous les seuils sont satisfaits.
- Le sous-ensemble recommande de **11 datasets** satisfait les seuils **directs** (Spearman >= 0.98, Kendall >= 0.95, accord par paires >= 0.97) ainsi que les seuils bootstrap p05 sur Spearman et l'accord. En revanche, le critere **plus strict** exigeant aussi `bootstrap kendall_p05 >= 0.95` est plus exigeant : la borne basse (p05) du Kendall sur le bootstrap des modeles vaut 0.9359 pour la taille recommandee.
- **Alternative conservatrice** (plus petite taille satisfaisant aussi `kendall_p05 >= 0.95`) : **26 datasets** (Spearman 0.9970, Kendall 0.9624, accord 0.9788, bootstrap kendall_p05 0.9518).
- Les IC bootstrap sont calcules en reechantillonnant les modeles ; ils mesurent la robustesse au pool de modeles. Nous ne surestimons pas la confiance : la stabilite du Kendall (statistique discrete a faible effectif) est plus difficile a garantir.

## Limites

- L'analyse est restreinte aux modeles ayant >=57 datasets dans la table AOM ; modeles a couverture partielle exclus pour eviter l'imputation.
- Les datasets hors intersection (jusqu'a 61) ne sont pas evalues -- leur ajout depend de runs supplementaires.
- Le score composite pondere arbitrairement Spearman/Kendall/accord ; un autre arbitrage pourrait modifier marginalement le classement entre tailles voisines.

## Reproduction

```
python3 bench/Subset_analysis/analyze_subset.py
```
