# Rapport d'amélioration AOM-PLS face au papier TabPFN/NIRS

Date: 2026-04-28

## Sources lues

- `publication/manuscript/main.tex`
- `publication/tables/relative_rmsep_per_variant.csv`
- `publication/tables/tabpfn_comparison_per_variant.csv`
- `publication/tables/tabpfn_comparison_per_dataset.csv`
- `benchmark_runs/full/results.csv`
- `bench/tabpfn_paper/master_results.csv`
- `bench/tabpfn_paper/Robin_s_article-1.pdf`
- `source_materials/tabpfn/*.md`
- `Summary.md`, `docs/CODEX_REVIEWS.md`, `docs/BENCHMARK_PROTOCOL.md`

## Synthèse courte

Le résultat actuel ne dit pas que AOM-PLS bat TabPFN. Il dit quelque chose de plus utile pour améliorer l'algorithme:

- La production `nirs4all-AOM-PLS-default` est essentiellement à parité avec PLS sur la médiane: 29/57 victoires, RMSEP relatif médian 0.999.
- Les meilleurs gains AOM viennent de deux directions simples:
  - réduire la variance de sélection: `AOM-compact-cv3-numpy`, 32/57 victoires, médiane 0.997 vs PLS;
  - ajouter une correction de diffusion non linéaire avant AOM: `SNV-AOM-compact-numpy`, 32/57 victoires, médiane 0.984 vs PLS.
- TabPFN-opt reste devant globalement: les meilleurs AOM sont autour de 1.141 à 1.211 de ratio médian vs TabPFN-opt, avec seulement 9 à 12 victoires sur 57.
- AOM compact rivalise mieux avec TabPFN-Raw, CatBoost et CNN:
  - `AOM-compact-simpls-covariance`: 27/57 victoires vs TabPFN-Raw, ratio médian 1.011;
  - `AOM-compact-cv3`: 29/56 victoires vs CatBoost, ratio médian 0.993;
  - `AOM-compact-simpls-covariance`: 30/51 victoires vs CNN, ratio médian 0.975.

Conclusion: la priorité n'est pas d'ajouter encore plus d'opérateurs. La priorité est de stabiliser la sélection, d'intégrer proprement les corrections de diffusion, puis de régulariser POP et les modes multi-vues.

## Diagnostic scientifique

### 1. Le goulot principal est la variance de sélection

La production AOM choisit un couple opérateur/composantes avec un holdout interne de 20 %. Sur les petits jeux NIRS, ce holdout peut contenir très peu d'échantillons. Avec la banque par défaut, la sélection compare environ `100 x Kmax` candidats. Avec `Kmax=15`, cela fait 1500 comparaisons.

Ce mécanisme crée un "winner's curse": le meilleur RMSE holdout est souvent le plus chanceux, pas le meilleur sur test. Le fait que la banque compacte de 9 opérateurs égale ou dépasse la banque de 100 opérateurs confirme ce diagnostic.

Signal observé:

- `AOM-compact-simpls-covariance`: 30/57, médiane 0.999 vs PLS.
- `nirs4all-AOM-PLS-default`: 29/57, médiane 0.999 vs PLS.
- `AOM-compact-cv3`: 32/57, médiane 0.997 vs PLS.

### 2. La banque stricte-linéaire ne couvre pas les meilleurs prétraitements de diffusion

SNV et MSC ne satisfont pas l'identité stricte `(X A^T)^T Y = A X^T Y`, car leurs paramètres dépendent de chaque spectre. Pourtant SNV améliore le meilleur résultat AOM:

- `SNV-AOM-compact`: 32/57, médiane 0.984 vs PLS.

Cela indique que le déficit d'AOM face à TabPFN-opt vient en partie de transformations spectrales hors contrat strict-linéaire. TabPFN-opt bénéficie d'un protocole de prétraitement plus riche et d'un modèle non linéaire qui peut exploiter des représentations que PLS linéaire ne capture pas.

### 3. POP augmente trop l'espace de recherche

POP autorise un opérateur différent à chaque composante. En théorie c'est séduisant; en pratique, avec `Kmax=15`, c'est instable:

- `POP-nipals-adjoint`: 13/57 victoires, médiane 1.342 vs PLS.
- `POP-simpls-covariance`: 13/57 victoires, médiane 1.353 vs PLS.
- `nirs4all-POP-PLS-default`: 0/57 victoire, médiane 4.793 vs PLS.

Le problème n'est probablement pas l'idée POP, mais l'absence de régularisation du chemin d'opérateurs et d'arrêt précoce robuste.

### 4. OSC actuel est trop agressif

OSC avec `n_components=2` dégrade fortement:

- `OSC-AOM-default`: 14/57 victoires, médiane 1.347 vs PLS.
- `OSC-AOM-compact`: 13/57 victoires, médiane 1.350 vs PLS.

Le diagnostic le plus probable est une suppression excessive de variance utile à `y`, avec double usage supervisé de `y` avant le solveur PLS.

### 5. Les banques profondes ne sont pas exploitées par une sélection globale simple

Les essais `deep3/deep4` ajoutent des chaînes plus longues, mais ne déplacent pas la médiane en AOM global. En revanche, `ActiveSuperblock-deep3` donne un signal positif sur le sous-ensemble testé: 11/19 victoires, médiane 0.998 vs PLS. Cela suggère que les chaînes profondes peuvent être utiles si elles sont combinées ou pondérées, pas seulement mises dans une compétition holdout plus large.

## Plan d'amélioration priorisé

### Priorité 1: remplacer le holdout unique par une sélection stable

Objectif: diminuer la variance sans changer le modèle final.

Actions:

1. Faire de `criterion="cv"` ou `criterion="repeated_cv"` le défaut pour `n_train <= 200` ou `n_train <= 500`.
2. Ajouter une règle "one-standard-error": choisir l'opérateur le plus simple dans l'intervalle statistiquement équivalent au meilleur.
3. Scorer les opérateurs par médiane des rangs sur folds, pas seulement par moyenne RMSE.
4. Ajouter une pénalité de taille de banque ou de famille: à performance égale, préférer `identity`, puis les familles compactes.
5. Logger la stabilité de sélection: fréquence de `b*` et `k*` sur folds.

Expérience minimale:

- Comparer `holdout`, `cv3`, `cv5`, `repeated-cv3`, `one-SE-cv3` sur les 57 splits.
- Mesures: médiane vs PLS, wins vs PLS, wins vs TabPFN-Raw, temps, stabilité de `b*`.

### Priorité 2: faire de la banque compacte le défaut scientifique

Objectif: éviter que la banque de 100 opérateurs ajoute surtout du bruit de sélection.

Actions:

1. Utiliser la banque compacte comme défaut pour les petits jeux.
2. Construire une banque adaptative par famille: maximum un ou deux opérateurs par famille après screening covariance.
3. Dédupliquer les opérateurs par cosinus de réponse `A S`: si deux opérateurs donnent des réponses quasi identiques, garder le plus simple.
4. Garder la banque de 100 opérateurs comme bibliothèque de génération, pas comme espace de sélection direct.

Expérience minimale:

- `compact`, `default`, `family-pruned-default`, `response-dedup-default`.
- Cible: faire mieux que `AOM-compact-cv3` sans augmenter fortement le temps.

### Priorité 3: intégrer SNV/MSC comme prétraitements sélectionnés sans fuite

Objectif: capturer le signal de diffusion qui manque au contrat strict-linéaire.

Actions:

1. Traiter SNV, MSC, EMSC comme des étapes externes de pipeline, sélectionnées dans la CV interne.
2. Ajouter `SNV + compact`, `MSC + compact`, `EMSC + compact`, `SNV + derivative + compact`.
3. Implémenter une variante "local SNV" par fenêtres spectrales.
4. Interdire tout ajustement de prétraitement sur le test; refitter chaque prétraitement dans chaque fold interne.

Expérience minimale:

- Reproduire `SNV-AOM-compact` avec sélection fold-interne stricte.
- Ajouter `EMSC-AOM-compact` et `local-SNV-AOM-compact`.
- Comparer à TabPFN-Raw et TabPFN-opt.

### Priorité 4: régulariser POP

Objectif: sauver l'idée per-component sans explosion de variance.

Actions:

1. Limiter `Kmax` POP à 3, 5 ou 8 selon `n_train`.
2. Ajouter une pénalité de changement d'opérateur entre composantes.
3. Ajouter un seuil de gain minimal: ne changer d'opérateur que si le gain CV dépasse une marge.
4. Forcer la diversité utile: interdire deux opérateurs quasi colinéaires en réponse.
5. Arrêter POP dès que la composante ajoute moins qu'un seuil en CV.

Expérience minimale:

- `POP-K5`, `POP-K8`, `POP-path-penalty`, `POP-one-SE`.
- Objectif initial: revenir au moins à la parité PLS avant de chercher un gain.

### Priorité 5: développer le multi-vues comme alternative à la sélection dure

Objectif: ne plus choisir un seul opérateur quand plusieurs vues sont utiles.

Actions:

1. Finaliser `ActiveSuperblock` avec screening, diversité et normalisation Frobenius.
2. Tester `ActiveSuperblock-deep3` sur les 57 splits, pas seulement 19.
3. Ajouter un empilement OOF de petits experts: `identity`, `SNV`, `MSC`, `SG derivative`, `detrend`, chacun avec PLS/AOM compact.
4. Combiner les experts avec Ridge/PLS faible dimension et pénalisation.

Expérience minimale:

- `ActiveSuperblock-compact`, `ActiveSuperblock-default-pruned`, `ActiveSuperblock-deep3`, `OOF-MoE-compact`.
- Cible: battre `SNV-AOM-compact` vs PLS et réduire l'écart vs TabPFN-Raw.

### Priorité 6: utiliser TabPFN comme guide de représentation

Objectif: comprendre ce que TabPFN-opt capte que AOM-PLS ne capte pas.

Actions:

1. Extraire des features latentes spectrales compatibles TabPFN: PCA, wavelets, statistiques de bandes, dérivées, scores PLS/AOM.
2. Tester deux hybrides:
   - AOM comme extracteur de scores latents, puis TabPFN/Ridge sur ces scores;
   - AOM-PLS + modèle résiduel TabPFN sur features compressées.
3. Comparer les erreurs sur les datasets où TabPFN-opt écrase AOM: BEER, PEACH, COLZA, BISCUIT, CORN Starch.
4. Comparer les datasets où AOM bat TabPFN-opt: DIESEL, DarkResp, PHOSPHORUS, ECOSIS Chla+b.

Cette piste peut devenir un papier séparé: "spectral priors for TabPFN" plutôt qu'une amélioration pure de AOM-PLS.

## Feuille de route expérimentale

1. `selector_stability_v1`
   - holdout vs cv3 vs repeated-cv3 vs one-SE-cv3;
   - banques compact/default/pruned;
   - sortie attendue: choix du sélecteur par défaut.

2. `scatter_pipeline_v1`
   - SNV, MSC, EMSC, local-SNV;
   - fit strictement fold-interne;
   - sortie attendue: savoir si `SNV-AOM-compact` est robuste ou chanceux.

3. `pop_regularized_v1`
   - Kmax réduit, pénalité de changement, seuil de gain;
   - sortie attendue: POP doit revenir près de 1.0 vs PLS avant d'être publiable.

4. `active_superblock_full_v1`
   - exécuter deep3/compact/default-pruned sur les 57 splits;
   - sortie attendue: vérifier si le signal 11/19 se généralise.

5. `tabpfn_gap_analysis_v1`
   - lister les datasets où TabPFN-opt gagne largement et inspecter les erreurs;
   - tester AOM-scores vers TabPFN/Ridge et résiduel TabPFN;
   - sortie attendue: décider si AOM doit rester un estimateur final ou devenir aussi un extracteur de représentation.

## Critère de décision pour la prochaine version

Ne pas accepter une modification parce qu'elle améliore un ou deux datasets. L'accepter seulement si elle respecte au moins deux critères:

- améliore la médiane vs PLS;
- augmente les victoires vs PLS;
- améliore le ratio vs TabPFN-Raw;
- ne dégrade pas fortement les temps;
- réduit la variance de sélection mesurée par stabilité de `b*`/`k*`;
- conserve la parité avec la production quand la configuration retombe sur AOM default.

La cible réaliste de court terme n'est pas de battre TabPFN-opt. La cible réaliste est:

- battre PLS de façon stable;
- égaler ou battre TabPFN-Raw sur la médiane;
- rester interprétable et auditable;
- identifier précisément les régimes où TabPFN-opt doit être préféré.
