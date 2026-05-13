# Manifeste de protocole de benchmark NIRS

*Un cadre de type **Arena** pour la spectroscopie proche infrarouge : protocole fixe, soumission ouverte, ranking dynamique sur des datasets réels*

| Champ | Valeur |
|---|---|
| Version | v0.1 — DRAFT |
| Date | 2026-05-13 |
| Statut | Document de recherche et de protocole. **Co-rédaction Claude + Codex** — Codex a apporté les extensions sur le plan factoriel (§4.4), les vues d'incertitude (§6.14), la contribution scientifique (§8.4), les risques R11-R15, et les sections de gouvernance, citation et anti-leakage (§§11-13). |
| Périmètre | **Protocole** : ce qu'un contributeur doit faire pour qu'un modèle entre dans le benchmark, et ce que l'infrastructure en fait. **Recherche** : quelles questions ce protocole rend tractables, quelles agrégations et visualisations émergent. |
| Hors-périmètre | implémentation détaillée du runtime, intégration avec le code historique de `bench/`, choix d'UI/UX du site public. |
| Audience | Chercheurs NIRS, contributeurs de modèles/datasets/préprocessings/augmenters, futurs reviewers du protocole. |

---

## 0. Avant-propos

Ce document n'est pas un audit de l'existant. C'est la formalisation d'un protocole **à mettre en place de A à Z**. Les résultats historiques peuvent y entrer comme citations externes, mais le benchmark commence avec un nouveau corpus de runs, exécuté sous le protocole décrit ici. Le but final est une **page publique** présentant des rankings, clusterings et analyses d'effets sur tous les axes d'exploration (modèles, datasets, préprocessings, augmentations, splits, filtrage des outliers, et autres axes à découvrir). Ce que LMArena est aux LLMs, ce protocole l'est à la modélisation NIRS — *avec* la différence cruciale qu'on évalue contre vérité-terrain chimique, pas contre des préférences humaines.

La **contribution scientifique** visée n'est *pas* le leaderboard dynamique lui-même. C'est la création d'un instrument expérimental partagé qui transforme des choix habituellement confondus — modèle, split, préprocessing, augmentation, outliers, taille d'échantillon, instrument — en interventions observables sous conditions contrôlées. C'est la base de contre-factuels ainsi produite, plus que les classements qu'elle alimente, qui distingue cette arène d'un énième tableau public.

---

## 1. Vision

### 1.1 Le constat

La NIRS est un domaine où l'on compare régulièrement modèles, préprocessings et augmentations, mais ces comparaisons restent **non-comparables** : protocoles d'évaluation différents, splits ad hoc, seeds non documentées, préprocessings sous-spécifiés. Conséquence : aucun consensus quantitatif sur *quel modèle marche le mieux dans quel régime*, sur *quel préprocessing apporte vraiment de l'information*, sur *quelle augmentation transfère hors-distribution*. Les revues décennales (e.g. sur la dérivée Savitzky-Golay) tournent à l'argumentation d'autorité plutôt qu'à la mesure.

Le champ a accumulé des **outils** (`nirs4all`, alternatives propriétaires) capables d'exécuter ces comparaisons. Ce qui manque, c'est le **cadre** : un protocole assez précis et assez exécutable pour que deux contributeurs indépendants produisent des résultats commensurables, et assez exhaustif pour alimenter des analyses informatiques (rankings, clusterings, effets d'interaction) sur tous les axes d'exploration simultanément.

### 1.2 Le pari

Construire une **arène ouverte** : un protocole fixe, exécutable par toute personne soumettant un modèle, qui produit pour chaque soumission une grille systématique de résultats sur un pool de datasets réels. Toutes les soumissions entrent dans une base centrale et alimentent une page publique de rankings et visualisations.

Trois principes structurent l'arène :

1. **Conditions identiques pour tous.** Un nouveau modèle est évalué sous la même grille canonique que tous les modèles précédents — mêmes datasets, mêmes splits, mêmes seeds, mêmes préprocessings explorés, mêmes augmentations, mêmes filtres d'outliers.
2. **Soumission hybride.** Le contributeur soumet un *pipeline* (description structurée nirs4all) + un *bundle* (`.n4a` reproductible) de son modèle entraîné sur un sanity-check. Le runtime central re-exécute le pipeline sur la grille canonique entière. Cela garantit (a) la reproductibilité forte, (b) l'absence de cherry-picking par le contributeur, (c) la légèreté côté contributeur.
3. **Pool de datasets ouvert, sous-ensemble curé.** Tous les datasets soumis avec une licence valide entrent dans le pool. Un sous-ensemble *selected* est curé centralement pour assurer la diversité et la stabilité des leaderboards de référence. Les vues publiques offrent les deux modes : **"Selected datasets"** (curé, ranking de référence) et **"All datasets"** (pool complet, ranking exploratoire).

### 1.3 Différences vs LMArena

| Dimension | LMArena | NIRS-Arena (ce protocole) |
|---|---|---|
| Signal d'évaluation | Préférence humaine pairwise | Vérité-terrain chimique (RMSE, R², F1, …) |
| Test set | Prompts crowdsourcés | Pool de datasets réels, extensible |
| Ranking | Elo (relatif, dynamique) | Métriques agrégées + rangs + clusterings + interactions |
| Soumission | API endpoint ou poids | Pipeline nirs4all + `.n4a` bundle |
| Re-exécution | Sur prompts en flux | Sur grille canonique fixe |
| Évolution | Nouveaux prompts | Nouveaux datasets, nouvelle version de grille → re-run |
| Anti-tricherie | Modération + IP | Re-exécution centrale + EnvCard + audit pipeline |
| Variation systématique des conditions | Aucune | Au cœur du protocole (PP × aug × split × outliers × seed × n) |

L'analogie clé : **un cadre où la valeur d'une nouvelle méthode est mesurée automatiquement contre toutes les autres dans des conditions identiques**, sans que l'auteur de la méthode contrôle l'évaluation. La nouveauté NIRS-spécifique : *toutes les autres conditions* — pas juste les datasets — sont elles aussi systématiquement variées.

---

## 2. Questions de recherche structurantes

Cette section sert de **cahier des charges scientifique** : toute décision de protocole en aval doit servir au moins une de ces questions. Codex a aidé à hiérarchiser leur tractabilité (cf §8.4).

### 2.1 Q1 — Performance absolue d'un modèle *(atteignable dès v0.1)*

*Étant donné un dataset et une tâche, quel est le score atteignable et avec quelle dispersion ?* — Multiples seeds, multiples splits ; reporting médiane + IC + dispersion.

### 2.2 Q2 — Transférabilité inter-datasets *(aspirationnel — exige assez de datasets divers)*

*Une méthode qui marche sur dataset A marche-t-elle sur un dataset B "voisin" ? sur un dataset "éloigné" ?* — Profil scalaire par dataset (taille, SNR estimé, autocorrélation spectrale, complexité de y, instrument), clustering, reporting par cluster.

### 2.3 Q3-Q4 — Effets et interactions du préprocessing *(atteignable, mais comme effets conditionnels)*

*Une chaîne PP apporte-t-elle de l'information ? L'ordre des opérateurs modifie-t-il la performance ? Certaines paires (MSC + SNV) sont-elles redondantes ?* — Grille croisée PP × modèle ; canonicalisation des chaînes ; reporting `Δscore` par paire. Les effets mesurés sont *conditionnels* aux autres axes du bloc factoriel (cf §4.4) — pas des effets causaux universels.

### 2.4 Q5 — Utilité réelle des augmentations *(atteignable comme effet conditionnel)*

*Une augmentation améliore-t-elle (a) la performance in-domain (régularisation), (b) la performance cross-instrument (robustesse physique) ?* — Grille avec/sans × intensité ; tests sur datasets multi-capteur pour séparer (a) et (b).

### 2.5 Q6 — Sensibilité au split *(atteignable dès v0.1)*

*Quelle est la part de la variance de score attribuable au choix du split vs à la seed vs au modèle ?* — Décomposition de variance par modèle mixte ou ANOVA hiérarchique (cf §6.14).

### 2.6 Q7 — Filtrage d'outliers : bénéfice ou triche ? *(atteignable comme effet conditionnel)*

*Un modèle entraîné sur données filtrées gagne-t-il en performance, et si oui, se généralise-t-elle au test ?* — Grille avec/sans × type × fraction ; comparaison train-filtré/test-non-filtré vs train-non-filtré/test-non-filtré.

### 2.7 Q8 — Complémentarité pour stacking *(aspirationnel — exige prédictions complètes comparables sur tous les datasets)*

*Quels modèles sont complémentaires au sens où leurs résidus sont décorrélés ?* — Matrice de corrélation des résidus normalisés ; clustering des modèles par profil de résidus.

### 2.8 Q9 — Coût-performance *(atteignable dès v0.1)*

*Le gain marginal d'une méthode plus chère est-il proportionné à son coût ?* — Reporting joint `(score, fit_time, peak_memory)` ; courbes Pareto.

### 2.9 Q10 — Robustesse au sous-échantillonnage *(atteignable dès v0.1)*

*Comment la performance évolue-t-elle quand on réduit `n_samples` ?* — Sous-échantillonnage stratifié ou KS à plusieurs tailles ; courbes scaling laws.

### 2.10 Q11 — Domaine d'applicabilité *(nouvelle, ajoutée par Codex)*

*Dans quel régime spectral, chimique et instrumental un modèle cesse-t-il d'être fiable ?* — Cette question est plus centrale pour la NIRS appliquée que le rang global. Implique de croiser les profils scalaires de dataset (§4.2) avec les statuts `failed_*` (cf matrice de fragilité, §6.13) et avec la distribution des résidus.

---

## 3. Architecture du benchmark

### 3.1 Trois pôles

```
┌─────────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
│   Contributeur      │      │   Runtime central     │      │   Base + Web        │
│  (modèle, dataset,  │ ───▶ │   (re-exécution sur   │ ───▶ │  (rankings,         │
│   PP, aug, etc.)    │      │    grille canonique)  │      │   clusterings,      │
│                     │      │                       │      │   dataviz)          │
└─────────────────────┘      └──────────────────────┘      └─────────────────────┘
        │                              │                              │
        │ Soumet pipeline.json         │ Produit results.parquet      │ Affiche
        │     + bundle.n4a             │     + bundles.n4a            │     selected/all
        │     + sanity_check.csv       │     + manifest gelé          │     leaderboards
```

### 3.2 Pôle contributeur

Quatre types de contributions ; les modèles supportent **trois formats de soumission** (cf [arena/06_submission_formats.md](arena/06_submission_formats.md)) :

| Type | Contenu | Qui le fait | Validation |
|---|---|---|---|
| **Modèle — Format A** (référence) | Pipeline nirs4all (JSON) + bundle `.n4a` sanity | Auteur du modèle | Re-exécution centrale + audit pipeline (fit-on-train-only) |
| **Modèle — Format B** | Paquet Python `pip install`-able + classe sklearn-compatible (`fit/predict`) + manifest | Auteur du modèle | `pip install` sandboxé + vérification API sklearn + sanity check |
| **Modèle — Format C** | Paquet R (CRAN/Bioconductor/GitHub) + fonction `fit/predict` + manifest | Auteur du modèle | Container Docker R + sanity check ; controller dédié `RModelController` |
| **Dataset** | Spectres + cibles + métadonnées + licence + CITATION.cff | Détenteur du dataset | Validation de schéma + revue éditoriale (selected/community) + check anti-leakage (§13) |
| **Préprocessing** | Implémentation nirs4all + manifest | Auteur du PP | Tests d'idempotence, intégration au registry, CITATION.cff |
| **Augmenter** | Implémentation nirs4all + intensité physique | Auteur de l'augmenter | Tests sur dataset jouet, CITATION.cff |

Toute contribution incrémentale **n'invalide pas** les soumissions précédentes : seule une bump de version de la grille canonique le fait (§7).

### 3.3 Pôle runtime central

Reçoit une soumission, valide, ordonnance les runs sur la grille canonique, persiste résultats, publie. Détails techniques (orchestration, parallélisation) : hors-périmètre de ce document.

Le runtime central est l'autorité unique sur :
- l'exécution effective de chaque run (gage d'identité de conditions),
- la mesure des métriques (mêmes implémentations partout, hash de fonction versionné),
- la persistance des artefacts (bundles re-exécutés, prédictions, résidus),
- la mise à jour des rankings publics.

### 3.4 Pôle base + web

Base structurée (SQLite ou DuckDB en interne + Parquet pour les arrays) indexant tous les résultats atomiques. Web app servant rankings, matrices, clusterings, scatter plots, vues d'incertitude (cf §6.14). **Aucune vue n'est précalculée** — toutes sont dérivées par requête, garantissant qu'une mise à jour de grille ou un ajout de soumission se propage immédiatement.

---

## 4. Grille canonique de variations

### 4.1 Structure d'un point expérimental

Un *run atomique* est défini par un tuple :

```
(modèle, dataset, split_scheme, seed, preprocessing_chain, augmentation, outlier_filter, target_processing)
```

Le résultat atomique contient : métriques de score (régression : RMSE, RMSEP, MAE, R², bias ; classification : accuracy, balanced_accuracy, macro_F1, kappa), prédictions complètes, résidus, temps de fit, temps de prédiction, pic mémoire, fingerprints (modèle, dataset, env), statut (`ok` ou code d'erreur typé).

### 4.2 Axes de la grille

#### Axe **modèle**

Tout modèle soumis. Le contributeur fournit un pipeline nirs4all qui définit `{"model": <instance>}`. Tous les hyperparamètres sont fixés ou recherchés via les générateurs (`_grid_`, `_log_range_`, `_sample_`) qu'il déclare. Le runtime n'optimise rien que le contributeur n'ait demandé.

#### Axe **dataset**

Tout dataset du pool (curé `selected` + extensions `community`). Sous-échantillonnage `n_samples ∈ {50, 100, 200, 500, 1000, all}` produit autant de **sub-datasets** indexés séparément ; cela alimente Q10 (scaling laws). Le clustering des datasets par profil scalaire (cf §6) génère une *carte de transférabilité*.

#### Axe **split_scheme**

Schémas systématiquement testés en v0 :
- `KS_70_30` (Kennard-Stone)
- `SPXY_70_30`
- `Random_70_30`
- `KS_5fold`
- `SPXY_5fold`
- `Random_5fold`
- (régression) `KBinsStratified_5fold`
- (classification) `Stratified_5fold`

Le choix de seed contrôle l'instance exacte du split. Le `split_hash` est attaché à chaque run.

#### Axe **seed**

Liste fixe `seeds ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}` (au minimum 10 ; calibrage par analyse de puissance à confirmer après v0).

#### Axe **preprocessing_chain**

Chaîne canonique = séquence ordonnée d'opérateurs. La grille v0 énumère un **noyau réduit** (≈ 15-20 chaînes, pas 40) pour faciliter l'interprétation initiale ; l'extension à 40+ chaînes est planifiée pour v1.x une fois le signal scientifique stabilisé :

- **Aucun** : chaîne vide (`[]`)
- **Normalisation seule** : `[SNV]`, `[MSC]`
- **Dérivée seule** : `[SG(w=11, d=1, p=2)]`, `[SG(w=15, d=2, p=3)]`
- **Normalisation + dérivée** : `[SNV, SG(w=11, d=1, p=2)]`, `[MSC, SG(w=11, d=1, p=2)]`
- **Triplets canoniques** : `[SNV, SG(w=11, d=1, p=2), Detrend(2)]`, `[MSC, SG(w=11, d=1, p=2), Detrend(2)]`
- **Ordres alternatifs** (test de la convention chimiométrique) : `[Detrend(2), SNV, SG(w=11, d=1, p=2)]`
- **Sélection de bandes** : `[SNV, CARS(n=50)]`
- **Orthogonalisation** : `[SNV, OSC(n=2)]`
- **Final scaling** : appliqué à toutes : `MinMax`, `Standard`, `Robust`.

Toutes les chaînes sont **canonicalisées** avant exécution : si deux notations différentes produisent le même `pipeline_hash` canonique (e.g. `SG(d=0) + Deriv1` ≡ `SG(d=1)`), elles sont fusionnées. La table de canonicalisation est versionnée.

#### Axe **augmentation**

Pour le **noyau v0** réduit (≈ 5 augmenters × 3 intensités + sans-augmentation = 16 configurations, pas 25) :

- `{"aug": None}` (baseline)
- `{"aug": GaussianAdditiveNoise, "intensity": low|med|high}`
- `{"aug": WavelengthShift, "intensity": low|med|high}`
- `{"aug": ScatterSimulationMSC, "intensity": low|med|high}`
- `{"aug": MixupAugmenter, "intensity": low|med|high}` (position post-PP)
- `{"aug": BatchEffectAugmenter, "intensity": low|med|high}`

Les autres augmenters du registry (`MultiplicativeNoise`, `LinearBaselineDrift`, `InstrumentalBroadeningAugmenter`, etc.) entrent en v1.x après stabilisation du signal scientifique sur le noyau.

**Intensités physiques** : pour chaque augmenter, σ ancré sur l'estimation du bruit instrument du dataset cible — `low` ≈ 0.5×, `med` ≈ 1.0×, `high` ≈ 2.0×. Ancrage assure comparabilité inter-datasets.

**Position dans le pipeline** : augmenters "physiques raw" (bruit, drift, scatter, batch, shift, broadening) → **avant** tout préprocessing. Augmenters "représentationnels" (`Mixup`) → **après** le préprocessing, avant le modèle.

#### Axe **outlier_filter**

Filtres testés en v0 :
- `None` (baseline)
- `YOutlierFilter(method="iqr", k=1.5)`
- `XOutlierFilter(method="mahalanobis", contamination=0.05)`
- `{exclude: [YOutlier, XOutlier], mode: any}`
- `{tag: YOutlier}` (marquage sans exclusion)

Le filtrage s'applique sur le set d'entraînement uniquement (jamais sur le test). Le `tag` permet de mesurer l'effet *informationnel* du marquage sans suppression d'échantillons.

#### Axe **target_processing**

Pour la régression : `None`, `MinMax`, `Standard`, `RobustScaler`. Pour la classification : pas de transformation.

### 4.3 Cardinalité — pourquoi le produit cartésien est hors d'atteinte

Le produit cartésien complet (~1 million de runs/modèle/dataset) est intractable. La v0 utilise une **structure factorielle fractionnaire par blocs** (§4.4). Cardinalité v0 estimée : ≈ 12 000-18 000 runs par modèle sur 30 datasets `selected`. À 1-30 s par run pour un modèle léger : 3-150 h-CPU. Pour un modèle GPU lourd (TabPFN, NN profond) : 100-1000 h-GPU. Acceptable pour un runtime cluster ; pas pour un poste de chercheur isolé. La soumission hybride (§5) règle ce point : le contributeur ne tourne que le sanity-check ; le cluster fait la grille.

### 4.4 Plan factoriel fractionnaire bloqué *(extension Codex)*

La grille v0 doit être décrite comme un **plan factoriel fractionnaire bloqué**, et non seulement comme une optimisation de coût. Le principe est classique en plan d'expériences : remplacer le produit cartésien complet par un sous-ensemble dont la matrice de design conserve le rang nécessaire pour estimer les contrastes jugés prioritaires (Box, Hunter & Hunter ; Montgomery ; Wu & Hamada (unverified)).

Pour chaque bloc, le protocole **doit déclarer explicitement les effets estimables** :

- **Bloc principal** : effets modèle, dataset, split, seed, préprocessing canonique, augmentation minimale et filtre minimal, plus interactions de premier ordre `modèle × dataset` et `modèle × PP_canonique`.
- **Bloc PP exploratoire** : effet de la chaîne PP *à autres axes fixés* (split = KS_70_30, seed ∈ {0,1,2}, aug = None, filter = None, target = MinMax).
- **Bloc augmentation** : effet conditionnel sous PP canonique, split KS, target MinMax, filtre absent.
- **Bloc outliers** : effet conditionnel sous PP canonique, split KS, aug = None.
- **Bloc scaling laws** : effet de `n_samples` à autres axes fixés.
- **Bloc cross-instrument** : effet du capteur, datasets multi-capteur uniquement.

Cette précision évite une ambiguïté importante : un effet mesuré dans le bloc augmentation n'est pas « l'effet général de l'augmentation », mais **l'effet sous PP canonique, split KS, target MinMax et filtre absent**. La publication doit donc fournir, avec chaque release de grille :

1. La matrice de design des blocs.
2. La structure d'aliasing entre effets principaux et interactions d'ordre supérieur non estimables.
3. La liste explicite des contrastes estimables.
4. La liste explicite des interactions **non identifiables** sur la grille v0 (e.g. interactions de troisième ordre `modèle × PP × aug`).

#### Choix du design par blocs vs alternatives

| Approche | Force | Limite pour NIRS-Arena |
|---|---|---|
| Produit cartésien complet | Tout est identifiable | Intractable (10⁶ runs) |
| Plans **Latin hypercube** / **Sobol** (McKay et al. 1979 ; Sobol 1993 (unverified)) | Exploration efficace d'espaces continus | Moins lisibles pour facteurs catégoriels (PP, split, filter) |
| Plans **Plackett-Burman** (Plackett & Burman 1946 (unverified)) | Crible beaucoup de facteurs à coût minimal | Aliase fortement les interactions — *dangereux* ici car `modèle × PP` et `modèle × dataset` sont les interactions *scientifiquement intéressantes* |
| **Plan fractionnaire par blocs** *(retenu)* | Interprétabilité, contrôle explicite des effets estimables | Non-couverture des interactions ≥ 3 (assumée) |

Le choix par blocs est donc justifié par l'**interprétabilité**, pas seulement par le coût.

### 4.5 Conditions de validité d'un run

- **Fit-on-train-only.** Toute statistique de population (scaling, sélection de bandes, OSC) est entraînée uniquement sur le train. Audit automatique du pipeline pour rejeter les violations.
- **Seed control complet.** PRNG numpy / python / torch / jax / sklearn / `PYTHONHASHSEED` propagés et hashés. Variables `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` fixées et loguées.
- **EnvCard.** Versions packages, BLAS backend, CUDA/cuDNN, GPU deterministic flags loguées. Le `env_hash` *exclut* le hostname (qui est métadonnée séparée).
- **Time/memory.** Mesure `fit_time_s`, `predict_time_s`, `peak_rss_mb` (CPU) et `peak_gpu_mb` (si GPU). Pour les modèles `fast` (< 60 s), répétition × 3 (médiane + IQR). Pour les modèles `slow`, mesure unique avec timeout déclaré.
- **Failure typé.** Tout échec produit un code (`failed_nan`, `failed_oom`, `failed_timeout`, `failed_convergence`, `failed_dispatch`) avec message tronqué. La matrice de fragilité (§6.13) est publiée.

---

## 5. Protocole de soumission d'un modèle

Le protocole admet **trois formats** (A : bundle nirs4all ; B : paquet Python sklearn-compatible ; C : paquet R) — spécifications complètes dans [arena/06_submission_formats.md](arena/06_submission_formats.md). Cette section décrit le flow général ; les particularités par format sont déléguées au doc 06.

### 5.1 Étapes côté contributeur

1. **Implémenter** le modèle :
   - Format A : pipeline nirs4all (soit comme `sklearn`-compatible existant, soit comme nouveau `Controller` via `@register_controller`).
   - Format B : paquet Python `pip install`-able exposant une classe sklearn-compatible (`fit`, `predict`, optionnellement `predict_proba`).
   - Format C : paquet R (CRAN, Bioconductor, ou GitHub) avec une fonction `fit` et une fonction `predict`.
2. **Écrire un `manifest.json`** déclaratif (format A : annexe B ; formats B et C : schémas dans le doc 06).
3. **Exécuter un sanity-check** localement sur `sample_data/regression` avec seed 0. Format A produit un bundle `.n4a` ; formats B et C produisent un CSV de prédictions.
4. **Soumettre** `manifest.json` + artefacts (bundle ou pas selon format) + `sanity_check.csv` + `CITATION.cff` à l'endpoint d'arène.

### 5.2 Étapes côté runtime central

1. **Validation** : détection du format (A/B/C) ; validation du manifeste contre le JSON Schema correspondant ; intégrité ; conformité du sanity-check à un re-run local. Audit pipeline (fit-on-train-only — automatique en A, déclaration + détection rétrospective en B/C).
2. **Ordonnancement** : la grille canonique est traduite en `N_runs` tâches indépendantes ; allocation selon `runtime_tier`.
3. **Exécution** parallèle (joblib loky pour CPU ; queue GPU pour modèles GPU).
4. **Persistance** : chaque run atomique → ligne `results.parquet` + array Parquet de prédictions/résidus + bundle `.n4a` (signed avec EnvCard).
5. **Validation post-hoc** : checksum des bundles ; vérification reproductibilité sur 1-2 runs aléatoires (re-exécution comparée à l'original).
6. **Indexation** : mise à jour des vues dérivées. Notification au contributeur (et aux auteurs de datasets/PP/augmenters utilisés — rétro-citation, §12). Soumission visible publiquement.

### 5.3 Cycle de vie d'une soumission

```
soumise → validée → en cours d'exécution (% progress) → terminée → indexée → publiée
            │            │                                 │
            ▼            ▼                                 ▼
         rejetée     timed_out                         retired (si bump grille)
```

### 5.4 Garanties offertes au contributeur

- **Reproductibilité forte** : la soumission est rejouable à l'identique tant que l'EnvCard est honoré ; tout échec de reproduction est imputable au système (déclaration explicite des limites pour modèles stochastiques).
- **Égalité de traitement** : pas de tuning post-soumission par le runtime ; le pipeline est exécuté tel quel.
- **Visibilité** : tous les résultats atomiques sont consultables, pas seulement les agrégés.
- **Audit-trail** : chaque modification → nouvelle soumission ; aucune mutation rétroactive.
- **Crédit** : citation structurée (§12) automatique dans toute vue dérivée mentionnant le modèle.

---

## 6. Agrégations et visualisations

Les vues publiques sont *dérivées* par requête sur la base. Toute mise à jour de grille ou ajout de soumission se propage immédiatement.

### 6.1 Vue **Leaderboard global**
Pour chaque tâche, classement des soumissions par métrique principale agrégée (médiane sur seeds, IC bootstrap 95 %). Filtres : *Selected* vs *All datasets*, *Régime n_samples*, *Cluster de datasets*, *Avec/sans augmentation*.

### 6.2 Vue **Matrix modèle × dataset**
Heatmap `score_ratio_vs_pls_canon` par cellule. Cellules manquantes (modèle non éligible) marquées explicitement.

### 6.3 Vue **Ranking préprocessings**
Pour un modèle sélectionné, classement des chaînes PP par `Δscore_ratio` médian. Affiche les paires gagnantes par cluster de dataset.

### 6.4 Vue **Matrix PP × modèle**
Pour chaque préprocessing × chaque modèle, score médian. Révèle les **interactions** : un modèle peut être insensible au PP, un autre extrêmement sensible.

### 6.5 Vue **Augmentation effect**
Pour chaque augmenter × intensité, `Δscore_ratio` agrégé. Distinction in-domain vs cross-instrument.

### 6.6 Vue **Sensibilité au split**
Box-plot ou violin : pour chaque modèle, distribution du score sur les 8 schémas de split (à seed et PP fixés).

### 6.7 Vue **Effet du filtrage d'outliers**
Pour chaque modèle, `Δscore_ratio(filter)` par dataset. Tagged-vs-removed comparé.

### 6.8 Vue **Diversité des modèles** (clustering)
Matrix de corrélation des résidus normalisés sur tous les datasets. Projection 2D (UMAP ou PCA) des modèles. Cluster détectés → groupes de modèles redondants.

### 6.9 Vue **Pareto coût-performance**
Scatter (`fit_time × score`) avec frontière Pareto. Filtres par dataset, par régime de taille.

### 6.10 Vue **Scaling laws**
Courbes `score = f(n_samples)` par modèle, sur les 6 datasets soumis au sous-échantillonnage.

### 6.11 Vue **Clusters de datasets**
Projection 2D des datasets dans l'espace des profils scalaires. Cluster colorés. Chaque cluster a son leaderboard. *Zero-shot routing : déplacé en feuille de route v1.x — pas en v0, trop spéculatif sans pool suffisant.*

### 6.12 Vue **Domaine d'applicabilité** *(nouvelle, Q11)*
Pour chaque modèle, projection des datasets où il échoue (`failed_*` ou `score_ratio > 1.0`) vs ceux où il réussit. Révèle les zones de l'espace de profils scalaires où le modèle est inopérant. Croisement avec la matrice de fragilité (6.13).

### 6.13 Vue **Fragilité**
Heatmap : pour chaque modèle × dataset, fraction `failed_*`. Distinguer les modèles qui plantent (instabilité) de ceux qui réussissent partout mais médiocrement.

### 6.14 Vues d'incertitude et diagnostics chimiométriques *(extension Codex)*

Trois vues complémentaires manquaient au draft initial pour éviter que l'arène ne devienne un simple tableau de scores.

**6.14.1 — Décomposition de variance.** Pour chaque cellule modèle × dataset, afficher non seulement la médiane, mais la part de variance attribuable au split, à la seed, au dataset, au préprocessing, et au résidu inexpliqué. Estimation par **bootstrap hiérarchique** ou **modèle mixte** (`score ~ 1 + (1|model) + (1|dataset) + (1|split) + (1|seed) + (1|PP)` puis ratios de variance des effets aléatoires). Permet d'éviter des conclusions abusives sur des écarts de score faibles : si la variance modèle vaut 5 % du total et la variance dataset 60 %, comparer deux modèles sur médiane globale est trompeur.

**6.14.2 — Comparaison interactive de type HELM** (Liang et al. 2023 (unverified)). Sélectionner deux modèles ; obtenir sur une même page : les deltas appariés par dataset, les IC, les coûts, la fragilité, la robustesse cross-instrument, la couverture. Cette vue privilégie les **comparaisons appariées** plutôt que les rangs globaux : beaucoup de conclusions chimiométriques sont locales (un modèle gagne sur 7 datasets sur 30, lesquels ?).

**6.14.3 — Diagnostics explicitement NIRS.** Comparaison des loadings PLS, des coefficients ou des importances par longueur d'onde ; Q-Q plots des résidus ; résidus vs y prédit ; leverage / Q-residuals quand disponibles. Ces vues **n'entrent pas dans le ranking** : elles servent à distinguer une amélioration statistique crédible d'un artefact de préprocessing ou d'un modèle qui gagne en exploitant une zone spectrale physiquement non plausible.

### 6.15 Vue **Soumissions over time**
Graphe temporel d'adoption : nouvelles soumissions, mises à jour de la grille, bumps majeurs. Indicateur de santé communautaire.

---

## 7. Versioning du protocole

### 7.1 Trois niveaux de version

| Niveau | Effet | Re-run nécessaire ? |
|---|---|---|
| **Patch** | Correction de bug runtime, ajout de métrique secondaire | Non |
| **Minor** | Ajout d'un dataset, d'un PP, d'un augmenter au registry sans modifier la grille | Optionnel pour les modèles déjà soumis |
| **Major** | Modification de la grille canonique (axes, blocs, métriques principales) | **Oui** pour tous les modèles `selected` |

### 7.2 Cycle de release

- **v0.x** — Pré-publique : grille en stabilisation, contributeurs invités, pas de leaderboard public.
- **v1.0** — Publique : grille gelée, leaderboard ouvert, DOI Zenodo.
- **v1.x** — Évolutions mineures : ajouts validés.
- **v2.0** — Bump majeur : nouvelle grille, re-run des `selected`.

### 7.3 Politique de re-run sur bump majeur

1. Annonce 30 jours avant.
2. Re-exécution automatique de tous les `selected` (statut `pre-bump-soumise`).
3. Anciens résultats tagués `v(N).0-frozen` ; restent accessibles, ne participent plus au ranking par défaut.
4. Rétro-notification aux contributeurs concernés (§12).

### 7.4 Politique de dépréciation des datasets

Un dataset est retiré du pool **selected** si : licence invalide, contamination/leakage démontré, ou nouvelle version (vM+1) ajoutée par décision éditoriale. **Aucun dataset n'est supprimé physiquement** — taggé `retired` avec raison.

---

## 8. Positionnement vis-à-vis des benchmarks ML existants

### 8.1 Analogies pertinentes

| Benchmark | Inspiration apportée | Limite vis-à-vis NIRS |
|---|---|---|
| **LMArena / ChatbotArena** | Cadre d'arène ouverte, soumission, ranking dynamique | Évaluation crowdsource (NIRS = vérité-terrain), pas de variation systématique des conditions |
| **OpenLLM Leaderboard (HuggingFace)** | Grille fixe, soumission, reproductibilité | Tâches NLP standardisées, pas de variation PP |
| **HELM** (Stanford 2023 (unverified)) | Évaluation multi-axes : précision, robustesse, biais, coût, calibration | LLM-centric ; concept des "scenarios × metrics × models" directement transposable |
| **OpenML** | Plateforme de partage de datasets et résultats ML | Plus généraliste, peu structuré sur la NIRS ; manque la grille canonique |
| **MoleculeNet** (Wu et al. 2018, *Chem. Sci.* — unverified) | Benchmark méthodologique en chimie computationnelle | Pas de variation PP systématique |
| **Open Graph Benchmark (OGB)** | Splits réalistes (scaffold), datasets curés, leaderboard ouvert | Graphes — mais design split-aware transposable |
| **DynaBench** | Benchmark "vivant", évolution avec nouveaux datasets | NLP — modèle d'évolution v0→v1→v2 inspire le versioning §7 |

### 8.2 Spécificités NIRS valorisées

- **Vérité-terrain chimique** : évaluation automatique, pas de crowdsourcing.
- **Calibration transfer (cross-instrument)** : axe *natif* dans la grille — différenciateur scientifique direct.
- **Interprétabilité physique** : croisement coefficients / SHAP / VIP avec bandes spectroscopiques connues — diagnostic qualitatif (cf §6.14.3).
- **Variation systématique du préprocessing** : aucun benchmark ML public ne le fait. Aspect le plus inhabituel scientifiquement.
- **Synthèse contrôlée** : `nirs4all.generate()` permet d'ajouter des stress tests synthétiques.

### 8.3 Positionnement éditorial à terme

Une fois v1.0 + ≥ 50 soumissions + ≥ 30 datasets, l'arène est candidate à une publication méthodologique. Précédents pertinents : MoleculeNet (Chem. Sci.) (unverified), Open Graph Benchmark (NeurIPS Datasets & Benchmarks track), CellBench et BEELINE (*Nature Methods*) (unverified). La cible idéale dépend de la traction au moment de la soumission ; ce document ne fige pas ce choix.

### 8.4 Contribution scientifique au-delà du leaderboard *(extension Codex)*

La contribution scientifique n'est pas le classement dynamique lui-même, mais la création d'un **instrument expérimental partagé** pour la NIRS. Le protocole transforme des choix habituellement confondus — modèle, split, préprocessing, augmentation, outliers, taille d'échantillon, instrument — en *interventions observables sous conditions contrôlées*. C'est cette base de contre-factuels qui distingue l'arène d'un leaderboard ML généraliste.

Hiérarchisation des questions de §2 selon leur **tractabilité v0.1** :

- **Atteignables dès v0.1** : Q1 (performance absolue), Q6 (sensibilité au split), Q9 (coût-performance), Q10 (scaling laws). Tous sont fonctions de données déjà capturées par la grille de base.
- **Atteignables comme effets conditionnels** : Q3-Q4 fusionnées (effets et interactions du préprocessing), Q5 (utilité des augmentations), Q7 (filtrage outliers), Q11 (domaine d'applicabilité). Formulables uniquement *à autres axes fixés* — risque de mauvaise interprétation si publiées comme effets causaux universels.
- **Aspirationnelles** : Q2 (transférabilité inter-datasets) et Q8 (complémentarité pour stacking) exigent un pool de datasets suffisamment large et divers et des prédictions complètes comparables. Pas avant v1.x.

Cette hiérarchisation doit apparaître dans la première publication méthodologique : sur-vendre les résultats Q2/Q8 sur un pool insuffisant est un risque réputationnel.

---

## 9. Risques et limites

### 9.1 Risques scientifiques (R1..R15)

**R1 — Goodhart sur la métrique principale.** Si le ranking n'utilise qu'un score, les contributeurs optimisent pour cette métrique au détriment d'autres axes (coût, robustesse, interprétabilité). *Mitigation* : leaderboard multi-axes (score × coût × fragilité × robustesse) ; pas de prix unique.

**R2 — Domination par les datasets faciles.** Le pool `all` peut être dominé par des datasets sur lesquels tous les modèles convergent. *Mitigation* : la vue `selected` reste la vue de référence ; analyses par cluster (§6.11).

**R3 — Biais de couverture (modèles disparates).** Tous les modèles ne sont pas éligibles à tous les datasets. *Mitigation* : reporting de couverture ; ranking restreint aux paires effectivement observées.

**R4 — Confounding par échelle de y inter-datasets.** RMSE non comparable entre datasets. *Mitigation* : `score_ratio_vs_pls_canon` comme métrique principale (ratio invariant) ; RMSE reportée en parallèle.

**R5 — Leakage par préprocessing fit-global.** Si un PP est fit sur tout le dataset, test contaminé. *Mitigation* : audit automatique du pipeline ; refus.

**R6 — Sensibilité au choix de grille.** Ranking dépend de la grille canonique. *Mitigation* : versioning explicite ; anciens rankings restent accessibles.

**R7 — Augmentation comme régularisation déguisée.** Aider in-domain sans aider cross-instrument = effet régularisation seul. *Mitigation* : axe cross-instrument dans la grille v0.

**R8 — Stratification du seed insuffisante.** 10 seeds peuvent être insuffisants pour des NN profonds. *Mitigation* : calibration par puissance post-hoc pour `slow`/`very_slow`.

**R9 — Sélection éditoriale des `selected` contestable.** Choix curé est subjectif. *Mitigation* : gouvernance publique (§11) ; vue `all` toujours accessible.

**R10 — Non-déterminisme résiduel.** cuDNN/BLAS/joblib pas bit-déterministes. *Mitigation* : déclaration explicite des limites dans EnvCard ; vérification post-hoc.

**R11 — Aliasing des blocs factoriels.** *(Codex)* Le plan réduit peut confondre un effet principal avec une interaction non mesurée. Une augmentation « gagnante » sous SNV+SG peut échouer sous MSC ou raw. *Mitigation* : **publier la structure d'aliasing** (§4.4) ; éviter les formulations générales hors du bloc concerné.

**R12 — Faux positifs par multiplicité.** *(Codex)* La matrice modèle × dataset × PP × augmentation × outliers produit des milliers de comparaisons. *Mitigation* : corrections FDR (Benjamini-Hochberg) pour les tableaux exploratoires ; emphase sur tailles d'effet et intervalles plutôt que sur p-values brutes ; validation sur blocs indépendants.

**R13 — Biais du sous-ensemble `selected`.** *(Codex)* Même curé, `selected` peut favoriser certains instruments, matrices ou plages de y. *Mitigation* : critères publics, rotation contrôlée, analyses de **sensibilité leave-one-selected-out** (le ranking est-il stable si on retire le dataset le plus influent ?).

**R14 — Biais de sélection des contributeurs.** *(Codex)* Les modèles soumis seront ceux que leurs auteurs pensent compétitifs ; les méthodes faibles ou anciennes seront sous-représentées. *Mitigation* : inclure des **baselines historiques maintenues centralement** (PLS-canon, Ridge-canon, RF, CNN-naïf) que le runtime central re-soumet à chaque bump.

**R15 — Censure par échec ou timeout.** *(Codex)* Exclure les runs failed favorise les modèles fragiles mais parfois performants. *Mitigation* : intégrer la **fragilité comme métrique principale** (cf §6.13), pas seulement comme diagnostic ; publier séparément les scores « sur runs réussis ».

### 9.2 Risques opérationnels

**O1 — Coût compute en explosion combinatoire.** *Mitigation* : structure factorielle réduite (§4.4) ; runtime tiering.

**O2 — Tricherie par soumission falsifiée.** *Mitigation* : re-exécution **systématique** par le runtime, jamais confiance dans le bundle pour le scoring.

**O3 — Maintenance des licences datasets.** Licences révocables. *Mitigation* : miroir Zenodo ; politique de remplacement par dataset équivalent (même cluster).

**O4 — Soutenabilité communautaire.** *Mitigation* : appels à contribution réguliers ; co-auteurs sur la publication méthodologique pour les contributeurs `selected` ; partenariats laboratoires NIRS.

**O5 — Test set leakage public.** Un test set public attire le sur-ajustement. *Mitigation* : possibilité de **hold-out servers** sur un sous-ensemble (modèle Kaggle, soumission server-side) — pas en v0, à considérer en v1.x.

### 9.3 Limites assumées

- Le protocole **ne tranche pas** entre méthodes équivalentes en performance ; il *expose* l'équivalence.
- Le protocole **ne remplace pas** la validation experte sur un dataset spécifique ; il fournit un *prior* statistique.
- Le protocole **ne couvre pas** les pipelines extra-modèles (multi-source, multi-target avec corrélation explicite, deep transfer learning avec pretraining externe) sans extension future.
- Le protocole **ne donne pas** d'interprétabilité physique automatique ; les vues de diagnostic d'alignement physique sont qualitatives.

---

## 10. Feuille de route minimale (v0 → v1.0)

### 10.1 v0.1 — Bootstrap (T0..T+2 mois)

- Geler la grille canonique v0.1 (axes, blocs, métriques) **et publier la structure d'aliasing**.
- Construire le runtime central minimal : ordonnancement, exécution, persistance.
- Curé une liste initiale de 12-20 datasets `selected`.
- Implémenter des **familles de baselines** maintenues centralement (PLS-canon, Ridge-canon, RF-canon, CNN-naïf) — pas des modèles nommés du paysage de recherche actuel, pour éviter de figer le marketing.
- Page web minimale : leaderboard global + matrice modèle × dataset + décomposition de variance.

### 10.2 v0.2 — Pre-public (T+2..T+5 mois)

- Étendre grille (augmentations supplémentaires, outliers, scaling laws).
- Ajouter vues : ranking PP, augmentation effect, sensibilité split, diversité, comparaison HELM-like.
- Ouvrir aux contributeurs invités (5-10 premiers).
- Documentation publique : guide de soumission, schéma `pipeline.json`, exemples.

### 10.3 v1.0 — Publique (T+6..T+9 mois)

- Grille canonique gelée. Leaderboard ouvert. Pool `selected` consolidé (≥ 30 datasets). ≥ 30 modèles soumis. DOI Zenodo. Annonce communautaire.

### 10.4 v1.x → v2.0

Selon traction et retours. Possibles bumps majeurs : axe multi-source, datasets synthétiques contrôlés, hold-out server.

---

## 11. Gouvernance du sous-ensemble `selected` *(nouvelle section, Codex)*

`selected` détermine le ranking de référence et devient *de facto* le point politiquement sensible du protocole. La gouvernance n'est pas un détail d'implémentation ; c'est ce qui distingue un protocole scientifique d'un outil de marketing.

### 11.1 Comité éditorial

- **Composition** : 5-7 chercheurs NIRS de provenances disparates (académie / industrie, géographies, sous-champs : agroalimentaire, pharma, géosciences, biomédical).
- **Mandat** : 2 ans, renouvelable une fois. Rotation décalée pour éviter la rupture totale.
- **Conflits d'intérêts** : déclaration obligatoire ; abstention sur les votes concernant ses propres datasets ou modèles.

### 11.2 Critères d'entrée d'un dataset dans `selected`

Publics, opérationnels, vérifiables :

1. Licence open (Creative Commons, CeCILL, MIT) ou équivalent commercial-autorisé.
2. Métadonnées complètes : signal_type, instrument, plage spectrale, résolution, conditions de mesure, citation.
3. `n_samples ≥ 50` (seuil v0, à recalibrer).
4. Pas de chevauchement détecté avec un dataset déjà dans `selected` (cf §13).
5. Diversité contributive : un nouveau dataset doit *augmenter* la diversité du pool (instrument nouveau, ou plage spectrale nouvelle, ou domaine sous-représenté).

### 11.3 Procédure d'admission

1. Soumission via formulaire structuré.
2. Validation automatique (schéma, licence, anti-leakage §13).
3. Revue éditoriale (au moins 2 reviewers du comité).
4. **Vote public archivé** (raisons publiées, qu'elles soient pour ou contre).
5. Période de commentaires communautaire (14 jours) avant entrée effective.

### 11.4 Procédure de sortie

Un dataset peut sortir de `selected` (mais reste dans `all`) si :
- révocation de licence,
- contamination détectée a posteriori,
- nouvelle version `vM+1` du même dataset disponible (l'ancien devient `frozen`).

Toute sortie est documentée et notifiée (rétro-notification §12).

### 11.5 Appel et conflits

Tout contributeur peut faire appel d'une décision (admission refusée, modèle invalidé). Procédure :
1. Demande motivée publique.
2. Réponse motivée du comité sous 30 jours.
3. En cas de désaccord, médiation externe (review par un panel ad hoc de la communauté).

Le **journal des décisions** est publié comme partie intégrante du dépôt git du protocole. Aucune décision tacite.

---

## 12. Citation, attribution et rétro-notification *(nouvelle section, Codex)*

### 12.1 Citation structurée

Chaque artefact contributif porte une `CITATION.cff` (Citation File Format) avec :
- DOI (Zenodo ou autre archive permanente)
- Auteurs, ORCID
- Licence
- Version
- Description courte

Cette structure est validée à la soumission. Les artefacts sans citation valide ne sont pas indexés dans `selected`.

### 12.2 Citation dans les vues dérivées

Toute vue (leaderboard, matrice, ranking) **exporte automatiquement** les citations des artefacts qu'elle agrège. Un utilisateur qui copie un classement obtient aussi un bloc BibTeX/CFF prêt à citer. Cela résout le problème classique des leaderboards où l'attribution se perd à mesure que les agrégations se propagent.

### 12.3 Rétro-notification sur événements

Mécanisme **inédit** dans les benchmarks existants, identifié comme proposition par Codex. Les auteurs sont notifiés automatiquement quand :

- une nouvelle soumission est testée sur leur **dataset** (nouveau modèle utilise leur donnée) ;
- une nouvelle soumission est testée avec leur **préprocessing** ou **augmenter** ;
- leur **modèle** est re-exécuté à l'occasion d'un bump majeur ;
- une vue dérivée éditoriale (e.g. « top-10 PP pour datasets cluster X ») mentionne leur contribution ;
- leur contribution sort de `selected` (avec raison documentée).

Effet attendu : incitation au maintien, signalement précoce de régressions, sentiment d'appartenance communautaire.

### 12.4 Co-auteurship sur la publication méthodologique

Les contributeurs de datasets `selected` au moment du freeze v1.0 sont **automatiquement listés comme co-auteurs** de la publication méthodologique (modèle inspiré des consortiums OpenAlex, MoleculeNet). Les auteurs de modèles `selected` reçoivent un *acknowledgment* formel. Les critères précis (entrée avant freeze, contribution non-triviale) sont publiés.

---

## 13. Protocole anti-leakage et contamination *(nouvelle section, Codex)*

Un dataset entrant dans le pool doit passer une **validation automatique de contamination** avant admission, et tout modèle dont la performance « refuse de chuter » sur un nouveau dataset doit être audité.

### 13.1 Validation d'entrée d'un dataset

- **Doublons spectraux exacts** : hash de chaque ligne X ; refus si > 1 % de doublons internes.
- **Quasi-doublons** : distance cosine ou Pearson ; signaler les paires `(i, j)` avec `corr > 0.999`. Soumis à revue éditoriale.
- **Chevauchement avec datasets existants** : pour chaque dataset déjà admis, vérifier l'absence de samples partagés (par hash) ou de quasi-doublons inter-dataset.
- **Métadonnées encodant y** : si une colonne de métadonnées est trop corrélée avec y (> 0.95), refus ou démarcation explicite. Évite les cas où une « catégorie » porte la cible.
- **Groupes de répétitions** : si le dataset a un identifiant `Sample_ID`, vérifier que les splits proposés ne violent pas le groupement. Si oui, forcer les splits par groupe.
- **Cohérence des unités** : valeurs y dans une plage physiquement plausible ; déclaration explicite de l'unité.

### 13.2 Validation continue : détection de modèles « trop bons »

Une fois en production, un modèle dont le `score_ratio` reste exceptionnel sur un *nouveau* dataset jamais vu doit être marqué `under_review` et audité. Trois suspicions principales :
- contamination amont (le contributeur a vu une fuite de y),
- pretraining sur un superset du nouveau dataset (cas TabPFN ou foundation models),
- bug de fit-on-train détecté tardivement.

Critère opérationnel : si `score_ratio` est dans le quantile 1 % bas (i.e. 1 % des meilleurs jamais observés) sur un dataset entrant *et* la fragilité du modèle sur d'autres datasets est typique, examen requis.

### 13.3 Politique de remediation

- Si contamination dataset : retrait de `selected`, notification.
- Si contamination modèle : tag `pretrained_on_subset` (e.g. TabPFN dont les datasets de pretraining sont publics).
- Si bug : invalidation des runs concernés, demande de re-soumission.

Une **transparence totale** est requise : tout marquage `under_review` ou `retired` est public.

---

## Annexe A — Glossaire

| Terme | Sens |
|---|---|
| **Arène** | métaphore : un cadre d'évaluation ouvert où les méthodes s'affrontent dans des conditions identiques. |
| **Grille canonique** | ensemble structuré de combinaisons d'axes que tout modèle soumis traverse. Versionnée. |
| **Plan factoriel fractionnaire bloqué** | design d'expériences où le produit cartésien complet est remplacé par des blocs ciblés ; les effets estimables et l'aliasing doivent être déclarés explicitement. |
| **Aliasing** | confusion entre un effet principal et une interaction non mesurée par le design — caractéristique des plans réduits. |
| **Bloc factoriel** | sous-ensemble de la grille croisant 1 ou 2 axes en fixant les autres ; permet d'isoler les effets principaux et interactions du premier ordre. |
| **Selected datasets** | sous-ensemble curé du pool, sert de leaderboard de référence. Gouvernance §11. |
| **All datasets** | pool complet (selected + community), sert de leaderboard exploratoire. |
| **Sanity check** | run sur un dataset jouet exécuté localement par le contributeur, joint à la soumission. |
| **Bundle .n4a** | export reproductible nirs4all (modèle + artefacts + EnvCard). |
| **EnvCard** | carte d'environnement (versions, BLAS, GPU determinism, thread env) attachée à chaque run. |
| **Run atomique** | unité expérimentale `(modèle, dataset, split, seed, PP, aug, outlier_filter, target_processing)`. |
| **Score ratio** | rapport du score d'un modèle au score d'un baseline canonique (PLS-canon) sur le même dataset ; invariant à l'échelle de y. |
| **Cluster de transférabilité** | k-means sur profils scalaires de datasets, sert à stratifier les rapports. |
| **Décomposition de variance** | attribution de la variabilité de score aux différents axes (split, seed, dataset, PP) via modèle mixte. |
| **Diagnostic d'alignement physique** | corrélation qualitative entre importances de features et bandes spectroscopiques connues. |
| **Bump majeur** | changement de la grille canonique → re-run de tous les modèles `selected`. |
| **Rétro-citation** | notification automatique aux auteurs de datasets/PP/augmenters/modèles à chaque événement les concernant. |

---

## Annexe B — Format du `pipeline.json` de soumission (esquisse)

```json
{
  "schema_version": "1.0",
  "submission": {
    "id": "auto-generated-uuid-or-author-slug",
    "author": {"name": "...", "email": "...", "affiliation": "...", "orcid": "..."},
    "license": "MIT | CeCILL | ...",
    "citation_cff_path": "CITATION.cff",
    "submitted_at": "ISO-8601",
    "depends_on_external_data": false
  },
  "model": {
    "canonical_name": "my-pls-variant",
    "module": "my_package.estimators",
    "class": "MyEstimator",
    "params": {"n_components": 10},
    "task_types": ["regression"],
    "input_constraints": {"min_n": 20, "max_features": 4096, "signal_types": ["absorbance", "reflectance"]},
    "runtime_tier": "fast",
    "compute_constraints": {"requires_gpu": false, "min_ram_gb": 4}
  },
  "pipeline": [
    {"y_processing": {"class": "MinMaxScaler", "params": {}}},
    {"model": {"$ref": "#/model"}}
  ],
  "hyperparameter_search": null,
  "sanity_check": {
    "dataset_alias": "sample_data/regression",
    "seed": 0,
    "expected_rmse": 0.45,
    "bundle_path": "sanity_check.n4a"
  },
  "declaration": {
    "fit_on_train_only": true,
    "reproducibility_caveats": ["stochastic — averaged on 5 seeds"],
    "pretraining_disclosure": "trained from scratch — no external data"
  }
}
```

Le runtime central valide ce manifeste contre un JSON Schema versionné. Le champ `pretraining_disclosure` est ajouté pour traiter explicitement le cas des foundation models (TabPFN, etc.) — cf §13.

---

## Annexe C — Verdict Codex sur le draft initial

> *« Le document est fort parce qu'il pose enfin la NIRS comme un problème d'évaluation expérimentale contrôlée, pas comme une addition de benchmarks locaux. Les deux faiblesses à corriger avant v0.1 sont la formalisation statistique du plan réduit (publier la structure d'aliasing est non-négociable) et la gouvernance de selected (sans cela, le protocole manque de crédibilité institutionnelle). »*

Les deux points sont intégrés respectivement en §4.4 et §11.

Coupes appliquées suite à la revue Codex :
- §4 axe **augmentation** réduit de 25 à 16 configurations en v0 (5 augmenters × 3 intensités + sans-augmentation) ; extension à 25 reportée en v1.x.
- §6.11 zero-shot routing déplacé en feuille de route v1.x (spéculatif sans pool suffisant).
- §10.1 modèles nommés (NICON, TabPFN) remplacés par **familles de baselines maintenues centralement** (PLS-canon, Ridge-canon, RF-canon, CNN-naïf).

---

## Prochaines étapes

1. Constitution du **comité éditorial** initial et publication de sa composition (§11.1).
2. Liste initiale `selected` (12-20 datasets) avec critères de diversité explicites (§11.2).
3. **Publication de la matrice de design v0.1** et de sa structure d'aliasing (§4.4) — prérequis non-négociable.
4. Implémentation du **runtime minimal** (ordonnancement, persistance, validation soumission).
5. Premier squelette du site web : 4 vues critiques en v0.1 (leaderboard, matrix modèle × dataset, décomposition de variance, fragilité).
6. **Manifeste de gouvernance et de citation** (§§11-12) publié comme document séparé, voté par le comité initial.
7. **Implémentation du protocole anti-leakage** (§13) sur les datasets candidats à `selected`.
8. Définition opérationnelle du **PLS-canon** comme baseline universel maintenue centralement.

Une fois ces huit items franchis, le document passe de `v0.1 DRAFT` à `v0.2 ready-for-call-for-contributions`.
