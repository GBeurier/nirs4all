# Design: Unified Sample Grouping in nirs4all

> **Status**: Draft — April 2026
> **Scope**: `nirs4all` library (not webapp)

---

## 1. Problematique

### 1.1 Le probleme fondamental

Un jeu de donnees NIRS contient des spectres individuels, mais ces spectres ne sont pas independants. Un meme echantillon biologique peut avoir ete mesure plusieurs fois (repetitions spectrales), provenir d'un site ou d'une annee specifique, ou etre relie a d'autres echantillons par n'importe quelle variable de metadata. Ces relations entre spectres ont un impact a **chaque etape** du pipeline ML, et chaque etape a des besoins differents:

| Etape du pipeline | Besoin de groupage | Consequence si ignore |
|---|---|---|
| **Split / CV** | Tous les spectres d'un meme groupe doivent rester dans le meme fold | Data leakage: les performances reportees sont sur-estimees |
| **Scoring / Reporting** | Les predictions des repetitions d'un meme echantillon doivent pouvoir etre aggregees | Les scores ne refletent pas la precision reelle au niveau de l'echantillon |
| **Ranking / Selection** | Le meilleur modele devrait (potentiellement) etre choisi sur les scores agreges | On peut selectionner un modele qui overfitte les repetitions individuelles |
| **Refit** | Le refit sur donnees completes doit respecter la meme logique de groupage | Incoherence entre le modele CV et le modele final |
| **Finetuning** | L'optimisation des hyperparametres (Optuna) devrait pouvoir optimiser le score agrege | On optimise une metrique qui n'est pas celle qui compte en production |

### 1.2 Les types de groupage

Aujourd'hui, nirs4all melange trois concepts distincts sous le meme vocabulaire (`repetition`, `aggregate`, `group_by`):

#### A. Repetitions spectrales (identite physique)

Plusieurs spectres d'un **meme echantillon physique**, donc avec le meme y. C'est un fait intrinseque au jeu de donnees, pas un choix de l'utilisateur. Exemples:
- 3 scans d'une meme pomme
- 5 mesures d'un meme echantillon de sol

**Impact universel**: Ces repetitions doivent etre groupees au split (anti-leakage), et les predictions doivent etre aggregees pour tout reporting (un echantillon = un score). C'est le groupage le plus fondamental.

#### B. Groupage anti-leakage generalise (dependance statistique)

Des echantillons **differents** (y potentiellement differents) qui partagent une dependance statistique et ne doivent pas apparaitre simultanement dans train et val. Exemples:
- Echantillons d'un meme site (effet site)
- Echantillons d'une meme annee (effet millésime)
- Combinaison site × annee (tuple de colonnes)
- Patient dans une etude longitudinale

**Impact au split uniquement**: Les groupes doivent etre respectes au split, mais il n'y a pas de raison d'agreger les predictions (chaque echantillon est distinct).

#### C. Agregation de predictions (reporting)

Regrouper des predictions **au moment du reporting** pour calculer des scores au niveau d'une entite plus large. Exemples:
- Score moyen par site
- Score median par annee
- Score par patient (repetitions spectrales)

**Impact reporting uniquement**: Ne change ni le split ni l'entrainement. C'est une vue sur les resultats.

### 1.3 Etat actuel: le melange des concepts

Le code actuel (v0.8.7) melange ces trois concepts:

**Dans `DatasetConfigs`:**
- `repetition` → concept A, propage au split ET a l'agregation
- `aggregate` → concept C, mais auto-propage comme concept A quand c'est un string
- `aggregate_method`, `aggregate_exclude_outliers` → concept C

**Dans le split controller:**
- `group_by` → concept B, combinable avec `repetition` (concept A)
- `aggregation`, `y_aggregation` → parametres du GroupedSplitterWrapper

**Dans `Predictions.top()`:**
- `by_repetition` → concept C (mal nomme: c'est de l'agregation de predictions)
- `repetition_method`, `repetition_exclude_outliers` → concept C

**Dans `TabReportManager`:**
- `aggregate`, `aggregate_method`, `aggregate_exclude_outliers` → concept C

**Problemes concrets:**
1. `repetition` et `aggregate` se configurent mutuellement en boucle (config.py L344-353)
2. `by_repetition` dans `top()` n'a rien a voir avec les repetitions spectrales — c'est de l'agregation de scores
3. Le `group_by` du split n'est pas accessible depuis `DatasetConfigs` ni `run()`
4. Aucun mecanisme dans le refit/orchestrateur pour utiliser des scores agreges pour la selection finale du meilleur modele
5. Un seul niveau d'agregation de score: on ne peut pas exprimer proprement "agreger par sample_id puis par site" ou "mediane par variete x site"
6. La semantique de combinaison des contraintes de split n'est pas formalisee: tuple de colonnes, composantes connexes, ou combinaison de plusieurs contraintes sont confondus
7. Le `aggregate_exclude_outliers` traverse tout le stack depuis `DatasetConfigs` jusqu'aux charts — couplage excessif

### 1.4 Ce que veut l'utilisateur

L'utilisateur veut pouvoir dire simplement:
1. "Mes spectres sont regroupes par `Sample_ID` — gere ca partout automatiquement"
2. "En plus, ne melange pas les sites dans train/val"
3. "Montre-moi les scores par site"
4. "Choisis le meilleur modele en se basant sur le RMSE agrege par echantillon"

---

## 2. Propositions de design

### 2.1 Principe directeur: separation des preoccupations

Le design repose sur une separation nette des trois concepts en objets distincts, avec un vocabulaire non-ambig:

| Concept | Nom propose | Portee | Lieu de definition |
|---|---|---|---|
| A. Identite physique | **`sample_id`** | Dataset-wide | `DatasetConfigs` / config dict |
| B. Anti-leakage | **`groups`** | Split-specific | Split step dans le pipeline |
| C. Reporting | **`report_by`** | Post-prediction | `result.top()`, `PredictionAnalyzer`, auto pour `sample_id` |

#### Pourquoi ces noms?

- **`sample_id`** plutot que `repetition`: Le terme "repetition" decrit le phenomene (des mesures repetees), pas le mecanisme (un identifiant de groupage). `sample_id` est ce que l'utilisateur manipule reellement: une colonne de metadata qui identifie l'echantillon physique. C'est aussi un terme API plus explicite pour l'utilisateur.

- **`groups`** plutot que `group_by`: `group_by` evoque SQL/pandas. `groups` est le terme sklearn standard (`GroupKFold(groups=...)`) et sera naturel pour tout utilisateur familier avec sklearn.

- **`report_by`** plutot que `aggregate`: "aggregate" est trop generique (aggreger quoi? comment?). `report_by` exprime clairement l'intention: "rapporte les resultats groupes par cette colonne".

### 2.2 Design haut niveau

```
                    ┌────────────────────────┐
                    │     DatasetConfigs      │
                    │  sample_id="Sample_ID"  │
                    └──────────┬─────────────┘
                               │
                    ┌──────────▼─────────────┐
                    │     SpectroDataset      │
                    │  .sample_id = "..."     │
                    └──────────┬─────────────┘
                               │
              ┌────────────────┼───────────────────┐
              │                │                    │
    ┌─────────▼────────┐  ┌───▼──────────┐  ┌─────▼────────────┐
    │   Split / CV     │  │   Training   │  │   Scoring        │
    │                  │  │              │  │                  │
    │ Automatique:     │  │  (inchange)  │  │ Automatique:     │
    │ sample_id →      │  │              │  │ sample_id →      │
    │   group par      │  │              │  │   report par     │
    │   defaut         │  │              │  │   defaut         │
    │                  │  │              │  │                  │
    │ Explicite:       │  │              │  │ Explicite:       │
    │ groups=["Site"]  │  │              │  │ report_by="Site" │
    │ groups_mode=     │  │              │  │ report_method=   │
    │   "tuple"        │  │              │  │   "median"       │
    └──────────────────┘  └──────────────┘  └──────────────────┘
```

### 2.3 Regles de propagation automatique

Le `sample_id` est le seul parametre qui se propage automatiquement:

1. **Au split**: Si `sample_id` est defini et aucune contrainte explicite dans le step de split, le split utilise automatiquement `sample_id` comme contrainte anti-leakage. Si un `groups` ou un `split_constraints` explicite est fourni, `sample_id` est ajoute comme **contrainte supplementaire** sauf si `ignore_sample_id=True`. Point important: `sample_id` ne doit pas etre "tuple" avec les autres colonnes par defaut; il doit etre compose avec elles au niveau des contraintes effectives de split.

2. **Au scoring/reporting**: Si `sample_id` est defini, le `TabReportManager` affiche automatiquement une ligne de scores agreges par `sample_id` en plus des scores bruts. C'est le comportement actuel mais avec un mecanisme propre.

3. **Au ranking/finetuning (opt-in)**: Un nouveau parametre `score_by` ou `score_plan` dans `refit` et finetuning permet de choisir si la selection utilise les scores bruts ou des scores agregees par `sample_id` ou par une entite plus large.

Les concepts B (groups) et C (report_by) ne se propagent **jamais** automatiquement — ils sont toujours explicites la ou ils sont utilises.

### 2.4 Composition des contraintes de split

Les termes `union` / `intersection` sont trop ambigus ici. Il faut distinguer trois questions differentes:

1. Comment une **contrainte elementaire** interprete plusieurs colonnes ?
2. Comment plusieurs **contraintes elementaires** se combinent-elles entre elles ?
3. Comment transformer ces contraintes en labels de groupes utilisables par un splitter ?

Proposition: formaliser le split comme un ensemble de **contraintes**. Chaque contrainte relie des echantillons, puis le split opere sur les **composantes connexes** de l'union de toutes les contraintes.

Deux modes elementaires suffisent pour couvrir la plupart des cas:

- **`mode="tuple"`**: deux echantillons sont relies s'ils ont le meme tuple de valeurs sur toutes les colonnes
- **`mode="connected"`**: deux echantillons sont relies s'ils partagent au moins une valeur sur les colonnes de la contrainte; la transitivite cree ensuite des composantes connexes potentiellement tres larges

Exemples:

```python
# Un seul raccourci: meme Site ET meme Year
# Equivalent a une contrainte {"keys": ["Site", "Year"], "mode": "tuple"}
{"split": KFold(5), "groups": ["Site", "Year"], "groups_mode": "tuple"}
# Groupe = (Site_A, 2020), (Site_A, 2021), (Site_B, 2020), ...

# Meme Site OU meme Year
# Equivalent a une contrainte {"keys": ["Site", "Year"], "mode": "connected"}
{"split": KFold(5), "groups": ["Site", "Year"], "groups_mode": "connected"}

# Cas reel: repetitions + interdiction de melanger les sites
# sample_id n'est PAS tuple avec Site; c'est une contrainte supplementaire
{
    "split": KFold(5),
    "groups": "Site",
    # Contraintes effectives:
    #   1. meme sample_id -> meme composante
    #   2. meme Site -> meme composante
}

# Cas avance: plusieurs contraintes explicites
{
    "split": KFold(5),
    "split_constraints": [
        {"keys": "$sample_id", "mode": "tuple"},
        {"keys": "Site", "mode": "tuple"},
        {"keys": ["Year", "Campaign"], "mode": "tuple"},
    ],
}
```

**Choix de design**:

- `groups=` reste un raccourci pour **une seule contrainte**
- `groups_mode="tuple"` est le defaut pour ce raccourci
- `split_constraints=[...]` est l'API generale quand il faut combiner plusieurs contraintes heterogenes
- une colonne composite creee en amont ne reproduit que la semantique `tuple`, jamais la semantique `connected`

Le mode `connected` signifie: deux spectres sont dans le meme composant s'ils partagent **au moins une** colonne de la contrainte. Concretement, si `groups=["Site", "Year"]` avec `groups_mode="connected"`:
- Sample(Site_A, 2020) et Sample(Site_A, 2021) → meme groupe (meme Site)
- Sample(Site_A, 2020) et Sample(Site_B, 2020) → meme groupe (meme Year)
- Ceci cree des groupes transitifs (union-find) potentiellement tres larges

Attention: pour l'anti-leakage, `tuple` est en general **moins restrictif** que `connected`, car il cree des groupes plus petits.

### 2.5 Gestion des outliers

L'exclusion d'outliers dans l'agregation est un detail d'implementation, pas un concept de premier ordre. Il doit etre configure la ou l'agregation est definie, pas propage depuis la config du dataset:

```python
# Niveau report — l'utilisateur controle au moment de la visualisation
result.top(5, report_by="$sample_id", report_method="median", exclude_outliers=True)

# Niveau analyzer — default pour toutes les charts
analyzer = PredictionAnalyzer(
    predictions,
    default_report_by="$sample_id",
    default_exclude_outliers=True,
)

# PAS au niveau DatasetConfigs — ca n'a rien a faire la
```

### 2.6 Impact sur le ranking, le refit et le finetuning

Aujourd'hui, `Predictions.top()` sait deja reranker sur des scores agreges, mais pas le refit/orchestrateur ni l'objectif de finetuning Optuna. On propose donc deux niveaux d'API:

- **API simple**: `score_by` pour une seule etape d'agregation
- **API generale**: `score_plan` pour une sequence d'agregations successives

Exemples:

```python
# Defaut: ranking sur scores bruts (comportement actuel)
nirs4all.run(..., refit=True)

# Opt-in: ranking sur scores agreges par sample_id
nirs4all.run(..., refit={"ranking": "rmsecv", "score_by": "$sample_id"})

# Avance: ranking sur scores agreges par une autre colonne
nirs4all.run(..., refit={"ranking": "rmsecv", "score_by": "Site"})

# Multi-niveaux: d'abord sample_id, puis mediane par variete x site
nirs4all.run(..., refit={
    "ranking": "rmsecv",
    "score_plan": [
        {"by": "$sample_id", "pred": "mean", "true": "first"},
        {"by": ["Variety", "Site"], "pred": "median", "true": "median"},
    ],
})
```

Le `score_by` dans le refit reprend la meme logique que `report_by`: on agrege les predictions avant de calculer la metrique de ranking. `"$sample_id"` signifie "utiliser le `sample_id` du dataset". Un string explicite (ex: `"Site"`) permet de  scorer par une autre colonne. `score_plan` permet d'exprimer des reductions successives quand une seule colonne ne suffit pas.

**Pour le finetuning Optuna**: `score_by` / `score_plan` s'applique aussi a la fonction objectif. Cela permet d'optimiser:
- un RMSE agrege par echantillon
- une mediane par `["Variety", "Site"]`
- ou une chaine explicite du type "mean par sample_id, puis median par variete x site"

---

## 3. Nomenclature et parametres

### 3.1 Parametres renommes

| Actuel | Propose | Raison |
|---|---|---|
| `DatasetConfigs(repetition=)` | `DatasetConfigs(sample_id=)` | Plus descriptif, standard en chimiometrie |
| `DatasetConfigs(aggregate=)` | **Supprime** | Split en `sample_id` (auto) et `report_by` (explicite) |
| `DatasetConfigs(aggregate_method=)` | **Supprime** | Deplace vers `report_by` options |
| `DatasetConfigs(aggregate_exclude_outliers=)` | **Supprime** | Deplace vers point d'usage |
| `dataset.set_repetition()` | `dataset.set_sample_id()` | Coherent avec le renommage |
| `dataset.repetition` | `dataset.sample_id` | Coherent avec le renommage |
| `dataset.repetition_groups` | `dataset.sample_groups` | Coherent avec le renommage |
| `dataset.repetition_stats` | `dataset.sample_stats` | Coherent avec le renommage |
| `top(by_repetition=)` | `top(report_by=)` | Nom plus clair, concept C |
| `top(repetition_method=)` | `top(report_method=)` | Coherent avec le renommage |
| `top(repetition_exclude_outliers=)` | `top(exclude_outliers=)` | Simplifie |
| Split step `group_by=` | Split step `groups=` | Aligne sur sklearn |

### 3.2 Objets et parametres par couche

#### Couche Dataset: `DatasetConfigs`

```python
DatasetConfigs(
    configurations=...,
    task_type="auto",
    signal_type=None,

    # CONCEPT A: Identite physique
    sample_id: str | None = None,
    #   Colonne metadata identifiant l'echantillon physique.
    #   Se propage auto au split (groupage) et au reporting (agregation).
)
```

On supprime `aggregate`, `aggregate_method`, `aggregate_exclude_outliers` de `DatasetConfigs` comme parametres dataset-level de premier ordre. Ces besoins restent utiles, mais doivent etre exprimes au point d'usage via `report_by` / `report_plan` et `score_by` / `score_plan`, pas couples a la definition structurelle du dataset.

#### Couche Split: step de pipeline

```python
# Minimal — le sample_id fait tout
pipeline = [
    KFold(5),           # sample_id auto-utilise comme groups
    PLSRegression(10),
]

# Explicite — groupes supplementaires
pipeline = [
    {
        "split": KFold(5),
        "groups": ["Site"],          # Anti-leakage supplementaire
        # sample_id ajoute automatiquement comme contrainte separee
        # Resultat effectif:
        #   meme sample_id -> meme composante
        #   meme Site      -> meme composante
        "groups_mode": "tuple",  # defaut sur le raccourci groups=
        "ignore_sample_id": False,       # defaut
    },
    PLSRegression(10),
]

# Override complet — ignore sample_id
pipeline = [
    {
        "split": GroupKFold(5),
        "groups": "Site",
        "ignore_sample_id": True,  # NE PAS inclure sample_id dans les groupes
    },
    PLSRegression(10),
]
```

**Parametres du step split:**

| Parametre | Type | Defaut | Description |
|---|---|---|---|
| `groups` | `str \| list[str] \| None` | `None` | Raccourci pour une contrainte de split basee sur une ou plusieurs colonnes metadata |
| `groups_mode` | `"tuple" \| "connected"` | `"tuple"` | Semantique de la contrainte definie par `groups` |
| `split_constraints` | `list[dict] \| None` | `None` | API generale pour combiner plusieurs contraintes heterogenes |
| `ignore_sample_id` | `bool` | `False` | Ne pas inclure `sample_id` dans les groupes du split |
| `aggregation` | `str` | `"mean"` | Methode d'agregation pour le GroupedSplitterWrapper (X) |
| `y_aggregation` | `str \| None` | `None` (auto) | Methode d'agregation pour y dans le wrapper |

#### Couche Scoring: `Predictions.top()` et `PredictionAnalyzer`

```python
# top() - parametres renommes
result.top(
    n=5,
    report_by="$sample_id",      # Agreger les predictions par sample_id du dataset
    report_method="mean",        # mean, median, vote
    exclude_outliers=False,      # Exclure les outliers avant agregation
)

# Agregation par une colonne ou un tuple explicite
result.top(n=5, report_by=["Variety", "Site"], report_method="median")

# Version generale: plusieurs etapes de reduction
result.top(
    n=5,
    report_plan=[
        {"by": "$sample_id", "pred": "mean", "true": "first"},
        {"by": ["Variety", "Site"], "pred": "median", "true": "median"},
    ],
)

# PredictionAnalyzer — defaults pour toutes les charts
analyzer = PredictionAnalyzer(
    predictions,
    default_report_by="$sample_id",
    default_report_method="mean",
    default_exclude_outliers=False,
)
```

**`report_by` valeurs possibles:**

| Valeur | Comportement |
|---|---|
| `None` / `False` | Pas d'agregation (scores bruts) |
| `"$sample_id"` | Utilise le `sample_id` du dataset |
| `"y"` | Agrege par valeur de y (utile pour classification ou y discret) |
| `["<col1>", "<col2>"]` | Agrege par tuple de colonnes metadata |
| `"<column_name>"` | Agrege par cette colonne de metadata |

Note: `"$sample_id"` est prefere a la chaine `"sample_id"` pour eviter toute ambiguite avec une colonne metadata qui s'appellerait litteralement `"sample_id"`.

#### Couche Refit/Ranking: `refit` param

```python
# Defaut: ranking sur scores bruts
nirs4all.run(..., refit=True)

# Score agrege pour le ranking
nirs4all.run(..., refit={
    "ranking": "rmsecv",
    "score_by": "$sample_id",          # Ranking sur scores agreges par sample_id
    "score_method": "mean",            # Methode d'agregation pour le ranking
})

# Multi-critere avec scores agreges
nirs4all.run(..., refit=[
    {"top_k": 3, "ranking": "rmsecv"},                                # Brut
    {"top_k": 1, "ranking": "rmsecv", "score_by": "$sample_id"},     # Agrege par sample_id
])
```

**Nouveaux parametres de `RefitCriterion`:**

| Parametre | Type | Defaut | Description |
|---|---|---|---|
| `score_by` | `str \| list[str] \| None` | `None` | Raccourci pour une seule etape d'agregation avant scoring |
| `score_plan` | `list[dict] \| None` | `None` | Sequence d'etapes d'agregation avant scoring |
| `score_method` | `str` | `"mean"` | Methode d'agregation pour le scoring |

#### Couche TabReport: generation automatique

Le `TabReportManager` change de comportement:

- **Si `sample_id` est defini**: affiche automatiquement deux lignes par partition — scores bruts ET scores agreges par sample_id (marques avec `*`)
- **Pas de parametres d'agregation dans l'API du report**: le report utilise le `sample_id` du dataset automatiquement. Pour des agregations custom, passer par `PredictionAnalyzer`.

Cela supprime le passage de `aggregate`, `aggregate_method`, `aggregate_exclude_outliers` a travers tout le stack orchestrateur → report.

### 3.3 `run()` API

L'API `run()` elle-meme ne change pas structurellement — `sample_id` est dans `DatasetConfigs`, pas dans `run()`:

```python
# Via DatasetConfigs (methode recommandee)
nirs4all.run(
    pipeline=[KFold(5), PLSRegression(10)],
    dataset=DatasetConfigs("path/to/data", sample_id="Sample_ID"),
    refit={"ranking": "rmsecv", "score_by": "$sample_id"},
)

# Via config dict
nirs4all.run(
    pipeline=[KFold(5), PLSRegression(10)],
    dataset={"train_x": "data.csv", "sample_id": "Sample_ID"},
)
```

### 3.4 Resume visuel de la nomenclature

```
DatasetConfigs(sample_id="S_ID")
       │
       ├─ SpectroDataset.sample_id → "S_ID"
       │
       ├─ Split: contrainte auto sur S_ID + contrainte explicite sur Site
       │         → composantes connexes qui respectent S_ID et Site
       │
       ├─ TabReport: auto dual row (raw + aggregated by S_ID)
       │
       ├─ Refit: score_by="$sample_id" → rank on aggregated scores
       │
       └─ Predictions.top(report_by="$sample_id") → aggregated results
```

---

## 4. Backlog haut niveau

### Phase 1: Renommage + simplification (breaking changes)

**Impact**: Renommage des parametres publics, suppression du couplage agregation↔dataset.

| Module | Action |
|---|---|
| `data/config.py` | Renommer `repetition` → `sample_id`. Supprimer `aggregate`, `aggregate_method`, `aggregate_exclude_outliers` du constructeur. Supprimer la logique de propagation croisee repetition↔aggregate. |
| `data/dataset.py` | Renommer `_repetition` → `_sample_id`, `set_repetition()` → `set_sample_id()`, `repetition` → `sample_id`, `repetition_groups` → `sample_groups`, `repetition_stats` → `sample_stats`. Supprimer `set_aggregate()`, `set_aggregate_method()`, `set_aggregate_exclude_outliers()`. Supprimer les proprietes `aggregate`, `aggregate_method`, `aggregate_exclude_outliers`. |
| `data/schema/config.py` | Mettre a jour le schema: `repetition` → `sample_id`, supprimer `aggregate*` |
| `data/parsers/normalizer.py` | Mettre a jour le parsing: `repetition` → `sample_id`, `aggregate` → supprime |
| `data/serialization/serializer.py` | Mettre a jour la serialisation |
| `config/validator.py` | Mettre a jour les validations |
| `data/aggregation/aggregator.py` | L'`Aggregator` pour l'agregation au chargement reste inchange (concept different). En revanche, `AggregationConfig.from_config()` ne lit plus `aggregate*` depuis le dataset config — l'agregation au chargement doit etre explicite. |
| `operators/data/repetition.py` | Renommer references `repetition` → `sample_id` dans `RepetitionConfig` (resolve_column, etc.) |
| Tests impactes | `test_config.py`, `test_parsers.py`, `test_dataset.py` + tout test utilisant les anciens noms |

### Phase 2: Mise a jour du split controller

**Impact**: Renommer `group_by` → `groups`, ajouter `groups_mode`, `split_constraints`, `ignore_sample_id`.

| Module | Action |
|---|---|
| `controllers/splitters/split.py` | Renommer `group_by` → `groups` dans le parsing du step dict. Ajouter `groups_mode` ("tuple"/"connected"), `split_constraints` et `ignore_sample_id`. Mettre a jour `compute_effective_groups()` ou introduire un resolver dedie pour construire les composantes connexes a partir des contraintes. |
| `operators/splitters/grouped_wrapper.py` | Ne pas y mettre la logique metier de combinaison des contraintes; le wrapper consomme simplement les groupes effectifs deja resolus. |
| `controllers/data/repetition.py` | Mettre a jour les references `repetition` → `sample_id` |
| Tests impactes | `test_splitters.py`, `test_grouped_wrapper.py`, `test_group_split_validation.py`, integration tests |

### Phase 3: Refactoring du scoring/predictions

**Impact**: Renommer les parametres d'agregation dans `Predictions.top()`, `PredictionAnalyzer`, et charts; ajouter la notion de plan d'agregation multi-etapes.

| Module | Action |
|---|---|
| `data/predictions.py` | Renommer `by_repetition` → `report_by`, `repetition_method` → `report_method`, `repetition_exclude_outliers` → `exclude_outliers`. Ajouter `report_plan` pour les reductions multi-etapes. Renommer `_dataset_repetition` → `_dataset_sample_id`, `set_repetition_column()` → `set_sample_id_column()`. Supprimer `set_aggregate_context()` (plus necessaire). |
| `visualization/predictions.py` | Renommer `default_aggregate*` → `default_report_by`, `default_report_method`, `default_exclude_outliers`. Ajouter `default_report_plan` si necessaire. |
| `visualization/charts/base.py` | Mettre a jour `_get_ranked_predictions()`: `aggregate` → `report_by`, etc. |
| `visualization/charts/*.py` | Mettre a jour tous les charts (candlestick, histogram, heatmap, confusion_matrix, top_k_comparison) |
| `visualization/reports.py` | Simplifier `TabReportManager`: lire `sample_id` depuis les predictions context, supprimer les parametres `aggregate*` des signatures. Auto-afficher la ligne agregee si `sample_id` present. |
| Tests impactes | `test_predictions.py`, `test_prediction_analyzer.py`, chart tests |

### Phase 4: Scoring agrege pour le ranking/refit

**Impact**: Nouveau `score_by` / `score_plan` dans `RefitCriterion`, propagation a `extract_top_configs()` et Optuna.

| Module | Action |
|---|---|
| `pipeline/execution/refit/config_extractor.py` | Ajouter `score_by`, `score_plan` et `score_method` a `RefitCriterion`. Modifier `extract_top_configs()` et `_compute_mean_val_scores()` pour supporter le scoring agrege et multi-etapes. |
| `pipeline/execution/orchestrator.py` | Propager `score_by` / `score_plan` au ranking. Modifier `_execute_refit_pass()`. |
| `pipeline/execution/refit/model_selector.py` | Supporter `score_by` / `score_plan` dans `_aggregate_scores_per_variant()`. |
| `pipeline/config/context.py` | Mettre a jour `BestChainEntry` si necessaire. |
| `optimization/optuna.py` | Supporter `score_by` / `score_plan` dans la fonction objectif. |
| Tests impactes | Refit tests, model_selector tests, optuna tests |

### Phase 5: Nettoyage

| Module | Action |
|---|---|
| `pipeline/runner.py` | Supprimer `_last_aggregate_column`, `_last_aggregate_method`, `_last_aggregate_exclude_outliers` et les proprietes correspondantes. |
| `pipeline/execution/orchestrator.py` | Supprimer `last_aggregate_column`, `last_aggregate_method`, `last_aggregate_exclude_outliers`. Supprimer tout le threading d'`aggregate*` a travers les methodes de reporting. |
| `api/result.py` | Mettre a jour si necessaire. |
| `CLAUDE.md` | Mettre a jour la documentation des parametres. |
| `docs/` | Mettre a jour les guides utilisateur (aggregation.md, loading_data.md, etc.). |
| Examples | Mettre a jour tous les exemples utilisant les anciens noms. |

---

## 5. Avis et retour

### 5.1 Etat des lieux: ce qui est bien fait

Le systeme actuel a de bonnes bases:
- Le `GroupedSplitterWrapper` est elegant et generalise bien le groupage a tout splitter sklearn
- La composition de colonnes via tuples fonctionne
- Le mecanisme de leakage detection est precieux
- La separation `rep_to_sources` / `rep_to_pp` pour les transformations de repetitions est un concept unique et puissant

### 5.2 Ce qui pose probleme

Le probleme principal est un cas classique de **feature creep organique**: des besoins ont ete ajoutes incrementalement sans refactoring du modele conceptuel. Le resultat est un couplage serré entre des concepts orthogonaux:

1. **DatasetConfigs comme fourre-tout**: La config du dataset ne devrait pas savoir comment les predictions seront aggregees ni avec quelle methode. C'est une violation de separation des preoccupations.

2. **Le threading de parametres**: `aggregate_exclude_outliers` traverse 7 couches (DatasetConfigs → SpectroDataset → Orchestrator → TabReport → Predictions.aggregate). Chaque couche le transporte sans le transformer. C'est un signe que le parametre est au mauvais endroit.

3. **La confusion terminologique**: "repetition", "aggregate", "group_by", "by_repetition" designent des aspects du meme probleme mais sont incoherents dans leur nommage et leur semantique.

### 5.3 Retour sur les pratiques ML/DL standard

En ML/DL, la gestion des groupes est bien codifiee:

**sklearn**: Le pattern standard est `GroupKFold(groups=groups)`. Les groupes sont passes au moment du split, pas configures sur le dataset. C'est le bon pattern.

**Pratiques en chimiometrie/NIRS**:
- La notion d'identifiant d'echantillon physique est fondamentale
- L'agregation des repetitions spectrales est standard (typiquement: mean du spectre avant le modele, ou mean des predictions apres)
- La separation site/annee au split est une pratique de validation externe classique

**PyTorch/Deep Learning**:
- `DataLoader` + `Sampler` permettent d'implementer un echantillonnage groupe, mais il n'existe pas d'equivalent natif aussi direct que `GroupKFold`
- L'agregation est une etape post-inference explicite
- Pas de couplage entre la config du dataset et la strategie d'evaluation

### 5.4 Le concept disruptif: `sample_id` comme citoyen de premiere classe

Le changement le plus important dans ce design n'est pas le renommage des parametres — c'est l'elevation de `sample_id` au rang de propriete fondamentale du dataset au meme titre que X, y, et metadata.

Dans la plupart des frameworks ML, un dataset est un ensemble de (X_i, y_i). Mais en NIRS, un dataset est un ensemble de (X_ij, y_i) ou j indexe les repetitions du meme echantillon i. Cette structure hierarchique est **intrinseque** aux donnees NIRS.

En faisant de `sample_id` un parametre de premier ordre du dataset (et non un ajout d'agregation), on reconnait cette structure et on permet au framework de l'exploiter automatiquement a chaque etape. C'est la difference entre "l'utilisateur doit penser a configurer le groupage a chaque etape" et "l'utilisateur declare la structure de ses donnees une fois, et le framework fait le reste".

### 5.5 Distinction importante: agregation au chargement vs. agregation des predictions

Il existe un 4e type d'agregation qui n'est pas couvert par le probleme de groupage mais qui partage le vocabulaire: l'**agregation au chargement** (`data/aggregation/aggregator.py`). Elle consiste a pre-agreger les spectres d'un meme groupe en un seul spectre (mean/median) **avant** de les injecter dans le pipeline. C'est un choix de preprocessing, pas un choix de reporting.

Cette agregation au chargement est conceptuellement distincte et ne devrait PAS etre confondue avec `report_by` (concept C). Elle reste un step de preprocessing configurable separement, potentiellement via un step pipeline (`{"aggregate_samples": "Sample_ID"}`) ou une option de chargement du dataset.

Le design propose ne modifie pas cette fonctionnalite — elle reste en l'etat dans `Aggregator`.

### 5.6 Risques et compromis

1. **Breaking changes**: Le renommage des parametres publics casse la compatibilite. C'est acceptable pour une version majeure (0.9.0) mais doit etre documente.

2. **Ambiguite semantique si on garde `union` / `intersection`**: ces termes donnent une intuition trompeuse. Il vaut mieux parler explicitement de `tuple` vs `connected`, sinon l'utilisateur pensera a tort qu'ajouter `sample_id` a `Site` par tuple suffit a interdire le melange des sites.

3. **Composantes connexes trop larges**: le mode `connected` peut creer des groupes tres larges qui rendent le split impossible. Il faut un warning quand la taille du plus grand groupe depasse un seuil (ex: 50% des echantillons).

4. **Score agrege pour le ranking**: Agreger avant de scorer change la metrique. Un RMSE sur 100 predictions agregees n'est pas le meme que sur 500 predictions brutes. Il faut documenter clairement cette difference et ne pas en faire le defaut.

5. **Simplicite vs expressivite**: Le design propose est plus expressif que l'actuel mais aussi plus complexe a documenter. Le `sample_id` comme parametre auto-propage plus des raccourcis (`groups`, `report_by`, `score_by`) est le bon compromis, a condition d'avoir une API generale (`split_constraints`, `report_plan`, `score_plan`) pour les cas avances.

6. **`sample_id` vs colonne litterale `sample_id`**: Si l'utilisateur a une colonne metadata qui s'appelle litteralement `"sample_id"`, il n'y a pas de conflit car `DatasetConfigs(sample_id="sample_id")` signifie "la colonne sample_id est l'identifiant physique". Le token `"$sample_id"` resout l'ambiguite dans les APIs de score/report.

---

## 6. Generalisation des contraintes et des scopes

### 6.1 Pourquoi `union` / `intersection` n'est pas suffisant

Les termes `union` / `intersection` ne disent pas explicitement:

1. comment une contrainte elementaire interprete plusieurs colonnes
2. comment plusieurs contraintes elementaires se combinent
3. a quel niveau du pipeline on applique une reduction de grain

Or ce sont trois problemes differents:

- **Split**: exprimer des dependances a ne jamais casser entre train et val
- **Score**: exprimer a quel niveau on veut comparer prediction et observation
- **Report**: exprimer a quel niveau on veut afficher ou analyser les resultats

Une meme colonne peut intervenir dans plusieurs de ces problemes, mais avec des roles differents.

### 6.2 Proposition: deux briques de premier ordre

#### A. `split_constraints`: contraintes de fuite

Le split doit raisonner en termes de **contraintes**, pas en termes d'une seule colonne de groupe.

Chaque contrainte a:
- `keys`: une colonne, plusieurs colonnes, ou `"$sample_id"`
- `mode`: au minimum `"tuple"` ou `"connected"`

Les groupes effectifs de split sont ensuite les **composantes connexes** induites par l'union de toutes les contraintes.

Cela permet d'exprimer correctement:
- repetitions spectrales (`"$sample_id"`)
- "ne pas melanger les sites"
- "ne pas melanger les couples site x annee"
- "ne pas melanger les echantillons qui partagent un site OU une annee"

#### B. `score_plan` / `report_plan`: reductions de grain

Le score et le report doivent raisonner en termes de **plan de reduction**. Une seule valeur `score_by="Site"` ou `report_by="Site"` ne suffit pas quand on veut plusieurs niveaux.

Chaque etape d'un plan a:
- `by`: colonne, tuple de colonnes, ou `"$sample_id"`
- `pred`: comment reduire `y_pred` (`mean`, `median`, `vote`, `first`, ...)
- `true`: comment reduire `y_true`
- optionnellement `proba`, `exclude_outliers`, etc.

Le pipeline de score/report applique ces etapes successivement, puis calcule la metrique ou affiche le resultat au grain final.

### 6.3 Exemples de cas d'usage

#### Split: repetitions + interdiction de melanger les sites

```python
{
    "split": KFold(5),
    "split_constraints": [
        {"keys": "$sample_id", "mode": "tuple"},
        {"keys": "Site", "mode": "tuple"},
    ],
}
```

Semantique:
- toutes les repetitions d'un sample restent ensemble
- tous les echantillons d'un meme site restent ensemble

#### Split: site x annee comme unite de validation externe

```python
{
    "split": KFold(5),
    "split_constraints": [
        {"keys": ["Site", "Year"], "mode": "tuple"},
    ],
}
```

Semantique:
- un groupe correspond au tuple `(Site, Year)`
- deux echantillons de meme site mais d'annees differentes peuvent etre dans des folds differents

#### Split: meme site OU meme annee

```python
{
    "split": KFold(5),
    "split_constraints": [
        {"keys": ["Site", "Year"], "mode": "connected"},
    ],
}
```

Semantique:
- partage du site -> meme composante
- partage de l'annee -> meme composante
- transitivite explicite

#### Finetuning/refit: mediane predite par variete x site

Si l'objectif metier est:

> "la mediane par variete par site predite doit etre proche de la mediane observee"

alors il faut pouvoir ecrire:

```python
refit = {
    "ranking": "rmsecv",
    "score_plan": [
        {"by": "$sample_id", "pred": "mean", "true": "first"},
        {"by": ["Variety", "Site"], "pred": "median", "true": "median"},
    ],
}
```

Ce plan veut dire:
1. si plusieurs spectres appartiennent au meme echantillon physique, on les collapse en une prediction d'echantillon
2. on regrouppe ensuite les echantillons par `(Variety, Site)`
3. on compare la mediane predite a la mediane observee a ce grain

Le meme plan doit pouvoir etre reutilise:
- pour le **refit**
- pour l'objectif de **finetuning Optuna**
- pour un **report** metier specifique

### 6.4 Consequence sur l'API

Pour garder une API simple:

- `groups`, `report_by`, `score_by` restent des **raccourcis**
- `split_constraints`, `report_plan`, `score_plan` deviennent l'API **generale**

Exemples de sucres:

```python
report_by="$sample_id"
score_by="Site"
groups=["Site", "Year"]  # shorthand pour une seule contrainte
```

Exemples avances:

```python
split_constraints=[...]
report_plan=[...]
score_plan=[...]
```

### 6.5 Invariants a expliciter avant implementation

Cette generalisation impose de formaliser plusieurs invariants:

- Un plan de reduction doit reduire ou conserver le grain, jamais l'expandre
- `"$sample_id"` doit etre valide comme colonne resolue du dataset, pas comme colonne litterale
- Si un groupe `"$sample_id"` contient plusieurs `y` incompatibles, il faut une erreur ou un warning fort
- Les valeurs manquantes dans les cles de split ou de score doivent avoir une politique explicite
- `rmsecv` et `mean_val` doivent definir clairement a quel moment le `score_plan` est applique
- Le `TabReportManager` auto ne doit sans doute afficher la ligne sample-level que si le plan reduit effectivement le nombre de lignes

### 6.6 Recommendation

La bonne integration n'est pas d'ajouter encore plus de flags `union` / `intersection` sur les APIs actuelles. La bonne integration est:

1. modeliser le **split** comme un ensemble de contraintes explicites
2. modeliser le **score/report** comme un plan de reduction explicite
3. garder des raccourcis simples pour les cas usuels
4. partager la meme semantique entre `Predictions.top()`, `PredictionAnalyzer`, `refit` et `Optuna`

Autrement dit:
- `sample_id` reste la structure fondamentale du dataset
- `split_constraints` decrit les dependances a preserver
- `score_plan` decrit l'unite metier sur laquelle on veut optimiser
- `report_plan` decrit l'unite metier sur laquelle on veut raconter les resultats
