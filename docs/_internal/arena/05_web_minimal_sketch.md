# Arena Web minimal v0.1 — sketch des 4 vues critiques

> Sketch UX/dataviz pour la page publique v0.1. **Arena designe ici le website**, pas la base de preuves interne. Pas d'implémentation détaillée — ce document fixe les contrats de données entre les exports du runtime/evidence engine et le frontend, ainsi que les 4 vues minimales qui doivent ouvrir le service public.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §6 (13 vues envisagées) |
| Liens | [concept + stockage minimal](00_arena_concept_storage.md) ; [grille v0.1b](03_canonical_grid_v0.1.md) ; [runtime](04_runtime_sketch.md) |
| Version | v0.1 |
| Statut | Sketch |
| Phase roadmap | sketch écrit en **Phase 0a** (Conception) ; implémenté en **Phase 4** (Web v0.1) ; ouvert publiquement en **Phase 5** (Bêta). Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 1. Périmètre v0.1

Quatre vues critiques. Une fois ces quatre vues fonctionnelles et alimentées par la base de preuves, le service public peut ouvrir.

| # | Vue | Question primaire | Source de données |
|---|---|---|---|
| V1 | Leaderboard global | "Quel modèle marche le mieux sur le pool selected ?" | B1 |
| V2 | Matrix modèle × dataset | "Quel modèle marche sur quel dataset ?" | B1 |
| V3 | Décomposition de variance | "Quelle part de variation vient du modèle vs du split vs du dataset ?" | B1 + B2 |
| V4 | Matrice de fragilité | "Quels modèles plantent et où ?" | tous blocs (status colonne) |

Les autres vues du manifeste (PP ranking, augmentation effect, scaling laws, clusters, evidence cards détaillées, recommandations de trousse à outils, …) sont v0.2 ou internes en v0.1.

## 2. Contrats de données

Le frontend ne sait *rien* du runtime. Il consomme une API ou des fichiers JSON statiques régénérés à chaque mise à jour. Ces fichiers sont des **snapshots publiables**, reconstruits depuis `arena.sqlite` + les workspaces `nirs4all` référencés ; ils ne sont pas la source primaire d'audit.

Cinq tables logiques publiques exposées :

### 2.1 `submissions.json`
```json
{
  "schema_version": "1.0",
  "generated_at": "2026-05-13T14:00:00Z",
  "grid_version": "v0.1b",
  "dataset_registry_version": "datasets-v0.1",
  "source_snapshot": {
    "arena_sqlite_hash": "blake2b:...",
    "workspace_set_hash": "blake2b:..."
  },
  "submissions": [
    {
      "submission_id": "pls-canon",
      "canonical_name": "PLS-canon",
      "author": {"name": "...", "orcid": "..."},
      "license": "MIT",
      "citation_doi": "...",
      "submitted_at": "...",
      "is_baseline": true,
      "task_types": ["regression"],
      "runtime_tier": "fast",
      "selected": true,
      "lineage": {
        "manifest_hash": "blake2b:...",
        "citation_hash": "blake2b:..."
      }
    },
    ...
  ]
}
```

### 2.2 `leaderboard_global.json` (pour V1)
```json
{
  "schema_version": "1.0",
  "view": "leaderboard_global",
  "source_snapshot": {
    "arena_sqlite_hash": "blake2b:...",
    "workspace_set_hash": "blake2b:..."
  },
  "filters": {
    "task": "regression",
    "dataset_pool": "selected",
    "block": "B1",
    "metric": "rmse_normalized"
  },
  "entries": [
    {
      "rank": 1,
      "submission_id": "tabpfn-v2.5",
      "canonical_name": "TabPFN v2.5",
      "score_ratio_vs_pls_canon_median": 0.78,
      "score_ratio_vs_pls_canon_ci95": [0.71, 0.85],
      "score_ratio_vs_pls_canon_iqr": [0.74, 0.83],
      "n_datasets_ok": 25,
      "n_datasets_failed": 1,
      "fragility_pct": 3.8,
      "fit_time_median_s": 45.2,
      "fit_time_p90_s": 89.0,
      "is_baseline": false,
      "detail_ref": "models/tabpfn-v2.5.json"
    },
    ...
  ]
}
```

### 2.3 `matrix_model_dataset.json` (pour V2)
```json
{
  "schema_version": "1.0",
  "view": "matrix_model_dataset",
  "task": "regression",
  "block": "B1",
  "rows": ["PLS-canon", "Ridge-canon", "RF-canon", ...],
  "cols": ["dataset_alias_1", "dataset_alias_2", ...],
  "cell_metric": "score_ratio_vs_pls_canon_median",
  "cells": [
    [1.0, 1.0, 1.0, ...],       // PLS-canon ratio = 1 par construction
    [0.95, 1.08, 0.91, ...],    // Ridge-canon
    ...
  ],
  "cell_status": [
    ["ok", "ok", "ok", ...],
    ["ok", "ok", "ok", ...],
    ...
  ]
}
```

### 2.4 `variance_decomposition.json` (pour V3)
```json
{
  "schema_version": "1.0",
  "view": "variance_decomposition",
  "task": "regression",
  "blocks": ["B1", "B2"],
  "model_used": "linear mixed: score ~ 1 + (1|model) + (1|dataset) + (1|split) + (1|seed)",
  "variance_components": [
    {"source": "dataset", "variance_pct": 62.4, "ci95": [58.0, 66.5]},
    {"source": "model", "variance_pct": 18.7, "ci95": [15.2, 22.4]},
    {"source": "split", "variance_pct": 7.1, "ci95": [5.4, 9.2]},
    {"source": "seed", "variance_pct": 2.8, "ci95": [2.0, 3.8]},
    {"source": "residual", "variance_pct": 9.0, "ci95": [7.5, 10.8]}
  ],
  "model_diagnostics": {
    "n_observations": 1220,
    "convergence_ok": true,
    "rank_warning": false
  }
}
```

### 2.5 `fragility_matrix.json` (pour V4)
```json
{
  "schema_version": "1.0",
  "view": "fragility_matrix",
  "task": "regression",
  "rows": ["PLS-canon", "Ridge-canon", ...],
  "cols": ["dataset_alias_1", ...],
  "cells_failure_pct": [
    [0.0, 0.0, ...],
    [0.0, 0.0, ...],
    ...
  ],
  "cells_failure_class_majority": [
    ["none", "none", ...],
    ["none", "none", ...],
    ...
  ]
}
```

Ces 5 fichiers sont générés par l'indexer à chaque publication (cf [04_runtime_sketch.md](04_runtime_sketch.md) §3.7). Ils sont versionnés (`generated_at`, `source_snapshot`, hash du résultat) pour debug et reproductibilité.

## 2.6 Drill-down interne inspiré du studio

Le studio/Inspector existant expose déjà de bons concepts pour une interface mainteneur :

- workspaces liés et scannés automatiquement ;
- runs groupés par dataset, pipeline, modèle, preprocessing et split ;
- statuts `completed/failed/partial`, `completed_results`, `failed_results` ;
- fiches dataset avec hash/version/statut ;
- accès aux prédictions et aux artefacts via `WorkspaceStore`.

Pour v0.1, ces concepts ne sont pas obligatoires dans le site public. Ils peuvent servir à une page interne `/ops/runs/<id>` ou à une CLI `arena inspect`, alimentée directement par `arena.sqlite` + `WorkspaceStore`, sans exposer tous les runs atomiques au public.

## 2.7 Exports internes hors website

Le website ne doit pas faire oublier les sorties les plus utiles a court terme :

- `evidence/recipe_cards/*.json` : fiches actionnables par recette.
- `evidence/anti_patterns.json` : combinaisons rarement utiles ou trop fragiles.
- `evidence/toolkit_recommendations.json` : recommandations conditionnelles pour la trousse a outils.
- `evidence/coverage_report.json` : cadence de criblage, domaines couverts, methodes restantes.

Ces exports peuvent rester prives au debut. Une partie pourra devenir publique plus tard sous forme de pages `/recipes/`, `/recommendations/` ou de supplement scientifique.

## 3. Vues — spécifications

### V1 — Leaderboard global

**Données** : `leaderboard_global.json`

**Layout proposé** :
```
┌──────────────────────────────────────────────────────────────────────┐
│ NIRS-Arena ⋯ Leaderboard                       [Task: Regression ▾]  │
│                                                                       │
│ Filtres :                                                             │
│   Dataset pool : ⦿ Selected   ○ All                                  │
│   Bloc : B1 (core)                                                    │
│   Métrique : score_ratio_vs_pls_canon (lower is better)              │
│                                                                       │
│ ┌────┬──────────────────┬──────────────┬───────────┬───────────────┐│
│ │ #  │ Modèle           │ Ratio (med)  │ IC 95%    │ Fragilité (%) ││
│ ├────┼──────────────────┼──────────────┼───────────┼───────────────┤│
│ │ 1  │ TabPFN v2.5      │ 0.78         │ [0.71,..] │ 3.8           ││
│ │ 2  │ AOM-PLS-best     │ 0.82         │ [0.77,..] │ 0.0           ││
│ │ 3  │ AOM-Ridge        │ 0.85         │ [0.80,..] │ 0.4           ││
│ │ ⋯  │                  │              │           │               ││
│ │ N  │ PLS-canon        │ 1.00         │ ---       │ 0.0           ││
│ │ N+1│ Ridge-canon      │ 1.03         │ [0.98,..] │ 0.0           ││
│ └────┴──────────────────┴──────────────┴───────────┴───────────────┘│
│                                                                       │
│ ➔ Click sur modèle : profil détaillé (V2 filtré)                     │
└──────────────────────────────────────────────────────────────────────┘
```

**Règles d'affichage** :
- Baselines (`is_baseline=true`) toujours visibles, signalées graphiquement (badge "baseline").
- Tri par défaut : `score_ratio_vs_pls_canon_median` croissant (lower-is-better).
- Possibilité de trier par `fragility_pct` (croissant) ou `fit_time_median_s` (croissant) — révèle les Pareto.
- Filtres "Selected" (curé) vs "All" (pool complet).
- Conditionnalité affichée : un encart "Mesuré sous : `P_canon = SNV+SG+StandardScaler`, `A=None`, `F=None`, `split ∈ {KS_70_30, SPXY_70_30}`."

### V2 — Matrix modèle × dataset

**Données** : `matrix_model_dataset.json`

**Layout** :
- Heatmap classique. Lignes = modèles ; colonnes = datasets ; cellule = `score_ratio_vs_pls_canon` médian (couleur).
- Lignes triables par moyenne, médiane, rang.
- Colonnes triables alphabétiquement, par taille (n_samples du dataset), par cluster de transférabilité.
- Cellules vides (`status != "ok"`) marquées explicitement (hachures + tooltip avec `failure_class`).
- Tooltips : `score_ratio`, `n_seeds_ok`, IC, lien vers les runs individuels.

**Couleur** : palette divergente centrée sur 1.0 (jaune-vert = bat PLS-canon, jaune-rouge = perd contre PLS-canon).

### V3 — Décomposition de variance

**Données** : `variance_decomposition.json`

**Layout** :
- Barplot empilé : 5 segments (dataset, model, split, seed, residual), tailles proportionnelles.
- Affichage explicite des pourcentages.
- Annotation : "Note : ces composantes sont estimées via modèle mixte conditionnellement à `P_canon` et `A_none`. Voir [aliasing_warnings.json] pour les conditions."
- Filtre : par tâche, par bloc (B1, B2, ou union).

**But pédagogique** : montrer au lecteur qu'un score médian global cache de fortes disparités si la variance dataset est dominante. Sans cette vue, V1 est trompeuse.

### V4 — Matrice de fragilité

**Données** : `fragility_matrix.json`

**Layout** :
- Heatmap analogue à V2, mais cellule = `cells_failure_pct ∈ [0, 100]`.
- Couleur : palette séquentielle blanc → rouge.
- Tooltip : `cells_failure_class_majority` (e.g. "OOM dominant", "Timeout dominant", "NaN dominant").
- Tri possible par modèle (modèles les plus fragiles en haut) ou par dataset (datasets sur lesquels le plus de modèles plantent).

**But** : équilibre la vue V1. Un modèle avec `ratio=0.75` mais fragilité 30 % n'est pas une recommandation responsable.

## 4. Stack technique — proposition de discussion

Trois options pour v0.1 :

### Option A — Site statique + JSON
- **Génération** : runtime exporte les 5 fichiers JSON ; un build script (Hugo, Eleventy, Astro, ou pur HTML+JS) génère le site.
- **Frontend** : HTML + Vega-Lite pour les visualisations (heatmaps, barplots).
- **Avantages** : ultra-léger, hébergeable sur GitHub Pages, pas de backend, déterministe.
- **Inconvénients** : pas de drill-down dynamique sur les runs atomiques sans backend.

### Option B — Datasette + frontend léger
- **Génération** : runtime persiste `arena.sqlite` + workspaces `store.sqlite` ; Datasette sert l'API SQLite ; un frontend léger (Svelte, React, ou Vega-Lite + JS pur) consomme.
- **Avantages** : drill-down dynamique sur tous les runs ; queries SQL exposées.
- **Inconvénients** : nécessite un hosting backend ; complexité de déploiement.

### Option C — FastAPI + frontend React/Vue
- **Génération** : runtime persiste `arena.sqlite` + workspaces `store.sqlite`/Parquet ; FastAPI sert l'API ; React/Vue pour le frontend.
- **Avantages** : flexibilité maximale, UI riche.
- **Inconvénients** : plus de dev, plus de maintenance.

**Recommandation v0.1** : **Option A** (statique). Suffisant pour les 4 vues critiques. Migration vers B ou C en v0.2 dès que la traction le justifie.

## 5. Layout général du site

```
/                     → page d'accueil : pitch + leaderboard V1
/leaderboard/         → V1 plein écran avec filtres
/matrix/              → V2 (matrice modèle × dataset)
/variance/            → V3 (décomposition de variance)
/fragility/           → V4 (matrice de fragilité)
/models/<id>/         → profil détaillé d'une soumission (toutes ses lignes)
/datasets/<alias>/    → profil détaillé d'un dataset
/datasets/            → liste des datasets (selected + all)
/protocol/            → protocole v0.1b complet (rendu du manifeste)
/grid/                → grille v0.1b (rendu du doc 03)
/about/               → gouvernance, citation, contact
/submit/              → comment soumettre (placeholder v0.1, PR Git ; UI v0.2)
```

## 6. Accessibilité et i18n

- Pages disponibles en français et anglais (au minimum).
- Heatmaps avec palettes colorblind-safe (cf `viridis`, `cividis`).
- Tooltips informatifs sur tous les éléments visuels.
- Tables triables au clavier (a11y).

## 7. Versioning et déploiement

- Chaque release du runtime → nouvelle release du site.
- Le site porte un footer `grid_version: v0.1b`, `data_version: <hash>`, `generated_at: <ISO>`.
- Anciens snapshots accessibles via `/v(N)-frozen/` pour reproductibilité.

## 8. Tests d'acceptation v0.1

Avant ouverture publique :

1. Les 4 vues V1-V4 s'affichent correctement avec ≥ 3 soumissions (les 3 baselines).
2. Les 5 fichiers JSON sont régénérables par le runtime sans erreur.
3. Au moins 1 cycle complet : nouvelle soumission → runtime exécute → 5 JSONs mis à jour → site reflète le changement.
4. Site accessible publiquement (GitHub Pages ou équivalent).
5. Site rendant correctement sur mobile (au moins V1 + V2 + V4, V3 peut être desktop-first).

## 9. Hors-périmètre v0.1 (à planifier v0.2)

- Comparaison interactive HELM-like entre deux modèles (§6.14.2 manifeste).
- Vues d'incertitude (variance decomposition par cellule, pas seulement globale).
- Vues NIRS-spécifiques (loadings PLS, Q-Q residuals).
- Drill-down dynamique sur les runs atomiques.
- Filtres avancés (par cluster de dataset, par instrument).
- Export utilisateur (CSV custom, BibTeX bundle).
- Submission UI (upload form, validation visuelle).

## 10. Décisions à arbitrer

- **Option A vs B vs C** (cf §4).
- **Hosting** : GitHub Pages (gratuit, statique seulement) ou serveur custom ?
- **Domaine** : `nirs-arena.org` (à acquérir) ou sous-domaine de `nirs4all.dev` (à créer) ?
- **Logo / branding** : marketing minimal en v0.1, ou polish dès v0.1 ?
- **Open data policy** : exporter tout, ou nécessiter login pour téléchargement de bundles `.n4a` lourds ?
