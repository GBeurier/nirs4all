# Grille canonique v0.1 — plan factoriel fractionnaire bloqué et structure d'aliasing

> Document de référence statistique. **Publier la structure d'aliasing est non-négociable** (revue Codex). Toute soumission au benchmark traverse exactement la grille définie ici ; toute conclusion qui en est tirée doit explicitement prendre en compte les effets *non identifiables*.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §4 |
| Version | v0.1b (intègre revue Codex initiale) |
| Statut | À soumettre à seconde revue avant freeze |
| Phase roadmap | rédigé en **Phase 0a** (Conception) ; générateur + tests en **Phase 2** ; exécuté à chaque soumission dès Phase 2. Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 0. Changelog v0.1 → v0.1b (post-revue Codex)

1. **Bug B2 split count corrigé** : le code listait 7 schémas de split (`SPLIT_SCHEMES_REG`), la spec annonçait 8. Cohérence imposée à 7. B2 = 20 × 7 × 5 = **700 runs** (au lieu de 800 annoncés). Total révisé : 3 660 runs/soumission.
2. **§5.1 reformulé en "effets prévus dans le modèle d'analyse"** (pas une vraie matrice d'aliasing formelle) avec formules statistiques par bloc, contrastes et degrés de liberté.
3. **Stitching escape valve programmatique** (§5.3) : tout export dataviz porte un champ `measured_condition_only`, et le stitching est *interdit* en production v0.1. Diagnostic d'interaction obligatoire avant toute extrapolation.
4. **Tests par rangs sur datasets** (Demšar 2006 *JMLR* ; Friedman + Nemenyi/Conover) privilégiés sur l'ANOVA hiérarchique. Variance composantes en analyse exploratoire seulement.
5. **`P_canon` figé en amont, pas sélectionné a posteriori** : la suggestion v0.1 d'utiliser "best PP de B3" est retirée ; à la place, 2-3 *chaînes ancres* sont gelées avant publication. Le protocole publie aussi un *robustness check* : refaire B4/B5 sous une seconde ancre PP, comparer les conclusions.
6. **B6 et B7 explicitement marqués "pilot/aspirational"** ; expansion en v0.2.
7. **Stratégie de sous-échantillonnage N spécifiée** (§3.B6) : stratifiée par bin de y en régression, stratifiée par classe en classification, seed contrôlée. Test set tenu hors-réduction.
8. **Bug code corrigé** : ordre canonique des inputs (datasets, splits, PPs, augs, filtres) figé via constantes ; B7 stocke explicitement la direction `train→test`.

## 1. Facteurs et niveaux

| Facteur | Symbole | Type | Niveaux v0.1 | Cardinalité |
|---|---|---|---|---:|
| Modèle | M | catégoriel | toutes les soumissions du runtime | par soumission : 1 |
| Dataset | D | catégoriel | union `selected_v0.1 = fast12 ∪ audit20` | ≈ 26 |
| Schéma de split | S | catégoriel | régression : `{KS_70_30, SPXY_70_30, Random_70_30, KS_5fold, SPXY_5fold, Random_5fold, KBinsStratified_5fold}` ; classification : `{KS_70_30, SPXY_70_30, Random_70_30, KS_5fold, SPXY_5fold, Random_5fold, Stratified_5fold}` | **7** |
| Seed | K | catégoriel ordonné | `{0, …, 9}` | 10 |
| Chaîne de préprocessing | P | catégoriel | 15 chaînes (voir §1.1) | 15 |
| Augmentation | A | catégoriel | 16 configurations (voir §1.2) | 16 |
| Filtre d'outliers | F | catégoriel | 5 niveaux | 5 |
| Target processing | T | catégoriel | régression `{T0_none, T1_minmax, T2_standard, T3_robust}` ; classif `{T0_none}` | 4 / 1 |
| Sample size | N | catégoriel ordonné | `{50, 100, 200, 500, 1000, all}` | 6 |

**Niveaux canoniques** (notation `*_canon`) :
- `P_canon = P5_snv_sg_standardscaler` (chaîne ancre principale)
- `P_canon2 = P12_snv_sg_minmax` (chaîne ancre alternative, sert au robustness check)
- `A_canon = A0_none`
- `F_canon = F0_none`
- `T_canon_reg = T1_minmax` ; `T_canon_clf = T0_none`
- `N_canon = all`
- `S_canon = KS_70_30`

### 1.1 Chaînes de préprocessing

```
P0_raw, P1_snv, P2_msc, P3_sg_d1, P4_sg_d2,
P5_snv_sg_standardscaler [= P_canon],
P6_msc_sg_standardscaler,
P7_snv_sg_detrend,
P8_msc_sg_detrend,
P9_detrend_snv_sg,
P10_snv_cars50,
P11_snv_osc2,
P12_snv_sg_minmax [= P_canon2],
P13_snv_sg_robust,
P14_msc_sg_robust
```

### 1.2 Configurations d'augmentation

```
A0_none,
A1_gauss_{low,med,high},
A2_shift_{low,med,high},
A3_scatter_{low,med,high},
A4_batch_{low,med,high},
A5_mixup_{low,med,high}
```

(`A1..A4` sont position-raw ; `A5` est position-représentationnelle, appliquée après PP.)

### 1.3 Filtres d'outliers

```
F0_none, F1_y_iqr_1.5, F2_x_mahalanobis_5pct, F3_both_any, F4_tag_y_iqr
```

## 2. Cardinalité totale et infaisabilité du produit cartésien

Produit cartésien complet :
```
|D| × |S| × |K| × |P| × |A| × |F| × |T| × |N|
= 26 × 7 × 10 × 15 × 16 × 5 × 4 × 6
≈ 52 millions de runs par soumission
```

Hors d'atteinte. La grille v0.1 est un **plan factoriel fractionnaire par blocs**, dimensionné à 3 660 runs par soumission (~14 000× moins).

## 3. Architecture par blocs

Chaque bloc fixe un sous-ensemble de facteurs à leurs valeurs canoniques et fait varier 1-2 facteurs ciblés. Les effets estimables sont **conditionnels aux valeurs canoniques des facteurs fixés**.

### B1 — Bloc principal (core matrix)

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | tous les `selected_v0.1` (26 datasets) |
| S | `{KS_70_30, SPXY_70_30}` (2 niveaux) |
| K | `{0, …, 9}` (10 seeds) |
| P, A, F, T, N | tous fixés à `_canon` |

**Runs/soumission** : 26 × 2 × 10 = **520**

**Modèle d'analyse exploratoire** (variance components) :
```
score_obs = μ + α_D + β_S + γ_K + (αβ)_DS + (αγ)_DK + (βγ)_SK + ε
```
Avec 1 réplique par cellule `(D, S, K)`, l'interaction de troisième ordre `D×S×K` est *confondue avec le résidu* et n'est pas estimable séparément. L'analyse exploratoire utilise un modèle mixte avec D, S, K comme effets aléatoires ; les interactions à 2 facteurs sont estimables. Pour M (la soumission), la comparaison se fait *vs* les autres soumissions déjà indexées dans l'arène sur le même `(D, S, K)`.

**Recommandation de reporting** : pour comparer soumissions, suivre Demšar 2006 — Friedman + post-hoc Nemenyi par dataset, plutôt qu'ANOVA. Décomposition de variance reportée séparément comme diagnostic, pas comme test confirmatoire.

### B2 — Bloc cross-split

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | `audit20_transfer_core` (20 datasets) |
| S | **les 7 schémas** |
| K | `{0, 1, 2, 3, 4}` (5 seeds) |
| P, A, F, T, N | fixés à `_canon` |

**Runs/soumission** : 20 × 7 × 5 = **700**

**Modèle d'analyse** : Kendall's W sur les rangs des soumissions à travers les 7 schémas (concordance multi-juge) ; complémenté par décomposition `score = μ + α_D + β_S + (αβ)_DS + γ_K + ε` en mixte.

### B3 — Bloc préprocessing exploratoire

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | `fast12_transfer_core` (12 datasets) |
| S | `KS_70_30` |
| K | `{0, 1, 2, 3, 4}` |
| P | **les 15 chaînes** |
| A, F, T, N | `_canon` |

**Runs/soumission** : 12 × 15 × 5 = **900**

**Modèle d'analyse** : Friedman sur les 15 chaînes pour chaque modèle ; post-hoc Nemenyi pour les paires significatives. Holm pour multiplicité quand on compare plusieurs modèles.

### B4 — Bloc augmentation

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | `fast12_transfer_core` (12 datasets) |
| S | `KS_70_30` |
| K | `{0, 1, 2, 3, 4}` |
| P | `P_canon` |
| A | **les 16 configurations** |
| F, T, N | `_canon` |

**Runs/soumission** : 12 × 16 × 5 = **960**

**Robustness check obligatoire** : pour les augmentations classées *utiles* (cf critères dans le manifeste §3.2), refaire le test sous `P_canon2` pour vérifier que la conclusion ne dépend pas du choix d'ancre PP. Coût : 12 × 5 (configurations top-5) × 5 = 300 runs additionnels — **inclus dans B4 dans le total ci-dessus si jamais l'analyse retient ≤ 5 augmenters comme top**.

### B5 — Bloc outliers

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | `fast12_transfer_core` (12 datasets) |
| S | `KS_70_30` |
| K | `{0, 1, 2, 3, 4}` |
| P | `P_canon` |
| A | `A_none` |
| F | **les 5 niveaux** |
| T, N | `_canon` |

**Runs/soumission** : 12 × 5 × 5 = **300**

### B6 — Bloc scaling laws *(pilot/aspirational)*

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | sous-ensemble de 6 datasets `large_enough` (à définir : `n_samples ≥ 500` strict) |
| S | `KS_70_30` |
| K | `{0, 1, 2, 3, 4}` |
| P | `P_canon` |
| A | `A_none` |
| F | `F_none` |
| T | `T_canon` |
| N | **`{50, 100, 200, 500, 1000, all}`** |

**Runs/soumission** : 6 × 6 × 5 = **180**

**Statut explicite** : *pilot* — IC larges, courbes individuelles non-publiables comme conclusions, agrégation multi-modèles uniquement. Expansion à 30 seeds et 12 datasets en v0.2.

**Stratégie de sous-échantillonnage N** (jamais omettre cette précision) :
- **Régression** : sous-échantillonnage stratifié par bins de y (`KBinsDiscretizer(n_bins=5, strategy="quantile")`) — garde la distribution de y. Seed contrôlée par `(seed, target_size)`.
- **Classification** : sous-échantillonnage stratifié par classe — proportions préservées.
- **Test set inchangé** : le sous-échantillonnage ne touche que le *train*. Le test reste celui défini par le split. C'est essentiel pour que la courbe `score(N)` mesure une *learning curve* et pas un effet conjoint apprentissage+évaluation.

### B7 — Bloc cross-instrument *(pilot/aspirational)*

| Facteur | Configuration |
|---|---|
| M | soumission courante |
| D | paires `cross_instrument` (≥ 3 paires requises — à confirmer par la cartographie T5.4) |
| S | `KS_70_30` sur le train_instrument |
| K | `{0, 1, 2, 3, 4}` |
| P | `P_canon` |
| A | `{A_none, A4_batch_med}` |
| **direction** | `train→test` ∈ {`A→B`, `B→A`} pour chaque paire `(A, B)` |

**Runs/soumission** : (n_pairs) × 2 directions × 5 seeds × 2 niveaux A.
- Si n_pairs = 3 : 60 runs ; si 5 : 100 runs.

**Statut explicite** : *pilot* — démonstration de l'axe cross-instrument plus que mesure populationnelle. Toute conclusion limitée aux paires mesurées (interdiction de généraliser à d'autres instruments). Expansion en v0.2 dès qu'au moins 10 paires existent.

## 4. Cardinalité totale v0.1b

| Bloc | Runs | Statut |
|---|---:|---|
| B1 core | 520 | core |
| B2 cross-split | 700 | core |
| B3 PP | 900 | core |
| B4 augmentation | 960 | core |
| B5 outliers | 300 | core |
| B6 scaling laws | 180 | pilot |
| B7 cross-instrument | 60-100 | pilot |
| **Total** | **≈ 3 620-3 660** | |

À 1-30 s/run sur modèle léger : 1-30 h-CPU. À 10-100 s/run sur GPU : 10-100 h-GPU. **Tractable** côté cluster.

## 5. Effets prévus dans le modèle d'analyse

### 5.1 Effets estimables par bloc (formels)

Pour chaque bloc, on précise (i) le modèle d'analyse, (ii) les contrastes estimables, (iii) les degrés de liberté résiduels, (iv) les effets *exclus* (fixés au canon).

**B1** :
- Modèle : `score = μ + α_D + β_S + γ_K + (αβ)_DS + (αγ)_DK + (βγ)_SK + ε`
- Effets estimables : `α_D` (25 dof), `β_S` (1 dof), `γ_K` (9 dof), `(αβ)_DS` (25 dof), `(αγ)_DK` (225 dof), `(βγ)_SK` (9 dof).
- Résidu : 520 - 1 - 25 - 1 - 9 - 25 - 225 - 9 = 225 dof. Suffisant pour une décomposition de variance exploratoire.
- Effets exclus (fixés) : P, A, F, T, N, tous les croisements impliquant ces facteurs.

**B2** : modèle `score = μ + α_D + β_S + γ_K + (αβ)_DS + ε` sur 700 cellules ; effets estimables D (19 dof), S (6 dof), K (4 dof), D×S (114 dof) ; résidu 556 dof.

**B3** : modèle `score = μ + α_D + δ_P + γ_K + (αδ)_DP + ε` sur 900 cellules ; D (11), P (14), K (4), D×P (154) ; résidu 716.

**B4** : modèle analogue à B3 avec A en place de P ; D (11), A (15), K (4), D×A (165) ; résidu 764.

**B5** : modèle analogue ; D (11), F (4), K (4), D×F (44) ; résidu 236.

**B6** : modèle `score = μ + α_D + ν_N + γ_K + (αν)_DN + ε` sur 180 cellules ; D (5), N (5), K (4), D×N (25) ; résidu 140 — *faible*, d'où le statut pilot.

**B7** : modèle `score = μ + α_pair + δ_direction + λ_A + γ_K + interactions limitées + ε` ; nombre de cellules dépend de n_pairs ; tous les effets sont exploratoires.

### 5.2 Effets *non* estimables sur la grille v0.1

Aucun des suivants n'est identifiable et leur invocation est interdite :

- `M × P × S` (B3 a S = KS uniquement) — l'effet PP × modèle est conditionnel à KS.
- `M × A × P`, `A × P` (B4 a P = P_canon uniquement).
- `M × F × P`, `F × P` (B5 a P = P_canon).
- Toute interaction d'ordre 3 entre {S, P, A, F, T, N}.
- Toute interaction modèle × dataset × { P, A, F, T, N } d'ordre 4+.

**Sortie obligatoire** de la grille : un fichier `aliasing_warnings_v0.1b.json` listant ces non-identifiabilités, importé par la dataviz pour bloquer toute visualisation qui les invoque.

### 5.3 Stitching post-hoc — interdiction programmatique

Le draft v0.1 mentionnait le "stitching" sous hypothèse d'additivité. La revue Codex a flaggé le risque que le stitching ré-apparaisse silencieusement dans la dataviz malgré l'avertissement. **Décision v0.1b** :

- Chaque export (CSV, JSON, image) porte un champ obligatoire `measured_condition_only ∈ {true, false}`.
- Toute vue dataviz qui afficherait une estimation impliquant des effets non mesurés (donc `measured_condition_only=false`) est **bloquée à la génération v0.1**.
- Le stitching est autorisé dans des *notebooks d'analyse* explicites, avec un préambule de diagnostic d'interaction obligatoire (test que les interactions à 2 facteurs sont petites avant d'invoquer l'additivité).
- À v0.2, possibilité de stitching contrôlé dans la dataviz *si* un diagnostic d'interaction est passé et reportés ensemble.

## 6. Choix méthodologiques justifiés

### 6.1 Pourquoi un design par blocs plutôt qu'un fractionnaire classique

Trois raisons :
1. **Lisibilité.** Plans Box-Hunter-Hunter ou Plackett-Burman supposent des facteurs binaires ou peu de niveaux ; nos facteurs sont catégoriels à 5-16 niveaux.
2. **Coût asymétrique.** Augmenter K coûte 10× plus qu'augmenter P. Plans uniformes sous-optimaux.
3. **Alignement sur questions scientifiques.** Les blocs correspondent un-à-un aux questions Q1-Q11 du manifeste.

### 6.2 Pourquoi `P_canon` est figé en amont et *pas* sélectionné depuis B3

La revue Codex a flaggé un risque post-selection : si `P_canon` était choisi comme top de B3 sur fast12, alors B4/B5/B6 conditionnés sur `P_canon` seraient biaisés par sélection sur les *mêmes datasets*. **Décision v0.1b** : `P_canon` est figé par convention chimiométrique (SNV+SG+StandardScaler) **avant** toute exécution de la grille. C'est une hypothèse a priori, pas une post-sélection. Robustness check via `P_canon2` couvre le risque que le choix soit pathologiquement mauvais.

### 6.3 Pourquoi `D ∈ fast12` plutôt que `D ∈ selected` dans B3-B5

Coût. Élargir B3 à 26 datasets multiplie son coût par > 2 ; même chose pour B4 (le plus gros bloc). En v0.2, expansion à 20-26 datasets une fois le signal scientifique stabilisé.

### 6.4 Pourquoi 10 seeds en B1, 5 ailleurs

B1 fournit l'IC du score principal — 10 seeds donne une variance inter-seed correctement estimée pour le bootstrap. Dans les blocs exploratoires, la variance inter-seed est réutilisée *depuis B1* comme baseline pour les IC ; 5 seeds locales suffisent pour estimer les effets principaux conditionnellement.

### 6.5 Calibrage du nombre de seeds par analyse de puissance — reporté en v0.2

Reconnaissance que "10 seeds" est un choix initial non calibré statistiquement. **Action v0.2** : exécuter PLS-canon × fast12 × 30 seeds, estimer l'écart-type inter-seed empirique, recalculer le nombre minimal pour détecter `δ = 0.05` sur `score_ratio` médian avec puissance 0.8.

## 7. Générateur Python (référence, déterministe)

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


# --- Constantes canoniques (ordre figé pour déterminisme) -------------

P_CANON = "P5_snv_sg_standardscaler"
P_CANON2 = "P12_snv_sg_minmax"
A_CANON = "A0_none"
F_CANON = "F0_none"
T_CANON_REG = "T1_minmax"
T_CANON_CLF = "T0_none"
N_CANON = "all"
S_CANON = "KS_70_30"

PP_CHAINS: tuple[str, ...] = (
    "P0_raw",
    "P1_snv",
    "P2_msc",
    "P3_sg_d1",
    "P4_sg_d2",
    "P5_snv_sg_standardscaler",
    "P6_msc_sg_standardscaler",
    "P7_snv_sg_detrend",
    "P8_msc_sg_detrend",
    "P9_detrend_snv_sg",
    "P10_snv_cars50",
    "P11_snv_osc2",
    "P12_snv_sg_minmax",
    "P13_snv_sg_robust",
    "P14_msc_sg_robust",
)

AUG_CONFIGS: tuple[str, ...] = (
    "A0_none",
    "A1_gauss_low", "A1_gauss_med", "A1_gauss_high",
    "A2_shift_low", "A2_shift_med", "A2_shift_high",
    "A3_scatter_low", "A3_scatter_med", "A3_scatter_high",
    "A4_batch_low", "A4_batch_med", "A4_batch_high",
    "A5_mixup_low", "A5_mixup_med", "A5_mixup_high",
)

OUTLIER_FILTERS: tuple[str, ...] = (
    "F0_none",
    "F1_y_iqr_1.5",
    "F2_x_mahalanobis_5pct",
    "F3_both_any",
    "F4_tag_y_iqr",
)

SPLIT_SCHEMES_REG: tuple[str, ...] = (
    "KS_70_30", "SPXY_70_30", "Random_70_30",
    "KS_5fold", "SPXY_5fold", "Random_5fold",
    "KBinsStratified_5fold",
)

SPLIT_SCHEMES_CLF: tuple[str, ...] = (
    "KS_70_30", "SPXY_70_30", "Random_70_30",
    "KS_5fold", "SPXY_5fold", "Random_5fold",
    "Stratified_5fold",
)

SCALING_LAW_SIZES: tuple[str, ...] = ("50", "100", "200", "500", "1000", "all")


# --- RunSpec (frozen + slots pour hashabilité et compacité) -----------

@dataclass(frozen=True, slots=True)
class RunSpec:
    block: Literal["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
    dataset_alias: str
    split_scheme: str
    seed: int
    pp_chain: str
    augmentation: str
    outlier_filter: str
    target_processing: str
    n_samples: str = "all"
    cross_instrument_direction: str | None = None  # uniquement B7


# --- Générateurs par bloc (déterministes : sortie ordonnée stable) ---

def _sorted_canonical(items: list[str]) -> list[str]:
    """Tri canonique : alphabétique sur l'alias. Garantit le déterminisme
    indépendamment de l'ordre du registry d'entrée."""
    return sorted(items)


def generate_B1(
    selected_datasets: list[str], task: Literal["regression", "classification"]
) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(selected_datasets):
        for s in ("KS_70_30", "SPXY_70_30"):
            for k in range(10):
                runs.append(RunSpec(
                    block="B1", dataset_alias=d, split_scheme=s, seed=k,
                    pp_chain=P_CANON, augmentation=A_CANON, outlier_filter=F_CANON,
                    target_processing=target,
                ))
    return runs


def generate_B2(audit20_datasets: list[str], task: str) -> list[RunSpec]:
    splits = SPLIT_SCHEMES_REG if task == "regression" else SPLIT_SCHEMES_CLF
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(audit20_datasets):
        for s in splits:
            for k in range(5):
                runs.append(RunSpec(
                    block="B2", dataset_alias=d, split_scheme=s, seed=k,
                    pp_chain=P_CANON, augmentation=A_CANON, outlier_filter=F_CANON,
                    target_processing=target,
                ))
    return runs


def generate_B3(fast12_datasets: list[str], task: str) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(fast12_datasets):
        for p in PP_CHAINS:
            for k in range(5):
                runs.append(RunSpec(
                    block="B3", dataset_alias=d, split_scheme=S_CANON, seed=k,
                    pp_chain=p, augmentation=A_CANON, outlier_filter=F_CANON,
                    target_processing=target,
                ))
    return runs


def generate_B4(fast12_datasets: list[str], task: str) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(fast12_datasets):
        for a in AUG_CONFIGS:
            for k in range(5):
                runs.append(RunSpec(
                    block="B4", dataset_alias=d, split_scheme=S_CANON, seed=k,
                    pp_chain=P_CANON, augmentation=a, outlier_filter=F_CANON,
                    target_processing=target,
                ))
    return runs


def generate_B5(fast12_datasets: list[str], task: str) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(fast12_datasets):
        for f in OUTLIER_FILTERS:
            for k in range(5):
                runs.append(RunSpec(
                    block="B5", dataset_alias=d, split_scheme=S_CANON, seed=k,
                    pp_chain=P_CANON, augmentation=A_CANON, outlier_filter=f,
                    target_processing=target,
                ))
    return runs


def generate_B6(scaling_datasets: list[str], task: str) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    runs: list[RunSpec] = []
    for d in _sorted_canonical(scaling_datasets):
        for n in SCALING_LAW_SIZES:
            for k in range(5):
                runs.append(RunSpec(
                    block="B6", dataset_alias=d, split_scheme=S_CANON, seed=k,
                    pp_chain=P_CANON, augmentation=A_CANON, outlier_filter=F_CANON,
                    target_processing=target, n_samples=n,
                ))
    return runs


def generate_B7(
    cross_instrument_pairs: list[tuple[str, str]], task: str
) -> list[RunSpec]:
    target = T_CANON_REG if task == "regression" else T_CANON_CLF
    aug_levels = (A_CANON, "A4_batch_med")
    runs: list[RunSpec] = []
    sorted_pairs = sorted(cross_instrument_pairs)
    for (a, b) in sorted_pairs:
        for (d_train, d_test) in ((a, b), (b, a)):
            for k in range(5):
                for a_lvl in aug_levels:
                    runs.append(RunSpec(
                        block="B7", dataset_alias=d_train,
                        split_scheme=S_CANON, seed=k,
                        pp_chain=P_CANON, augmentation=a_lvl, outlier_filter=F_CANON,
                        target_processing=target,
                        cross_instrument_direction=f"{d_train}->{d_test}",
                    ))
    return runs


def generate_canonical_grid(
    selected_datasets: list[str],
    fast12_datasets: list[str],
    audit20_datasets: list[str],
    scaling_datasets: list[str],
    cross_instrument_pairs: list[tuple[str, str]],
    task: Literal["regression", "classification"],
) -> list[RunSpec]:
    runs: list[RunSpec] = []
    runs += generate_B1(selected_datasets, task)
    runs += generate_B2(audit20_datasets, task)
    runs += generate_B3(fast12_datasets, task)
    runs += generate_B4(fast12_datasets, task)
    runs += generate_B5(fast12_datasets, task)
    runs += generate_B6(scaling_datasets, task)
    runs += generate_B7(cross_instrument_pairs, task)
    return runs
```

## 8. Sortie JSON canonique

```json
{
  "schema_version": "1.0",
  "grid_version": "v0.1b",
  "task": "regression",
  "blocks": {
    "B1": {"runs": [...], "fixed_factors": {"pp_chain": "P5_snv_sg_standardscaler", "augmentation": "A0_none", "outlier_filter": "F0_none", "target_processing": "T1_minmax", "n_samples": "all"}},
    "B2": {...},
    ...
  },
  "total_runs_per_submission": 3660,
  "anchor_pp_chains": ["P5_snv_sg_standardscaler", "P12_snv_sg_minmax"],
  "identifiable_effects": {...},
  "non_identifiable_effects": [...],
  "aliasing_warnings": [...],
  "reporting_standards": {
    "B1": "Friedman + Nemenyi post-hoc (Demšar 2006), Holm pour multiplicité",
    "B2": "Kendall W concordance multi-juge",
    ...
  },
  "stitching_policy": "interdit en dataviz v0.1, autorisé en notebooks avec diagnostic d'interaction obligatoire"
}
```

## 9. Tests d'acceptation de la grille

Avant freeze v0.1b :

1. **Déterminisme** : `generate_canonical_grid(...)` produit la même liste bit-à-bit sur deux appels avec mêmes entrées.
2. **Ordre canonique** : la sortie est ordonnée alphabétiquement par dataset_alias → schéma → seed → reste. Test : pour deux ordres d'entrée de `selected_datasets` aléatoires, la sortie est identique.
3. **Couverture des facteurs** : pour chaque facteur, la grille produit au moins un run où ce facteur prend chacun de ses niveaux. Test exhaustif.
4. **Cardinalité** : `len(grid)` est dans la fourchette annoncée (3 620-3 660 selon `n_pairs` du B7).
5. **Pas de doublons** : `len(set(grid)) == len(grid)`.
6. **B7 direction** : tous les `B7` runs ont `cross_instrument_direction` non-null ; tous les autres l'ont null.
7. **Effets identifiables** : pas un test de rang algébrique (impossible sans codage des facteurs) mais une *table de vérification* : pour chaque effet listé `identifiable_effects`, vérifier qu'il existe au moins 2 niveaux distincts du facteur correspondant dans la grille.

## 10. Points ouverts à arbitrer en v0.2

- **Power calculation** : 10 seeds en B1 calibré empiriquement.
- **Stratégie de stitching contrôlé** dans la dataviz v0.2.
- **B6/B7 expansion** : 30 seeds, 12 datasets B6 ; 10 paires B7.
- **`P_canon` validation** : confirmation après v0.1 que la chaîne ancre est raisonnable sur ≥ 80 % des datasets `selected`.
- **`Random_5fold` redondance** : à confirmer s'il est essentiel à côté de `KBinsStratified_5fold` / `Stratified_5fold`.
- **Reporting unifié** : table standard `(block, effect, test, p, effect_size, IC)` pour chaque soumission, à figer.

## 11. Livrables

- Code du générateur dans `nirs4all_arena.protocol.grid_v01b` (à créer).
- Tests dans `tests/benchmark/test_grid_v01b.py` (déterminisme, ordre canonique, cardinalité, couverture).
- Fichier `grid_v0.1b.json` publié avec la release.
- Document `aliasing_warnings_v0.1b.json` listant les effets non-identifiables.
- Document `reporting_standards_v0.1b.md` : tests statistiques et corrections par bloc.

## 12. Références

- Demšar J. (2006) "Statistical comparisons of classifiers over multiple data sets." *Journal of Machine Learning Research* 7:1-30.
- Nadeau C., Bengio Y. (2003) "Inference for the generalization error." *Machine Learning* 52(3):239-281.
- Box G.E.P., Hunter J.S., Hunter W.G. (2005) *Statistics for Experimenters* 2nd ed., Wiley.
- Montgomery D.C. (2017) *Design and Analysis of Experiments* 9th ed., Wiley.
