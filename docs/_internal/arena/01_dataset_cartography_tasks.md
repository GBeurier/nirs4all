# Cartographie des datasets — plan de tâches

> Document planificateur. **Ne pas exécuter** — l'utilisateur prépare une DB dédiée pour cette étape. Ce document fixe le périmètre, le schéma cible et les vérifications à faire pour que chaque dataset entre proprement dans l'arène.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §3.2, §4.2, §11.2, §13 |
| Version | v0.1 |
| Statut | Plan |
| Phase roadmap | rédigé en **Phase 0a** (Conception) ; exécuté en **Phase 1** (DB + cartographie). Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 1. Périmètre

Deux ensembles à couvrir :

- **`selected` (v0.1)** : reprend les deux sous-ensembles définis dans `bench/Subset_analysis/rethought_subsets.json` :
  - `fast12_transfer_core` (12 datasets) : gate d'itération rapide.
  - `audit20_transfer_core` (20 datasets) : audit second-niveau. Inclut les 12 de `fast12` ? Vérifier le chevauchement et expliciter ; selon le JSON courant, `MP_spxyG`, `TIC_spxy70`, `Biscuit_Sucrose_40_RandomSplit`, `Ccar_spxyG_block2deg`, `LUCAS_pH_Organic_1763_LiuRandomOrganic`, `All_manure_K2O_SPXY_strat_Manure_type` apparaissent dans les deux.
  - Réunion `selected_v0.1` = `fast12 ∪ audit20` ≈ 26 datasets uniques (à confirmer après dédoublonnage).
- **`all` (v0.1)** : toutes les paires `(dataset_alias, task)` exécutables depuis `bench/tabpfn_paper/data/` :
  - `regression/` : 29 familles (ALPINE, AMYLOSE, BEEFMARBLING, BEER, BERRY, BISCUIT, CASSAVA, COLZA, CORN, DIESEL, DarkResp, ECOSIS_LeafTraits, FUSARIUM, GRAPEVINES, GRAPEVINE_LeafTraits, IncombustibleMaterial, LUCAS, MALARIA, MANURE21, MILK, PEACH, PHOSPHORUS, PLUMS, QUARTZ, RICE_Redox, SOIL_ESDAC_19969, TABLET, WOOD_density, WOOD_sustain).
  - `classification/` : 11 familles (ARABIDOPSIS_CEFE, BEEF_Impurity, COFFEE_orig, COFFEE_sp, Cassava, FUSARIUM, FruitPuree, MALARIA, MILK, PISTACIA, Wood_Sustainability).
  - Chaque famille peut contenir plusieurs *dataset_alias* (différentes cibles, différents splits, différents instruments — cf master CSV historique pour la nomenclature).
- **Source de vérité métadonnées** : `bench/tabpfn_paper/data/DatabaseDetail.xlsx` (à exploiter ; format à confirmer).

## 2. Schéma cible : `DatasetCard`

```python
@dataclass(frozen=True)
class DatasetCard:
    # Identification
    name: str                       # dataset_alias (e.g. "All_manure_K2O_SPXY_strat_Manure_type")
    family: str                     # parent folder (e.g. "MANURE21")
    task: Literal["regression", "classification"]
    dataset_fingerprint: str        # blake2b canonicalisé (cf §4.3.1 du manifeste)

    # Structure
    n_samples: int
    n_features: int
    spectral_range_nm: tuple[float, float] | None
    spectral_resolution_nm: float | None
    has_gaps: bool
    signal_type: Literal["absorbance", "reflectance", "log_reflectance", "unknown"]

    # Cible
    target_kind: Literal["regression_continuous", "classification_binary", "classification_multiclass", "multi_target_regression"]
    target_name: str                # e.g. "K2O", "manure_type"
    target_unit: str | None         # e.g. "% w/w"
    y_mean: float | None
    y_std: float | None
    y_range: tuple[float, float] | None
    y_complexity: dict              # {"cv": ..., "kurtosis": ..., "n_modes_silverman": ...}
    imbalance_ratio: float | None   # classification only
    multi_target_dim: int           # 1 for univariate

    # Profil "qualité de signal"
    autocorr_lambda: float          # corr moyenne wavelengths adjacentes
    snr_estimate: float
    snr_estimate_method: Literal["repetitions", "pca_residual", "unknown"]
    snr_estimate_uncertainty: float

    # Contexte expérimental
    instrument: str | None          # capteur si connu
    n_repetitions_avg: float | None # mesures répétées par sample
    domain_tag: str                 # e.g. "agro", "pharma", "geo"
    cross_instrument_pair: str | None  # dataset_alias du partenaire si multi-capteur

    # Splits canoniques
    available_splits: list[str]     # {"KS_70_30_seed0", "SPXY_70_30_seed0", ...}

    # Licence et citation
    license: str
    license_url: str | None
    citation: str                   # BibTeX ou CFF
    source_url: str | None
    date_added: str                 # ISO-8601

    # Statut éditorial
    selected: bool                  # in selected_v0.1
    selected_subset: list[str]      # ["fast12_transfer_core", "audit20_transfer_core"]
    status: Literal["active", "under_review", "retired"]
    notes: str
```

## 3. Tâches (TODO pour l'opérateur de la DB)

### T1 — Inventaire et dédoublonnage

- T1.1 Parser le contenu de `bench/tabpfn_paper/data/regression/` et `classification/`. Lister les sous-dossiers et fichiers.
- T1.2 Identifier la convention de nommage des `dataset_alias` (un dossier = combien d'aliases ?). Confronter aux dataset_aliases référencés dans `bench/Subset_analysis/rethought_subsets.json` (le côté `selected`).
- T1.3 Construire un *crosswalk* `family/folder/file → dataset_alias`. Persister en `crosswalk_v0.1.csv`.
- T1.4 Détecter les *dataset_alias* manquants : présents dans `selected` mais absents physiquement (ou inversement). Établir liste d'arbitrage.

### T2 — Calcul des champs de structure

- T2.1 Pour chaque `dataset_alias` : charger X, y, métadonnées via `nirs4all.data.loaders` (le format précis dépend du dataset — CSV, Parquet, NumPy). En cas d'ambiguïté, documenter et choisir un loader canonique.
- T2.2 Calculer `n_samples`, `n_features`, `spectral_range_nm`, `spectral_resolution_nm`, `has_gaps`. Vérifier qu'une longueur d'onde est explicitement définie (sinon, marquer `unknown` et flag à audit).
- T2.3 Inférer `signal_type` via `nirs4all.data.signal_type.SignalTypeDetector`. Loguer la confiance ; si < 0.8, demander confirmation humaine.

### T3 — Profil de la cible

- T3.1 `target_kind`, `target_name`, `target_unit` (à extraire des métadonnées ou nommage). En classification, calculer `imbalance_ratio = n_minority / n_majority`.
- T3.2 `y_mean`, `y_std`, `y_range` ; en régression seulement.
- T3.3 `y_complexity` : `{"cv": std/|mean|, "kurtosis": scipy.stats.kurtosis, "n_modes": kde_silverman_n_modes}`.

### T4 — Profil "qualité de signal"

- T4.1 `autocorr_lambda` : moyenne de `corr(X[:, i], X[:, i+1])` sur tout l'axe spectral.
- T4.2 `snr_estimate` :
  - **Si répétitions identifiables** (groupe `Sample_ID` ou équivalent) : `σ_noise = std intra-groupe moyennée` ; `SNR = mean(|X|) / σ_noise`. Méthode = `"repetitions"`. Incertitude = bootstrap sur groupes.
  - **Sinon** : décomposition PCA centrée sur X. Calculer la variance résiduelle après retrait des composantes expliquant 99 % de variance. `σ_noise² = mean residual variance`. Méthode = `"pca_residual"`. Incertitude = analyse de sensibilité sur seuil ∈ {0.95, 0.99, 0.999}.
  - Sauvegarder les deux estimations si répétitions disponibles + comparaison.

### T5 — Contexte expérimental

- T5.1 `instrument` : depuis `DatabaseDetail.xlsx` ou nommage.
- T5.2 `n_repetitions_avg` : depuis le mapping `Sample_ID → rows`.
- T5.3 `domain_tag` : taxonomie à fixer ; proposition initiale : `{agro_food, agro_field, pharma, biomedical, materials, environment, fuels, beverages}`.
- T5.4 `cross_instrument_pair` : pour les datasets ayant un jumeau multi-capteur (e.g. `An_spxyG70_30_byCultivar_NeoSpectra` ↔ `An_spxyG70_30_byCultivar_MicroNIR_NeoSpectra`), lier explicitement.

### T6 — Splits canoniques

- T6.1 Pour chaque dataset, **pré-calculer les masques de split** pour les 8 schémas de v0 (KS_70_30, SPXY_70_30, Random_70_30, KS_5fold, SPXY_5fold, Random_5fold, et selon la tâche KBinsStratified_5fold ou Stratified_5fold).
- T6.2 Pour chacun, sur seeds ∈ {0, 1, …, 9}. Cardinalité par dataset : 8 schémas × 10 seeds = 80 masques.
- T6.3 Persister sous `splits/<dataset_alias>/<scheme>_seed<n>.parquet` (colonnes `sample_id, fold_index ∈ {0..k-1}, role ∈ {train, test}`).
- T6.4 Calculer le `split_hash` de chaque masque (hash canonicalisé) ; le stocker dans la DatasetCard.

### T7 — Validation anti-leakage (§13 du manifeste)

- T7.1 Doublons spectraux exacts : hash de chaque ligne X ; flag si > 1 % de duplicates.
- T7.2 Quasi-doublons : `cosine_similarity` ; signaler paires avec `corr > 0.999`. Suspend admission à `selected` si > 5 cas.
- T7.3 Chevauchement inter-dataset : pour chaque paire de datasets admis, intersection des hashes ; flag si non-vide.
- T7.4 Métadonnées encodant y : pour chaque colonne de métadonnées catégorielle ou numérique, calculer `mutual_info(y, meta_col)` ou `corr(y, meta_col)` ; flag si > 0.95.
- T7.5 Groupes de répétitions respectés par splits : pour chaque masque, vérifier qu'un même `Sample_ID` n'est pas à la fois en train et en test (sauf si schéma de split prévoit explicitement le mélange — non pour v0).
- T7.6 Plausibilité physique de y (plages, unités). Demande humaine si suspect.

### T8 — Licence et citation

- T8.1 Pour chaque dataset, identifier la licence. Sources : `DatabaseDetail.xlsx`, README originaux, contacts auteurs.
- T8.2 Construire la `CITATION.cff` correspondante.
- T8.3 Datasets sans licence claire ou propriétaires → exclus du pool `all` public ; conservés en `private` pour audits internes.

### T9 — Persistance dans la DB

- T9.1 Schéma SQL/Parquet : table `datasets` avec colonnes correspondant à `DatasetCard`. Index sur `dataset_fingerprint`, `family`, `task`, `selected`.
- T9.2 Tables auxiliaires :
  - `splits(dataset_fingerprint, scheme, seed, mask_path, mask_hash)`
  - `cross_instrument_pairs(dataset_a, dataset_b, instrument_a, instrument_b)`
  - `leakage_findings(dataset_fingerprint, finding_type, severity, description, resolved)`
  - `citations(dataset_fingerprint, cff_path, doi, license)`

### T10 — Décisions d'arbitrage (humaines)

À documenter au fil de l'eau, idéalement dans un fichier `decisions_log.md` :

- Crosswalk ambigus (T1.4).
- Signal type incertain (T2.3 avec confiance < 0.8).
- Domain tag à attribuer (T5.3).
- Quasi-doublons à arbitrer (T7.2).
- Datasets propriétaires (T8.3).

## 4. Cardinalité estimée

| Item | Cardinalité |
|---|---:|
| Familles régression | 29 |
| Familles classification | 11 |
| `dataset_alias` totaux estimés | ≈ 60-100 (à confirmer après T1) |
| Datasets `selected` (fast12 ∪ audit20) | ≈ 26 |
| Masques de splits par dataset | 80 |
| Masques de splits totaux | ≈ 4 800 - 8 000 |

## 5. Points ouverts (à arbitrer avec le comité éditorial v0.2)

- **Politique pour les datasets propriétaires** : exclus du `all` public, conservés en `private` ? Ou rejetés purement ?
- **Politique pour les splits historiques sous-spécifiés** : certains dataset_alias encodent un split (`*_KS`, `*_70_30`, `*_YbaseSplit`) qui devient redondant avec les 8 schémas v0. Doit-on respecter le split historique comme "canonical" pour ce dataset, ou écraser ? Recommandation v0.1 : **écraser systématiquement** par les 8 schémas standard pour assurer la comparabilité, et conserver le split historique uniquement comme métadonnée informative.
- **Multi-target** : `bench/tabpfn_paper/data/PHOSPHORUS/` ou similaires peuvent contenir plusieurs cibles. Décision : un `dataset_alias` par cible, pas de fusion multi-target en v0.1.
- **Cross-instrument** : la formalisation des paires (T5.4) est-elle exhaustive à partir des nommages, ou faut-il consulter chaque détenteur de dataset ?

## 6. Critères de fin de T1..T10

La cartographie v0.1 est considérée terminée quand :

1. Toutes les `dataset_alias` de `selected_v0.1` ont une `DatasetCard` complète (sans champ `unknown` non documenté).
2. Tous les datasets de `bench/tabpfn_paper/data/` ont au moins une `DatasetCard` minimale (struct + cible + licence). Les profils qualité (T4) peuvent rester pour une seconde passe.
3. Les masques de splits canoniques sont calculés et persistés pour `selected`.
4. Le crosswalk `family → dataset_alias` est public et review.
5. La détection anti-leakage (T7) est exécutée sur `selected` ; les findings sont résolus ou tagués `accepted_with_caveat`.

## 7. Prérequis croisés avec le runtime

- Le runtime central (cf [04_runtime_sketch.md](04_runtime_sketch.md)) attend que la DB datasets soit lisible par `nirs4all.benchmark.datasets.load_registry()` — interface à figer avant T9.
- Le PLS-canon (cf [02_pls_canon.md](02_pls_canon.md)) doit pouvoir s'exécuter sur tout dataset cartographié *sans paramétrage manuel*. Si un dataset requiert un loader custom, le documenter dans `DatasetCard.notes`.

## 8. Livrables attendus à la fin de cette étape

- `datasets_registry_v0.1.duckdb` (ou équivalent) — base structurée.
- `splits/<dataset_alias>/<scheme>_seed<n>.parquet` — masques.
- `crosswalk_v0.1.csv` — mapping famille ↔ alias.
- `decisions_log.md` — historique des arbitrages.
- `leakage_findings_v0.1.csv` — résultats de T7.
- Rapport d'inventaire `cartography_v0.1.md` — synthèse pour la communauté.
