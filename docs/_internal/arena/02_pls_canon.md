# PLS-canon et Ridge-canon — définition opérationnelle des baselines

> Document de spécification. Les baselines sont **maintenues centralement** par le runtime, soumises à chaque bump majeur, et servent de dénominateurs pour `score_ratio_vs_pls_canon` (cf §6, §8.4 du manifeste). Ce document fixe leurs implémentations.

| Champ | Valeur |
|---|---|
| Référence amont | [systematic_benchmarking_protocol.md](../systematic_benchmarking_protocol.md) §5.3 (`honest brokers`), §10.1 |
| Version | v0.1b (intègre revue Codex initiale) |
| Statut | À soumettre à seconde revue avant freeze |
| Phase roadmap | rédigé en **Phase 0a** (Conception) ; implémenté et calibré en **Phase 2** (Baselines + runtime alpha) ; dénominateur de toutes les phases suivantes. Voir [07_nirs4all_arena_roadmap.md](07_nirs4all_arena_roadmap.md). |

## 0. Changelog v0.1 → v0.1b (post-revue Codex)

1. **Bug double-scaling corrigé** : `PLSRegression(scale=True)` par défaut centre et réduit X et y ; combiné avec un `StandardScaler` amont, X était scalé deux fois. Correction : `PLSRegression(scale=False)`, le `StandardScaler` du pipeline est l'unique source de centrage/réduction de X. Le centrage de y est explicité (cf §2.1).
2. **Cohérence inner-CV PLS / Ridge** : Ridge utilisait LOO/GCV (`cv=None`), PLS utilisait `KFold(5)`. Mismatch confondant le dénominateur. Correction : Ridge-canon utilise aussi `KFold(5, shuffle=True, random_state=seed)`.
3. **Spec PLSDA alignée avec le wrapper réel** : la spec mentionnait `LabelBinarizer`, le wrapper `nirs4all.operators.models.PLSDA` utilise en réalité `LabelEncoder` (binaire) / `OneHotEncoder` (multiclasse), sans `class_weight`, seuil 0.5 / argmax, probabilités brutes sans calibration. Spec mise en cohérence et `proba_unavailable` documenté.
4. **RF-canon régression** : `max_features="sqrt"` remplacé par `max_features=1.0/3.0` — défaut courant en régression (et plus aligné avec les recommandations RF-regression-on-spectra) ; ajout d'un RF-canon classification manquant.
5. **n_lv_max** : précision sur l'edge case `n_train < 5` ; le runtime journalise la borne effective.
6. **Métriques inner-CV** : explicitation du fait que l'inner-CV est *dataset-local* (donc RMSE brut acceptable) tandis que l'arène compare des *ratios* cross-dataset ; reporting secondaire de `nRMSE = RMSE / σ(y_train)` et `RPD` exigé.
7. **Citations ajoutées** : Savitzky & Golay 1964, Barnes et al. 1989, Geladi & Kowalski 1986, Hoerl & Kennard 1970, Breiman 2001.
8. **Version pinning** : exigence explicite que chaque baseline déclare `sklearn_version`, `nirs4all_version`, BLAS backend dans l'EnvCard du run.

## 1. Pourquoi deux baselines

- **PLS-canon** : référence *chimiométrique* native. Toute conclusion "X bat PLS" en NIRS doit être mesurée contre une implémentation de PLS qui ne soit pas tunée à l'avantage de X (Geladi & Kowalski 1986, *Anal. Chim. Acta*). C'est la baseline historique du domaine.
- **Ridge-canon** : référence *linéaire universelle* (Hoerl & Kennard 1970, *Technometrics*). Sur certains datasets (n_features ≫ n_samples, faible structure latente), Ridge bat PLS naturellement. Avoir les deux comme baselines évite que `score_ratio_vs_pls_canon` soit trompeur quand le dataset est mal adapté à la projection PLS.

Les deux baselines sont publiées avec leurs scores sur chaque dataset, et la métrique principale rapportée pour un challenger est `score_ratio_vs_best_canon = score / min(pls_canon_score, ridge_canon_score)` *en plus* du ratio vs PLS seul.

## 2. PLS-canon — régression

### 2.1 Définition

```python
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate


def build_pls_canon_regression(*, n_lv_max: int = 30, inner_cv_seed: int = 0) -> GridSearchCV:
    """PLS-canon de régression — honest broker du protocole arène.

    Pipeline:
        SNV (par-spectre, sans paramètre de population)
        → SavitzkyGolay (window=11, derivative=1, polyorder=2)
        → StandardScaler (centrage + réduction de X, fit-on-train-only)
        → PLSRegression(n_components=optimal_cv, scale=False)

    Notes:
        - PLSRegression(scale=False) : on désactive le scaling interne de
          sklearn pour éviter le double-scaling avec le StandardScaler amont.
        - Le centrage / réduction de y est laissé à PLSRegression(scale=False)
          c'est-à-dire NON appliqué ; les `y` sont passés bruts. Le runtime
          applique en amont la `target_processing` du run atomique (e.g.
          `T1_minmax`), qui produit un y déjà normalisé selon la convention
          du bloc factoriel.

    Sélection d'hyperparamètres:
        n_components ∈ {1, 2, ..., n_lv_max_effective}
        avec n_lv_max_effective = min(n_lv_max, n_features, max(1, n_train // 5))
        sélectionné par neg_root_mean_squared_error (inner CV).

    L'inner-CV est dataset-local : son scoring brut (RMSE) est utilisé pour
    *choisir* n_components ; il n'est pas comparé inter-datasets. La métrique
    arène est `score_ratio_vs_pls_canon`, qui est invariante à l'échelle de y.

    Edge cases (le runtime journalise la décision) :
        - n_train < 5 : n_lv_max_effective = 1 ; pas d'inner-CV, refit direct.
        - n_lv_max_effective == 1 : pas de GridSearch, refit direct.
        - n_features == 1 : impossible en NIRS — rejet.
    """
    pipe = Pipeline([
        ("snv", StandardNormalVariate()),
        ("sg", SavitzkyGolay(window=11, derivative=1, polyorder=2)),
        ("scale", StandardScaler()),
        ("pls", PLSRegression(n_components=10, scale=False)),
    ])
    param_grid = {"pls__n_components": list(range(1, n_lv_max + 1))}
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=inner_cv_seed)
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=1,                  # parallélisme au niveau du runtime, pas du modèle
        verbose=0,
        return_train_score=False,
    )
```

### 2.2 Règles d'invocation par le runtime

1. À chaque run atomique, le runtime instancie `build_pls_canon_regression(n_lv_max=30, inner_cv_seed=seed)` où `seed` est la seed externe du run atomique.
2. Le runtime *calcule* `n_lv_max_effective = min(30, n_features, max(1, n_train // 5))` et tronque la grille avant l'appel `.fit()`. Cette borne effective est loguée dans le résultat atomique (`notes["n_lv_max_effective"]`).
3. La grille d'hyperparamètres est **fixe** : `n_components ∈ {1, …, n_lv_max_effective}`. Pas de step. Tie-breaking dans `GridSearchCV` : sklearn choisit le premier ; on documente que pour le canon, le tie-breaking favorise le `n_components` le plus *petit* (parcimonie).
4. Aucune autre option exposée (pretrained, finetune, custom loss).
5. **Version pinning** : le runtime persiste `sklearn.__version__` et `nirs4all.__version__` dans l'EnvCard du run. Toute drift est détectable.

### 2.3 Justification des choix d'hyperparamètres

| Choix | Valeur | Justification | Citation |
|---|---|---|---|
| Préprocessing | `SNV → SG(w=11, d=1, p=2)` | Convention chimiométrique la plus citée. Hypothèse, pas loi (cf §3.1.2 du manifeste). | Barnes et al. 1989 ; Savitzky & Golay 1964 |
| `window=11` | impair | Compromis bruit/résolution standard pour spectres VIS-NIR à 100-200 features ; sensibilité à `w` mesurée en bloc B3. | Rinnan et al. 2009 *TrAC* (unverified) |
| `derivative=1` | dérivée première | la plus utilisée ; corrige le décalage baseline résiduel. | idem |
| `polyorder=2` | quadratique | minimum pour `d=1` ; standard. | Savitzky & Golay 1964 |
| `StandardScaler` après SG | centrage + réduction de X uniquement | requis par PLS pour stabilité numérique ; `PLSRegression(scale=False)` évite le doublon. | — |
| `PLSRegression(scale=False)` | **scaling interne désactivé** | évite le double-scaling de X et garde le contrôle explicite. | sklearn docs |
| `n_lv_max=30` | borne haute paramétrable | au-delà, risque de surapprentissage domine sur datasets NIRS (n_samples < 500). | Geladi & Kowalski 1986 |
| Inner CV K=5 | 5 folds | compromis biais-variance ; cohérent avec Ridge-canon (§4.1). | Hastie et al. 2009 |
| `shuffle=True` | mélangé | nécessaire pour la K-Fold interne ; seed propage. | — |
| Scoring | `neg_root_mean_squared_error` | aligne CV interne avec métrique reportée. Dataset-local. | — |
| Tie-breaking | n_components le plus petit | parcimonie ; documentation explicite. | — |

### 2.4 Reporting de métriques

- **Métrique principale (arène)** : `score_ratio_vs_pls_canon = score_M / score_PLSCANON` sur le même split, calculé par dataset.
- **Métriques secondaires (obligatoires)** :
  - `RMSE_brut` (`score_value`).
  - `nRMSE = RMSE / σ(y_train)` (normalisation par l'écart-type du train — invariant à l'échelle de y, comparable inter-datasets sous hypothèse).
  - `RPD = σ(y_test) / RMSE` (Residual Prediction Deviation — convention chimiométrique).
  - `MAE`, `R²`, `bias` (`mean(y_pred - y_true)`).

Ces secondaires sont *toujours* persistés dans le résultat atomique ; les vues dataviz peuvent les filtrer mais le runtime les calcule systématiquement.

### 2.5 Audit fit-on-train-only

Le runtime audite avant exécution :
- `SNV` est par-spectre, *sans* paramètre de population — trivialement safe.
- `SavitzkyGolay` est un filtre déterministe sans paramètre estimé — safe.
- `StandardScaler` est fit *uniquement* sur train (sklearn standard) — safe par construction de `Pipeline`.
- `PLSRegression(scale=False)` fit sur train — safe.
- `GridSearchCV` fait CV sur train *seulement* (jamais sur test) — safe par construction.

Cet audit est automatisable (audit de pipeline JSON) et constitue un test d'acceptation du runtime.

### 2.6 Faisabilité fold

Avant `.fit()`, le runtime vérifie :
- `n_train >= 5 * inner_cv.n_splits = 25` pour que K-Fold soit raisonnable. Si `n_train < 25`, le runtime réduit `n_splits` à `max(2, n_train // 5)` et logue.
- Si `n_train < 2`, refus du run avec `failed_dispatch`.

## 3. PLSDA-canon — classification

### 3.1 Définition (alignée avec `nirs4all.operators.models.PLSDA`)

```python
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nirs4all.operators.models import PLSDA
from nirs4all.operators.transforms import SavitzkyGolay, StandardNormalVariate


def build_plsda_canon_classification(*, n_lv_max: int = 30, inner_cv_seed: int = 0) -> GridSearchCV:
    """PLSDA-canon de classification — honest broker.

    Le wrapper PLSDA de nirs4all encode les classes via:
        - LabelEncoder (binaire, 0/1 → seuil 0.5)
        - OneHotEncoder (multiclasse, K colonnes → argmax sur K)
    Aucun class_weight, aucune calibration de probabilités.
    Conséquence : predict_proba retourne les sorties brutes de PLSRegression
    (non bornées dans [0,1]), donc `proba_unavailable = True` dans le schéma
    de résultat ; les métriques probabilistiques (log-loss, AUC bien calibrée)
    ne sont pas reportées pour ce canon.
    """
    pipe = Pipeline([
        ("snv", StandardNormalVariate()),
        ("sg", SavitzkyGolay(window=11, derivative=1, polyorder=2)),
        ("scale", StandardScaler()),
        ("plsda", PLSDA(n_components=10)),
    ])
    param_grid = {"plsda__n_components": list(range(1, n_lv_max + 1))}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_cv_seed)
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        verbose=0,
    )
```

### 3.2 Reporting secondaire (classification)

- **Métrique principale (inner CV)** : `balanced_accuracy` (invariante au déséquilibre).
- **Métriques secondaires (obligatoires)** :
  - `macro_F1` : moyenne non pondérée des F1 par classe — sensible au déséquilibre.
  - `weighted_F1` : pondéré par support.
  - `per_class_recall` (vecteur) — révèle l'instabilité par classe que `balanced_accuracy` lisse.
  - `kappa` (Cohen) : accord ajusté pour le hasard.
  - `confusion_matrix` complète persistée.
- **Probabilités** : marqué `proba_unavailable = True` (le wrapper ne calibre pas) — pas de log-loss / AUC reportées pour ce canon.

### 3.3 Faisabilité fold (classification)

- `StratifiedKFold(n_splits=5)` exige `min(class_counts) >= 5`. Si une classe a moins de 5 samples, réduire à `n_splits = min(class_counts)`.
- Si `min(class_counts) < 2`, refus du run.

## 4. Ridge-canon — régression

### 4.1 Définition (cohérente avec PLS-canon : K-Fold 5)

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_canon_regression(*, inner_cv_seed: int = 0) -> GridSearchCV:
    """Ridge-canon de régression — second honest broker.

    Choix d'inner-CV : K-Fold 5, cohérent avec PLS-canon. (Note : la version
    précédente utilisait RidgeCV(cv=None) qui fait du LOO/GCV ; cette
    inconsistance avec PLS-canon créait un biais sur les ratios. Cohérence
    K=5 imposée à v0.1b.)
    """
    alphas = np.logspace(-4, 4, num=50)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    param_grid = {"ridge__alpha": alphas.tolist()}
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=inner_cv_seed)
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        verbose=0,
    )
```

### 4.2 Variant `ridge_canon_snv_sg`

```python
def build_ridge_canon_snv_sg_regression(*, inner_cv_seed: int = 0) -> GridSearchCV:
    """Variant avec préprocessing chimiométrique.

    NOTE: le `Δscore` entre ridge_canon et ridge_canon_snv_sg n'est PAS un
    estimateur propre de "PP utility for Ridge" car α est ré-sélectionné
    indépendamment dans les deux pipelines. Pour comparer proprement
    l'apport du PP, un *second* variant fixe α au α optimal du baseline raw
    serait nécessaire (`ridge_canon_snv_sg_fixed_alpha`) — reporté en v0.2.
    """
    alphas = np.logspace(-4, 4, num=50)
    pipe = Pipeline([
        ("snv", StandardNormalVariate()),
        ("sg", SavitzkyGolay(window=11, derivative=1, polyorder=2)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    param_grid = {"ridge__alpha": alphas.tolist()}
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=inner_cv_seed)
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        verbose=0,
    )
```

### 4.3 Choix-clés

- **K-Fold 5 (et non LOO/GCV)** : cohérence avec PLS-canon. LOO est plus rapide, mais le risque d'incohérence sur le dénominateur l'emporte sur le gain CPU.
- **Grille α log-spaced** : 50 points sur 9 ordres de magnitude.
- **Pas de SNV/SG dans le canon raw** : Ridge-canon répond à "que vaut un modèle linéaire universel sur spectres bruts ?". Le variant `_snv_sg` est *informatif*, pas un estimateur propre d'utilité du PP (cf §4.2).

## 5. Ridge-canon — classification

```python
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge_canon_classification(*, inner_cv_seed: int = 0) -> GridSearchCV:
    """Ridge-canon classification — cohérence K=5."""
    alphas = np.logspace(-4, 4, num=50)
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", RidgeClassifier(alpha=1.0)),
    ])
    param_grid = {"ridge__alpha": alphas.tolist()}
    inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=inner_cv_seed)
    return GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        refit=True,
        n_jobs=1,
        verbose=0,
    )
```

Comme PLSDA-canon, `RidgeClassifier` n'expose pas `predict_proba` ; `proba_unavailable = True`.

## 6. RF-canon

### 6.1 RF-canon régression

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_rf_canon_regression(*, seed: int = 0) -> Pipeline:
    """RF-canon régression (pas d'HPO interne — baseline rapide et stable).

    max_features=1/3 : défaut courant en régression RF (Breiman 2001 ;
    Hastie et al. 2009). Plus adapté que "sqrt" pour spectres NIRS
    fortement corrélés.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=500,
            max_features=1.0 / 3.0,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=1,
        )),
    ])
```

### 6.2 RF-canon classification

```python
from sklearn.ensemble import RandomForestClassifier


def build_rf_canon_classification(*, seed: int = 0) -> Pipeline:
    """RF-canon classification — max_features='sqrt' classique."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )),
    ])
```

### 6.3 Justification

- **`max_features=1/3` en régression** : recommandation classique pour RF-regression sur features corrélées (Breiman 2001 *Machine Learning* ; Hastie et al. 2009 §15.3.4) ; "sqrt" donne ≈ √200 = 14 features par split sur un spectre de 200 longueurs d'onde, ce qui est très peu et favorise les arbres trop décorrélés.
- **`max_features="sqrt"` en classification** : recommandation classique pour RF-classification (Breiman 2001).
- **Pas d'HPO** : RF-canon est délibérément *non* tuné. L'écart `score(rf_tuned_par_contributeur) - score(rf_canon)` est mesuré par les soumissions.
- **`class_weight="balanced"`** en classification : assure que `balanced_accuracy` n'est pas trompée par le déséquilibre.

## 7. CNN-naïf-canon — reporté en v0.2

V0.1 est *explicitement positionné comme un benchmark de baselines classiques et linéaires*. Le rapport méthodologique v0.1 doit énoncer :

> *"Aucun CNN-canon n'est inclus dans la grille v0.1. Les ratios reportés pour les challengers à base de réseaux profonds sont calculés vs PLS-canon et Ridge-canon, mais le lecteur doit garder à l'esprit qu'aucun NN baseline n'est maintenu centralement. Une baseline CNN-canon est planifiée pour v0.2 après stabilisation du protocole."*

Cette transparence est non-négociable : sans elle, l'absence de CNN-canon pourrait être interprétée comme un avantage indu pour les CNN soumis.

## 8. Politique de maintenance

### 8.1 Versioning

- Chaque baseline porte un `canon_version` (semver) lié à son commit Git.
- Toute modification (changement de window SG, ajout d'un step) est un bump *minor* au minimum.
- Toute modification produisant un écart > 1e-9 sur les scores du sanity test (cf §9) est un bump *major* avec re-soumission obligatoire des modèles `selected`.

### 8.2 Reproductibilité

- Le runtime exécute les baselines à chaque bump majeur, sur tous les `selected`. Les scores primaires restent dans les workspaces `nirs4all` (`store.sqlite`) ; `arena.sqlite` stocke seulement le lien baseline/version/run_spec nécessaire au calcul des ratios.
- Ces scores **sont le dénominateur** des ratios reportés ; si la baseline est modifiée, tous les ratios changent. Le bump majeur garantit que l'ancien jeu de ratios reste accessible (tag `v(N)-frozen`).
- **Version pinning obligatoire** : chaque run baseline persiste `sklearn.__version__`, `numpy.__version__`, `scipy.__version__`, `nirs4all.__version__`, BLAS backend, et n_threads dans l'EnvCard. Sans ces champs, le run est invalide.

### 8.3 Ownership

- Baselines co-ownées par le comité éditorial (à constituer). Pas d'auteur singulier.
- Proposition de modification = PR public + revue éditoriale + démonstration que l'écart vs l'ancienne baseline améliore la "vraie performance NIRS" et n'avantage pas un challenger spécifique.

## 9. Sanity test

```python
def test_pls_canon_on_synthetic_data():
    import nirs4all
    dataset = nirs4all.generate.regression(n_samples=300, n_features=200, seed=0)
    X_train, X_test, y_train, y_test = split_kennard_stone_70_30(dataset, seed=0)

    pls_canon = build_pls_canon_regression(inner_cv_seed=0)
    pls_canon.fit(X_train, y_train)
    rmse = float(np.sqrt(np.mean((pls_canon.predict(X_test).ravel() - y_test) ** 2)))

    # Valeur attendue à figer empiriquement après calibration v0.1b
    # (rmse_expected, tol) sera persisté dans canon_sanity_v0.1b.yaml
    # une fois la suite des baselines exécutée pour la première fois.
    pass
```

Le runtime exécute ce test à chaque release et compare la sortie au `rmse_expected` ; tout écart > tolerance trigge un bump major automatique. Les valeurs `(rmse_expected, tol)` sont gelées à v0.1b après la première campagne et persistées dans `canon_sanity_v0.1b.yaml`.

## 10. Questions ouvertes (à arbitrer en v0.2)

- **Window du SG** : 11 est un choix médian. *Décision v0.1b* : un seul canon (w=11) ; l'effet du window est mesuré dans le bloc B3 du PP exploratoire, pas dans le canon.
- **`y` scaling dans PLS-canon** : actuellement, y est passé brut à PLSRegression(scale=False), et le `target_processing` du run atomique l'a déjà transformé via `T_canon = MinMax`. Confirmer que `target_processing` est toujours actif pour PLS-canon (oui : c'est un axe T canonique fixé à `T1_minmax`).
- **PLSDA probability calibration** : ajouter Platt scaling ou isotonic en v0.2 ? Pour l'instant `proba_unavailable`.
- **Ridge regularization path** : 50 α uniformes log-spaced — granularité fine. Si trop fine, signature de surinterprétation ; à valider sur fast12.
- **Comparaison ridge_canon vs ridge_canon_snv_sg** : pour mesurer proprement l'utilité du PP, ajouter `ridge_canon_snv_sg_fixed_alpha` (α gelé au α optimal du baseline raw) en v0.2.

## 11. Livrables

- Implémentations dans `nirs4all_arena.baselines` (à créer) : `pls_canon.py`, `ridge_canon.py`, `rf_canon.py`.
- Tests dans `tests/benchmark/test_baselines.py` :
  - Test de sanity sur synthétique (§9).
  - Test de drift (la valeur sanity est gelée).
  - Test d'audit fit-on-train-only.
  - Test de fold feasibility (n_train petit).
  - Test de version pinning (EnvCard non-vide).
- Pour chaque dataset `selected`, calcul et persistance des scores PLS/Ridge/RF-canon dans les workspaces, avec liens `arena.sqlite`, avant ouverture des soumissions externes.
- Fichier `canon_sanity_v0.1b.yaml` gelant les valeurs attendues du sanity test.

## 12. Références

- Barnes R.J., Dhanoa M.S., Lister S.J. (1989) "Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra." *Applied Spectroscopy* 43(5):772-777.
- Breiman L. (2001) "Random forests." *Machine Learning* 45(1):5-32.
- Geladi P., Kowalski B.R. (1986) "Partial least-squares regression: a tutorial." *Analytica Chimica Acta* 185:1-17.
- Hastie T., Tibshirani R., Friedman J. (2009) *The Elements of Statistical Learning* 2nd ed., Springer.
- Hoerl A.E., Kennard R.W. (1970) "Ridge regression: biased estimation for nonorthogonal problems." *Technometrics* 12(1):55-67.
- Savitzky A., Golay M.J.E. (1964) "Smoothing and differentiation of data by simplified least squares procedures." *Analytical Chemistry* 36(8):1627-1639.
- Rinnan Å., van den Berg F., Engelsen S.B. (2009) "Review of the most common pre-processing techniques for near-infrared spectra." *Trends in Analytical Chemistry* 28(10):1201-1222 (unverified).
