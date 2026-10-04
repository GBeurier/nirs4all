# Fusion tardive avec observations partielles

Le pipeline public `by_source` apprend un modèle par source, puis un modèle de
fusion sur les prédictions OOF organisées par DAG-ML. Les sources peuvent conserver
leurs rangs d'origine : chaque branche contient son encodeur, par exemple
`TensorPCA` pour une image ou une série de taille fixe.

Deux politiques restent distinctes :

- `missing_source_policy="error"` exige toutes les sources pour chaque ligne ;
  `"zero_with_indicator"` ne prédit une branche que sur ses lignes présentes et
  ajoute un indicateur de présence à ses prédictions.
- `target_policy="complete"` exige des cibles complètes ; `"per_target"` apprend
  chaque cible de régression uniquement sur ses valeurs observées, avec un
  encodeur refitté dans cette même intersection. Les cibles ne sont pas imputées.

```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import nirs4all
from nirs4all.operators.models.multimodal import TensorPCA

# cohort est un MultimodalDataset de nirs4all-io : les sources portent leurs
# sample_ids et presence_mask ; les cibles portent target_names et target_mask.
pipeline = [
    GroupKFold(3),
    {"branch": {
        "by_source": True,
        "missing_source_policy": "zero_with_indicator",
        "target_policy": "per_target",
        "steps": {
            "nir": [StandardScaler(), Ridge(alpha=0.1)],
            "image": [TensorPCA(n_components=2), Ridge(alpha=0.2)],
        },
    }},
    {"merge": "predictions"},
    Ridge(alpha=0.5),
]
result = nirs4all.run(pipeline, cohort, engine="dag-ml", refit=True,
                     save_artifacts=True, random_state=19)
meta = result.runs[-1]
archive = meta.export("late_partial.n4a")
prediction = nirs4all.predict(archive, prediction_cohort, engine="dag-ml")
result.close()
```

Sélectionner le run du modèle final pour exporter le stack complet. Le meilleur
score global peut appartenir à une seule branche ; son `export()` `.n4a` est
refusé avant écriture, car cette composante ne porte pas le replay autonome des
sources absentes. L'export ne réentraîne pas et ne remplace pas ce modèle par un
autre implicitement.

Pour classifier, utiliser une cible unique complète et des classifieurs possédant
`predict_proba` dans toutes les branches et dans le modèle final. La fusion reçoit
les **K colonnes** du vocabulaire commun, dans le même ordre, et un indicateur par
source. Les lignes absentes ne produisent aucune distribution de classe ; les
zéros du tableau de fusion sont accompagnés de masques distincts. Toutes les
classes doivent être observées dans chaque véritable scope d'apprentissage.

L'archive Python conserve les ancres des composants réellement refittés : poids,
schéma IO complet, ordre des sources, politiques, vocabulaire et lignes utilisées
par cible. Son chargement vérifie séparément les octets joblib et ces ancres. Le
replay exécute uniquement la phase native `PREDICT`, sans FIT, CV ni HPO. Il s'agit
d'une archive Python de confiance, avec ses dépendances, et non d'un modèle
portable Core pour R/WASM.

La cohorte de prédiction doit déclarer les mêmes sources, dans le même ordre et
avec les mêmes schémas. Une source complètement absente se déclare avec son
`presence_mask` entièrement faux et des buffers de la forme déclarée ; sa branche
n'est pas appelée. Aucun padding, remappage implicite de vocabulaire ou changement
silencieux d'ordre n'est effectué. Les groupes et les masques de cibles déterminent
les vrais scopes d'entraînement ; les données de Test restent une cohorte native
séparée.

Ce profil partiel travaille dans l'espace des cibles d'origine. Une étape
`y_transform` séparée est refusée avant apprentissage ; un
`TransformedTargetRegressor` interne à un modèle reste inclus dans l'état appris
capturé. La classification nécessite ses labels complets ; `per_target` concerne
la régression.

Les petites données utilisées par les tests de cette fonctionnalité sont des
fixtures déterministes. Aucun générateur multimodal de produit n'est ajouté.
