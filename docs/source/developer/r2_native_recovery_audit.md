# Audit de récupération R2 — backend natif

Ce document fixe la méthode de récupération des travaux R2 historiques avant
tout cutover du moteur par défaut. Il est une trace de portage : ce n'est pas
une déclaration que R2 est prêt.

## Point de départ

L'audit a été réalisé depuis `main` au commit `25a4d652` (`feat(native): close
R2 API safety boundaries (#111)`). La référence produit reste le roadmap
racine `ROADMAP_BACKEND_NATIF_V1.md`, en particulier les lots `API-001` à
`API-005`, `DAG-001` et `HPO-001`.

Le défaut est encore `legacy`. Le chemin `dag-ml` est sélectionnable et
fail-closed par défaut ; un rollback legacy demande explicitement
`allow_legacy_fallback=True`. Cette frontière ne doit pas être inversée avant
que toutes les capacités déclarées R2 disposent d'une preuve native.

## Évidence de cycle intégrée après l'audit historique

Le lot #122 (merge `0f612509`) ajoute une preuve locale pour le sous-ensemble
Methods strict déjà déclaré : le résultat expose un témoin vivant, lié à la
façade `dag_ml.TrainingResult`, qui observe l'appel callback-free
`dag_ml.execute_methods_training`. Ce témoin est volontairement non
sérialisable et cesse d'être utilisable après `close()`/`detach()` ; l'outcome
signé et le package portable restent les seules preuves durables, et permettent
l'export Archive V2 après cette fermeture.

À l'instant historique du lot #122, la CI `methods-installed.yml` installait
les roues publiées `dag-ml==0.3.19` et `nirs4all-methods==1.0.13`, retirait
`N4M_LIB_PATH`, puis vérifiait le cycle réel claim → close → Archive V2. Cette
preuve renforce la forme `run(..., engine="native")` déjà supportée ; elle ne rend
pas R2 complet,
ne change pas le défaut `legacy` et ne justifie aucune bascule de moteur.

## Méthode

Pour chaque branche candidate, on utilise `git range-diff` contre son merge
base avec `main`, puis on classe chaque changement :

1. déjà absorbé ou remplacé ;
2. portable après adaptation aux contrats actuels ;
3. obsolète ou dangereux, à réimplémenter depuis l'API actuelle.

Un cherry-pick massif est interdit. Les branches R2 anciennes ont divergé des
contrats Archive, conformal, HPO, package et identité. Chaque reprise doit être
un lot petit, avec une preuve de parité ou un refus de capacité explicite.

## Résultats

| Source historique | État dans `main` | Décision |
| --- | --- | --- |
| `codex/r2-native-public-cycle` | Les sessions Methods, le run/predict avec identités, l'export et les refus inter-moteurs sont déjà intégrés via le train #66 et les durcissements ultérieurs. | Ne pas cherry-pick ; étendre les surfaces actuelles. |
| `codex/r2-native-public-api` | Ses correctifs de fallback explicite sont déjà présents, sous une forme plus stricte et instrumentée. | Ne pas cherry-pick. |
| `codex/r2-native-retrain` et `codex/r2-native-conformal` | Le refit sélectionné, les packages V3 et la calibration liée à l'identité ont évolué depuis leurs prototypes. | Reprendre uniquement les scénarios de test non couverts, sur les contrats V2/V3 actuels. |
| `refactor/W82-cutover-strict`, `W83-export-no-legacy`, `W98-full-parity-gate` | Ils contiennent des extensions de lowering multi-source/stacking et des gates de parité, mais divergent fortement et supposent d'anciens artefacts/workspaces. | Extraire les preuves par capacité ; ne pas fusionner la branche. |
| `rc/v1-full-refactor-python` | Branche historique de consolidation, utile comme inventaire de scénarios mais non comme base de code. | Utiliser pour compléter la matrice R2 et les tests différentiels. |

## Ordre de portage R2 côté bibliothèque

1. Définir la matrice de capacités publique : chaque forme de `run`, `predict`,
   `session`, `retrain`, `explain` et `generate` est native, plugin explicite
   ou refusée avant consommation de données.
2. Fermer les écarts de cycle produit autour du lowering déjà étendu :
   persistence/workspace natifs, cache fingerprinté, hydratation hors processus
   et export sans réentraînement legacy.
3. Ajouter les scénarios différentiels issus des branches historiques seulement
   quand ils exercent ces contrats actuels (variants, CV/OOF, multi-source,
   stacking, HPO, conformal et refit).
4. Bascule du défaut vers `dag-ml` seulement lorsque la matrice est verte et
   que le rollback legacy demeure opt-in, visible et mesuré.

Studio et Web ne font pas partie de ce lot : ils consommeront le contrat R2
stabilisé après la fermeture du cycle bibliothèque.
