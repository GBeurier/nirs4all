# Analyse DuckDB: sauvegardes workspace et lecture des prédictions

Date: 2026-03-25

Contexte: dans le code actuel, le mot "sauvegarde" recouvre surtout la persistance du workspace de run (`store.duckdb` + `arrays/*.parquet` + `artifacts/*`), pas seulement l'export `.n4a`. Les problèmes remontés par les utilisateurs concernent surtout cette persistance workspace et la relecture des prédictions depuis ce workspace.

Version locale vérifiée pendant l'analyse:
- `duckdb==1.4.4`

## Fonctionnement high level des sauvegardes et lectures

### 1. Architecture de persistance actuelle

Le workspace persistant repose sur 3 couches distinctes:

- `nirs4all/pipeline/storage/workspace_store.py`
  - métadonnées relationnelles dans `store.duckdb`
  - runs, pipelines, chains, predictions, artifacts, logs, projects
- `nirs4all/pipeline/storage/array_store.py`
  - arrays de prédictions dans `arrays/<dataset>.parquet`
  - `y_true`, `y_pred`, `y_proba`, `sample_indices`, `weights`
- répertoire `artifacts/`
  - objets sérialisés content-addressed
  - modèles, preprocessings, etc.

Autrement dit:

- DuckDB porte l'index et les relations.
- Parquet porte les gros tableaux.
- Les binaires restent sur disque hors DB.

### 2. Ouverture du workspace

`WorkspaceStore` ouvre immédiatement une connexion DuckDB RW dans `__init__`, puis exécute tout de suite:

- création du répertoire workspace
- `duckdb.connect(...)`
- `create_schema(...)`
- migrations éventuelles
- création des dossiers `artifacts/` et `arrays/`

Références:

- `nirs4all/pipeline/storage/workspace_store.py:225-254`
- `nirs4all/pipeline/storage/store_schema.py:438-520`

Point important: aujourd'hui, une simple ouverture "lecture" du workspace ouvre en réalité une connexion DuckDB read-write avec création/migration potentielle du schéma.

### 3. Chemin de sauvegarde pendant `run()`

Le flux nominal est:

1. `nirs4all.api.run.run()` crée un `PipelineRunner`.
2. `PipelineRunner` crée un `PipelineOrchestrator`.
3. `PipelineOrchestrator.__init__()` ouvre un `WorkspaceStore`.
4. Pendant l'exécution:
   - `begin_run()`
   - `begin_pipeline()`
   - `save_chain()`
   - enregistrement des artifacts
   - flush des prédictions
   - mise à jour des résumés de chain
   - `complete_pipeline()`
   - `complete_run()`
5. `run()` retourne un `RunResult` qui garde un `runner` vivant, donc garde indirectement le `WorkspaceStore` vivant tant qu'on ne ferme pas explicitement le résultat.

Références:

- `nirs4all/api/run.py:395-504`
- `nirs4all/api/result.py:269-290`
- `nirs4all/pipeline/execution/executor.py:208-244`

### 4. Détail du flush des prédictions

Le flush des prédictions se fait en 2 phases:

- phase 1: métadonnées ligne par ligne dans DuckDB via `WorkspaceStore.save_prediction()`
- phase 2: arrays en batch dans Parquet via `ArrayStore.save_batch()`

Références:

- `nirs4all/data/predictions.py:679-787`
- `nirs4all/pipeline/storage/workspace_store.py:911-1030`
- `nirs4all/pipeline/storage/array_store.py:206-255`

Conséquence directe: la persistance n'est pas atomique entre DuckDB et Parquet.

### 5. Cas particulier: exécution parallèle

En mode parallèle:

- les workers `joblib/loky` n'écrivent pas dans DuckDB
- ils accumulent leurs prédictions et traces en mémoire
- le process principal reconstruit ensuite l'état DuckDB/Parquet

Cette reconstruction se fait "après coup" dans le process principal, avec une transaction DuckDB autour de:

- création du pipeline
- insertion des artifacts
- insertion des chains
- flush des prédictions
- completion du pipeline

Références:

- `nirs4all/pipeline/execution/orchestrator.py:307-479`

### 6. Lecture des prédictions

Il y a 3 modes principaux.

#### a. Exploration du workspace via `Predictions.from_workspace(...)`

Le code:

- ouvre un `WorkspaceStore`
- requête les lignes `predictions`
- recharge éventuellement les arrays depuis Parquet
- garde le store ouvert tant que `Predictions.close()` n'est pas appelé

Références:

- `nirs4all/data/predictions.py:267-285`
- `nirs4all/data/predictions.py:322-393`

#### b. Prédiction store-based via `predict(chain_id=...)`

Le code:

- ouvre un `WorkspaceStore`
- rejoue une `chain`
- ferme explicitement le store dans un `finally`

C'est aujourd'hui le chemin le plus sain côté durée de vie des connexions.

Référence:

- `nirs4all/api/predict.py`

#### c. Prédiction model-based / resolver-based

Le resolver utilise le workspace pour résoudre pipeline, chain et artifacts quand on part d'un `result.best`, d'un `prediction_id`, etc. Ici aussi, la disponibilité du workspace est critique.

Référence:

- `nirs4all/pipeline/resolver.py`

## Problèmes, défauts, possibilités évidentes d'améliorations

### 1. Le verrou principal vient de la durée de vie de la connexion, pas seulement des écritures

Le défaut principal n'est pas un "mauvais mutex Python". Le vrai problème est plus structurel:

- `WorkspaceStore` ouvre une connexion RW dès sa construction
- `nirs4all.run()` retourne un `RunResult` qui conserve le `runner`
- le `runner` conserve l'orchestrator
- l'orchestrator conserve le `WorkspaceStore`

Donc après un `run()` terminé, la connexion DuckDB reste ouverte tant que:

- le `RunResult` existe
- ou que `result.close()` n'est pas appelé
- ou que le process ne se termine pas

Même problème pour `Predictions.from_workspace(...)`: l'objet garde inutilement un `WorkspaceStore` ouvert après avoir déjà chargé les données en mémoire.

Conséquence pratique:

- un run fini peut encore rendre le workspace inaccessible à un autre process
- une simple session d'exploration des prédictions peut elle-même verrouiller le workspace

Code concerné:

- `nirs4all/api/run.py:498-503`
- `nirs4all/api/result.py:269-290`
- `nirs4all/data/predictions.py:277-279`
- `nirs4all/data/predictions.py:388-393`

### 2. Les erreurs de lock les plus fréquentes arrivent à la connexion, pas dans `_retry_on_lock`

Le code a bien un retry sur `duckdb.TransactionException`, mais:

- le lock inter-processus le plus fréquent arrive dès `duckdb.connect(...)`
- ce cas remonte comme `duckdb.IOException`
- ce chemin n'est pas couvert par `_retry_on_lock`

Donc le mécanisme de retry actuel ne protège pas le problème utilisateur principal.

Code concerné:

- `nirs4all/pipeline/storage/workspace_store.py:231`
- `nirs4all/pipeline/storage/workspace_store.py:276-320`

### 3. DuckDB n'est pas adapté à un workspace partagé multi-processus "toujours ouvert"

D'après la documentation DuckDB:

- un seul process peut lire/écrire
- plusieurs process peuvent lire seulement en `READ_ONLY`
- écrire depuis plusieurs process n'est pas supporté automatiquement
- le pattern recommandé si on insiste est: mutex inter-processus + ouvrir/fermer la DB quand la requête est finie

Références DuckDB officielles:

- https://duckdb.org/docs/stable/connect/concurrency
- https://duckdb.org/docs/stable/clients/python/dbapi

Ce point colle exactement au comportement local observé.

### 4. Vérifications locales reproductibles

Tests faits sur le code actuel, avec `duckdb==1.4.4`:

- Si un process A garde juste un `WorkspaceStore` ouvert en RW, un process B ne peut pas se connecter au même `store.duckdb`.
- Le même échec apparaît même si le process B tente une connexion DuckDB `read_only=True`.
- Si un process A fait `run(...)` puis garde le `RunResult` vivant, un process B qui essaie `Predictions.from_workspace(...)` échoue avec un lock.
- Deux process `read_only=True` ouvrent correctement le fichier seulement quand aucun process writer n'a de connexion RW ouverte.

Conclusion factuelle:

- le problème n'est pas théorique
- il n'est pas limité aux transactions longues
- il suffit qu'une connexion RW vive encore

### 5. La lecture des prédictions depuis le workspace est inefficace

`Predictions._populate_buffer_from_store()` charge les arrays ainsi:

- une ligne DuckDB
- puis `load_single(pred_id, dataset_name=...)`
- qui appelle `load_batch([pred_id], ...)`
- donc un scan Parquet par prédiction

Sur un gros workspace, c'est mécaniquement sous-optimal. On rescane le même fichier dataset plusieurs fois.

Code concerné:

- `nirs4all/data/predictions.py:332-342`
- `nirs4all/pipeline/storage/array_store.py:261-331`

Amélioration évidente:

- grouper les `prediction_id` par `dataset_name`
- charger les arrays en batch dataset par dataset

### 6. La persistance DuckDB + Parquet n'est pas atomique

Le flush des prédictions fait:

- insertion metadata DuckDB
- puis écriture Parquet

Risques:

- crash après insert DuckDB mais avant write Parquet
  - lignes metadata sans arrays
- crash après write Parquet mais avant commit/fin logique
  - arrays orphelins sans metadata
- en parallèle, la reconstruction store est enveloppée dans une transaction DuckDB, mais `ArrayStore.save_batch()` n'est évidemment pas transactionnel avec DuckDB

Code concerné:

- `nirs4all/data/predictions.py:716-787`
- `nirs4all/pipeline/execution/orchestrator.py:415-463`

Le design actuel offre donc une cohérence "best effort", pas une cohérence transactionnelle forte.

### 7. `ArrayStore.save_batch()` réécrit le fichier complet à chaque append

Le chemin actuel:

- lit tout le Parquet existant
- concatène avec le batch
- réécrit tout le fichier via temp + replace

C'est robuste à l'échelle d'un fichier unique, mais:

- le coût grossit avec la taille du dataset
- le temps de fenêtre critique augmente
- il n'y a aucun verrouillage inter-processus explicite sur `arrays/*.parquet` ni sur `_tombstones.json`

Code concerné:

- `nirs4all/pipeline/storage/array_store.py:242-249`

### 8. Le schéma DuckDB est initialisé/migré de façon non transactionnelle

`create_schema()` exécute:

- plusieurs DDL séparés
- puis des migrations `ALTER TABLE`
- puis recréation de vues

Sans enveloppe transactionnelle explicite.

Code concerné:

- `nirs4all/pipeline/storage/store_schema.py:438-520`

Conséquence observée localement:

- sur un workspace neuf, un `SIGKILL` pendant la première initialisation + écriture a laissé un `store.duckdb.wal` que `WorkspaceStore` n'a pas réussi à rejouer ensuite
- sur un workspace déjà initialisé proprement, le même test de crash sur une écriture simple a bien récupéré

Conclusion factuelle:

- le point fragile n'est pas seulement "DuckDB en général"
- c'est aussi le fait d'exécuter création/migration du schéma dans le chemin d'ouverture standard

### 9. Le pattern d'écriture est très OLTP / petites transactions

DuckDB documente explicitement que:

- il est optimisé pour des opérations bulk
- beaucoup de petites transactions ne sont pas son objectif principal

Or le code actuel fait beaucoup de petites opérations metadata:

- `begin_run`
- `begin_pipeline`
- `save_chain` x N
- `save_prediction` x N
- `update_chain_summary`
- `complete_pipeline`
- `complete_run`

Même si le process writer est unique, ce n'est pas le terrain idéal de DuckDB.

Référence DuckDB officielle:

- https://duckdb.org/docs/stable/connect/concurrency

## Proposition pour améliorer le mécanisme

### Recommandation principale

Avant la v1, je ne garderais pas DuckDB comme source de vérité "live" du workspace.

Je recommande:

- `SQLite WAL` pour les métadonnées opérationnelles
- conserver `Parquet` pour les arrays
- conserver les `artifacts/` content-addressed
- réserver DuckDB à l'analyse ad hoc ou à l'export analytique, pas au store partagé vivant

En pratique:

- SQLite devient l'index du workspace
- les arrays restent dans `arrays/*.parquet`
- les modèles restent dans `artifacts/`
- DuckDB devient optionnel, par exemple:
  - export analytique
  - snapshot local
  - requêtes lourdes ponctuelles

### Pourquoi SQLite WAL est plus cohérent ici

Le workload réel de `nirs4all` côté workspace est plutôt:

- petites écritures metadata
- beaucoup de filtres/lectures ciblées
- besoin de robustesse inter-processus
- besoin de relancer facilement après crash

Ce n'est pas un pur workload OLAP.

DuckDB est élégant pour l'analyse. Mais ici, comme store opérationnel interactif partagé entre:

- runs
- exploration des prédictions
- relances
- notebooks / scripts / outils adjacents

il est mal aligné avec le besoin.

SQLite WAL donne un meilleur compromis pour:

- un writer unique
- des readers concurrents
- des transactions courtes
- une ouverture/fermeture fréquente sans surprise

Et DuckDB lui-même recommande, pour le multi-processus, d'envisager MySQL/PostgreSQL/SQLite comme store transactionnel, puis de requêter ensuite avec DuckDB si besoin.

Référence DuckDB officielle:

- https://duckdb.org/docs/stable/connect/concurrency

### Architecture cible proposée

#### 1. Nouveau découpage

Source de vérité:

- `workspace/index.sqlite`
  - runs
  - pipelines
  - chains
  - predictions
  - artifacts
  - logs
  - projects

Données volumineuses:

- `workspace/arrays/<dataset>.parquet`
- `workspace/artifacts/<hash>.joblib`

Exports:

- `.n4a`
- `.yaml`
- `.parquet`
- éventuellement snapshot DuckDB analytique reconstruit à la demande

#### 2. Contrat de persistance

Écrire un pipeline ne doit pas faire vivre une connexion globale pendant toute la vie du résultat.

Le contrat devrait devenir:

- ouverture courte
- transaction courte
- commit
- fermeture immédiate

Un `RunResult` ne devrait garder que:

- `workspace_path`
- les prédictions en mémoire
- éventuellement les `chain_id` gagnants

mais pas un `runner` vivant porteur d'une connexion DB ouverte.

#### 3. Chargement des prédictions

`Predictions.from_workspace(...)` devrait par défaut:

- ouvrir l'index
- charger les rows metadata
- charger les arrays en batch
- fermer immédiatement l'index
- retourner un objet détaché, purement mémoire

Si on veut des opérations de maintenance (`remove_run`, `compact`, etc.), il faut alors une API séparée de type:

- `WorkspaceAdmin(...)`
- ou `Predictions.attach_store(...)`

#### 4. Écriture des prédictions

Le flush devrait être refactoré en 2 étages:

- batch metadata
- batch arrays

avec un protocole de commit explicite.

Exemple raisonnable:

1. écrire les arrays sous forme temporaire
2. écrire les metadata dans la transaction SQL
3. marquer le batch comme committed
4. renommer/activer les fichiers temporaires

Ou plus simple:

- écrire d'abord dans des fichiers/run directories immuables
- n'indexer qu'après succès complet

#### 5. Option analytique DuckDB conservée, mais hors chemin critique

Si on veut garder DuckDB pour son confort analytique:

- construire un snapshot DuckDB à la demande
- ou ouvrir DuckDB en lecture sur SQLite/Parquet via extensions / imports

Mais DuckDB ne doit plus être le verrou central qui conditionne:

- la relance
- l'exploration
- la maintenance

### Plan de transition réaliste

#### Phase A: corrections immédiates sans migration complète

À faire même si la migration SQLite n'est pas décidée tout de suite:

1. Fermer le store avant de retourner un `RunResult`.
2. Faire en sorte que `result.export(...)` rouvre un store court si nécessaire.
3. Faire en sorte que `Predictions.from_workspace(...)` soit détaché par défaut.
4. Introduire un mode `WorkspaceStore(read_only=True, create_schema=False, migrate=False)` pour les lectures pures.
5. Charger les arrays en batch par dataset au lieu de `load_single` par prédiction.
6. Sortir la création/migration du schéma du chemin standard de lecture.

#### Phase B: fiabilisation DuckDB si on veut encore le garder un temps

Si la migration complète est différée:

1. Ajouter un vrai verrou inter-processus explicite au niveau workspace.
2. Gérer proprement le retry au moment du `connect()`, pas seulement sur `TransactionException`.
3. Regrouper les écritures metadata en batch.
4. Éviter les connexions RW longues.
5. Avoir un utilitaire de réparation / diagnostic du workspace.

Cette phase peut réduire fortement la douleur, mais ne change pas le mauvais alignement de fond entre DuckDB live et le besoin multi-processus.

#### Phase C: migration vers SQLite WAL

1. Introduire une abstraction `WorkspaceIndex`.
2. Implémenter `SQLiteWorkspaceIndex`.
3. Conserver l'API de haut niveau inchangée.
4. Migrer automatiquement le contenu `store.duckdb` vers `index.sqlite`.
5. Garder `store.duckdb` seulement en export ou archive si nécessaire.

### Alternative encore plus radicale

Si l'objectif prioritaire absolu est la robustesse locale, il existe une option encore plus "boring":

- abandonner toute DB centrale live
- passer à des runs immuables sur disque
- utiliser un index rebuildable séparé

Exemple:

- `runs/<run_id>/manifest.json`
- `runs/<run_id>/predictions.parquet`
- `runs/<run_id>/chains.json`
- `artifacts/`

Puis:

- exploration via Polars
- index SQLite facultatif reconstruit si besoin

Cette approche est très robuste au crash, mais moins pratique si on veut un index global sophistiqué en permanence. Entre les deux options, SQLite WAL me semble le meilleur équilibre.

## Review général: avis, perspective, robustesse, etc.

### Mon avis global

Le design actuel n'est pas absurde. Il est même assez élégant analytiquement:

- DuckDB pour l'index relationnel
- Parquet pour les arrays
- artifacts sur disque

Mais il est utilisé dans un rôle pour lequel DuckDB n'est pas le meilleur composant:

- workspace opérationnel interactif
- multi-processus implicite
- objets Python qui gardent la connexion ouverte longtemps

Le résultat est prévisible:

- verrous fréquents
- workspace "inaccessible"
- erreurs au redémarrage
- difficulté à explorer les prédictions depuis un autre process

### Ce qui est déjà mieux qu'avant

Le code actuel a déjà quelques garde-fous utiles:

- `close()` fait un `CHECKPOINT`
- `WorkspaceStore` a un context manager
- il n'y a plus de mode dégradé global silencieux
- le chemin `predict(chain_id=...)` ouvre/ferme correctement le store
- les writes parallèles DuckDB ne sont pas faits dans les workers

Donc le problème n'est pas "tout est cassé". Le noyau du problème est plus ciblé:

- mauvaise durée de vie des connexions
- lecture qui garde le store attaché inutilement
- absence d'un vrai mode read-only détaché
- design non atomique DuckDB/Parquet
- schéma/migration dans le chemin d'ouverture standard

### Ce que je ferais avant la v1

Si l'objectif est "meilleur choix pour la lib", je ferais:

1. correction immédiate de la durée de vie des connexions
2. détachement par défaut des lectures workspace
3. migration du store live vers SQLite WAL

Je ne repousserais pas cette décision après la v1, parce que:

- c'est précisément le bon moment pour casser proprement l'architecture
- la dette ici est structurelle, pas cosmétique
- les symptômes touchent le coeur de l'expérience utilisateur

### Verdict

Verdict court:

- DuckDB est correct pour l'analyse.
- DuckDB n'est pas un bon choix comme index live partagé et long-lived pour ce workspace.
- Le meilleur mouvement avant v1 est de sortir DuckDB du chemin opérationnel principal.

## Références

### Références code

- `nirs4all/pipeline/storage/workspace_store.py`
- `nirs4all/pipeline/storage/array_store.py`
- `nirs4all/data/predictions.py`
- `nirs4all/pipeline/execution/executor.py`
- `nirs4all/pipeline/execution/orchestrator.py`
- `nirs4all/api/run.py`
- `nirs4all/api/result.py`
- `nirs4all/pipeline/storage/store_schema.py`

### Références externes

- DuckDB Concurrency
  - https://duckdb.org/docs/stable/connect/concurrency
- DuckDB Python DB API (`read_only=True`)
  - https://duckdb.org/docs/stable/clients/python/dbapi
- DuckDB CHECKPOINT
  - https://duckdb.org/docs/1.3/sql/statements/checkpoint
- DuckDB FAQ
  - https://duckdb.org/faq
