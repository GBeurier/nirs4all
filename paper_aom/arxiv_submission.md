# Publier le draft AOM sur arXiv

Guide pratique pour déposer `main.tex` (+ supplément) en preprint sur
[arxiv.org](https://arxiv.org). Le draft est en LaTeX standard
(`article`, `natbib`, `graphicx`) — il est compatible avec la chaîne de
compilation arXiv (TeX Live, sans `shell-escape`).

---

## 1. Pré-requis côté auteur

1. **Compte arXiv** : créer un compte sur https://arxiv.org/user/register
   avec une adresse institutionnelle (CIRAD/INRAE de préférence). Le mail
   institutionnel évite la procédure d'endorsement dans la plupart des
   catégories.
2. **Endorsement** : pour un premier dépôt dans une catégorie donnée,
   arXiv peut demander un "endorser" (un auteur ayant déjà publié dans
   cette catégorie sur arXiv). Demander à un co-auteur déjà publié, ou
   solliciter un collègue référent NIRS / chimiométrie.
3. **ORCID** : lier son ORCID au compte arXiv. Faire de même pour les
   co-auteurs avant la soumission (champ "Authors" lit l'ORCID).
4. **Accord des co-auteurs** : confirmer par écrit (mail) l'accord de
   chaque co-auteur sur la version déposée. arXiv considère que le
   submitter atteste cet accord.

---

## 2. Choisir les catégories

Catégorie principale recommandée pour ce papier :

- `stat.AP` — Statistics / Applications (benchmark + méthodologie PLS).

Cross-listings pertinents :

- `eess.SP` — Signal Processing (prétraitements spectraux).
- `physics.data-an` — Data Analysis, Statistics and Probability
  (chimiométrie quantitative).
- `cs.LG` — Machine Learning (uniquement si on veut viser une audience
  ML ; sinon éviter pour ne pas diluer le scope).

Choisir **une** catégorie principale et 1–2 cross-lists maximum.

---

## 3. Choisir la licence

Recommandation : **arXiv non-exclusive license to distribute** (par
défaut). Elle est compatible avec une soumission ultérieure à *Talanta*
(Elsevier autorise le dépôt en preprint avant soumission ; cf. politique
"sharing" Elsevier).

Si on veut un usage plus large (réutilisation par des tiers) :
`CC BY 4.0`. **Ne pas** choisir `CC BY-NC` ou `CC BY-SA` si la cible
reste Talanta : certains éditeurs refusent les preprints sous licences
restrictives.

> ⚠️ La licence arXiv est **définitive** dès la première version. Elle
> ne peut pas être assouplie après publication (uniquement durcie).

---

## 4. Constituer l'archive source

arXiv exige une **archive de sources LaTeX** (pas un PDF nu, sauf
exception). Préparer un dossier `arxiv_submission/` contenant :

```
arxiv_submission/
├── main.tex
├── main.bbl                ← REQUIS (arXiv ne lance pas bibtex)
├── references.bib          ← optionnel mais utile
├── supplement.tex          ← si on veut un PDF séparé
├── supplement.bbl
├── tables/*.tex
├── figures/*.pdf
└── 00README.XXX            ← optionnel, ordre des fichiers
```

Étapes pratiques depuis `paper_aom/` :

```bash
# 1. Build complet pour générer .bbl et .aux à jour
./build.sh

# 2. Préparer un dossier propre
mkdir -p /tmp/arxiv_submission
cp main.tex supplement.tex references.bib /tmp/arxiv_submission/
cp main.bbl supplement.bbl /tmp/arxiv_submission/
cp -r tables figures /tmp/arxiv_submission/

# 3. Tester la compilation arXiv (sans bibtex, comme arXiv le fait)
cd /tmp/arxiv_submission
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # 2e passe pour les refs

# 4. Vérifier qu'il n'y a aucun warning bloquant ("Undefined references"
#    sur les citations doit disparaître après la 2e passe).

# 5. Créer l'archive
cd /tmp
tar czf arxiv_submission.tar.gz arxiv_submission/
```

### Points de vigilance LaTeX

- **Pas de `\write18` / `shell-escape`** : arXiv refuse. Vérifier qu'on
  n'utilise pas `minted` (utiliser `listings` si besoin).
- **Pas de chemins absolus** dans `\includegraphics` ou `\input`.
- **Figures en PDF** (pas en EPS ni PNG haute résolution sans raison).
  Toutes les figures sont déjà en PDF via `scripts/make_figures.py`.
- **`natbib`** : OK, arXiv le supporte. Les `.bbl` doivent être
  inclus dans l'archive.
- **Pas de fichiers cachés** (`.DS_Store`, `__pycache__/`, etc.) dans
  le tar.
- **Encodage UTF-8** : déjà configuré (`inputenc utf8`). Vérifier les
  caractères Unicode dans les noms d'auteurs (`{\'e}`, `{\^u}`).
- **Taille** : limite arXiv = 50 MB par soumission. Le draft actuel
  (~440 kB PDF + figures) est très en dessous.

---

## 5. Supplément : un ou deux PDFs ?

Deux options :

**A. PDF unifié** (recommandé pour la lisibilité arXiv) : ajouter le
contenu du supplément à la fin de `main.tex` via `\appendix` ou
`\input{supplement.tex}`. arXiv n'affiche qu'**un seul PDF principal**,
donc l'audience le verra mieux.

**B. PDFs séparés (`main` + `supplement`)** : déposer `supplement.tex`
comme **ancillary file** via l'interface arXiv (onglet "Ancillary
files"). Le PDF supplément est alors téléchargeable séparément, mais
n'apparaît pas dans la prévisualisation par défaut.

Pour ce draft, l'option **A** est plus simple : ajouter à la fin de
`main.tex` :

```latex
\appendix
\input{supplement_body.tex}  % version sans \documentclass / \begin{document}
```

(nécessite de scinder `supplement.tex` en préambule + corps, ou
d'utiliser `\subfile`).

Sinon, garder l'option B et déposer `supplement.pdf` comme ancillary.

---

## 6. Workflow de dépôt (interface web)

1. Se connecter sur https://arxiv.org/submit.
2. **Start New Submission**.
3. **License** : choisir (cf. §3).
4. **Archive** : uploader `arxiv_submission.tar.gz`.
5. arXiv compile et affiche le PDF généré. **Comparer pixel à pixel**
   avec `main.pdf` local. Si écart (police, figures manquantes,
   références cassées) → corriger et re-uploader.
6. **Metadata** :
   - Title : copier le titre exact de `main.tex` (sans `\textbf`,
     sans LaTeX).
   - Authors : `Beurier, Grégory; Reiter, Robin; Noûs, Camille;
     Rouan, Lauriane; Cornet, Denis` (format `Last, First`,
     séparés par `;`).
   - Abstract : copier-coller depuis `main.tex`. **Convertir LaTeX en
     texte plat** (`$\beta$` → `beta`, supprimer `\cite`, etc.).
     Limite : 1920 caractères.
   - Comments : `XX pages, YY figures, ZZ tables. Submitted to Talanta.`
     Ajouter un lien GitHub si le code est public :
     `Code: https://github.com/...`.
   - MSC class / ACM class : laisser vide (non standard en chimiométrie).
   - Journal-ref / DOI : laisser vide tant que non accepté à Talanta.
   - Report number : laisser vide.
7. **Categories** : primary + cross-lists (cf. §2).
8. **Preview & Submit**. Une fois soumis, le preprint passe en file de
   modération (10 min à 24 h en semaine, plus long le week-end).
9. Le mail de confirmation arrive avec l'identifiant `arXiv:YYMM.NNNNN`.

---

## 7. Après publication

- **DOI arXiv** : `https://doi.org/10.48550/arXiv.YYMM.NNNNN` (utilisable
  immédiatement, citer dans la cover letter Talanta).
- **Mise à jour (v2, v3…)** : interface arXiv → "Replace". Garder le
  même `.tar.gz` structure. Pas de limite au nombre de versions.
  Documenter les changements majeurs dans le champ "Comments".
- **Retrait** : possible mais le metadata reste public. À éviter ; faire
  une `v2` corrigée à la place.
- **Synchronisation avec Talanta** : informer le journal du dépôt arXiv
  dans la cover letter (`cover_letter_talanta.md` à mettre à jour). La
  politique Elsevier autorise le preprint avant soumission.

---

## 8. Checklist finale avant soumission

- [ ] `./build.sh` passe sans warning bloquant.
- [ ] `main.bbl` et `supplement.bbl` à jour et inclus dans le tar.
- [ ] Toutes les figures (`figures/*.pdf`) présentes.
- [ ] Tous les tableaux (`tables/*.tex`) présents.
- [ ] Pas de `shell-escape`, pas de `minted`, pas de `write18`.
- [ ] Pas de chemins absolus dans `\input` / `\includegraphics`.
- [ ] Abstract converti en texte plat (< 1920 caractères).
- [ ] Co-auteurs OK sur la version exacte qui sera déposée.
- [ ] ORCIDs des co-auteurs disponibles.
- [ ] Endorser confirmé (si premier dépôt en `stat.AP`).
- [ ] Licence choisie (et acceptée comme définitive).
- [ ] Lien code/data ajouté dans "Comments" si applicable.
- [ ] Test de compilation sur `/tmp/arxiv_submission/` réussi.

---

## 9. Références utiles

- Aide arXiv (LaTeX) : https://info.arxiv.org/help/submit_tex.html
- Catégories arXiv : https://arxiv.org/category_taxonomy
- Politique Elsevier (preprints) :
  https://www.elsevier.com/about/policies/sharing
- Endorsement arXiv : https://info.arxiv.org/help/endorsement.html
