# Synthese NIRS synthetic spectra

Ce repertoire est le point d'entree canonique pour le travail de bench lie a la
synthese de spectres NIRS.

Les anciens chemins restent disponibles sous forme de liens symboliques pour ne
pas casser les commandes, rapports et notebooks existants:

| Ancien chemin compatible | Chemin canonique | Role |
|---|---|---|
| `bench/synthetic` | `bench/synthesis/synthetic` | Prototype historique de generation physique, comparaison real/synthetic, notebooks d'exploration. |
| `bench/nirs_synthetic_pfn` | `bench/synthesis/nirs_synthetic_pfn` | Programme bench principal: priors synthetiques, realisme X, generation XY, validations, rapports. |
| `bench/ViTnirs` | `bench/synthesis/ViTnirs` | Cadrage du futur encodeur spectral universel de type ViT/Perceiver. |
| `bench/Synthesis_Objectives.md` | `bench/synthesis/Synthesis_Objectives.md` | Version reformulee des objectifs sous forme de sujet de recherche. |

Le code de production reste dans `nirs4all/synthesis`. Ce repertoire `bench` ne
doit pas devenir une seconde librairie: il sert a organiser les hypotheses, les
experiences, les rapports, les echecs documentes et les gates avant integration.

## Documents de cadrage

- `Synthesis_Objectives.md`: reformulation et extension des objectifs comme
  sujet de recherche.
- `Work_Done_Summary.md`: synthese haut niveau du travail deja realise, avec
  pointeurs vers l'existant.
- `Future_Work_Program.md`: programme experimental pragmatique pour la suite.
- `original/Synthesis_Objectives_initial.md`: notes initiales conservees telles
  quelles apres reorganisation.

## Reperes rapides

- Moteur de synthese maintenu: `nirs4all/synthesis`.
- Prototype physique historique: `bench/synthesis/synthetic`.
- Validation/PFN/XY: `bench/synthesis/nirs_synthetic_pfn`.
- Plan encodeur latent: `bench/synthesis/ViTnirs`.
- Donnees de reference courantes: `bench/tabpfn_paper/data`.

Les fichiers `bench/benchmark_synthesis.md` et
`bench/build_benchmark_synthesis.py` ne sont pas deplaces ici: ils synthetisent
les resultats de benchmarks de modeles au sens "rapport de synthese", pas la
generation de spectres synthetiques.
