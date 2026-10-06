# 4. Define and build a dataset

Describe the scientific observations before describing file locations. A dataset declaration identifies sources, sample IDs, representations, axes, units, targets, partitions and experimental groups. File readers and catalog datasets are ways to assemble that declaration.

## Separate inputs from representations

| Input | Reader/assembly route | Required decisions |
|---|---|---|
| Matrix or array | In-memory SDK/IO declaration | Rows, feature axes, source IDs and dtype |
| CSV/Parquet/Excel | IO loaders and table assembly | ID columns, targets, metadata and join rules |
| Vendor instrument payload | Formats reader then IO | Representation, physical coordinates and units |
| Catalog dataset | DOI-pinned catalog then IO | Dataset/version and the intended evaluation cohort |
| Image or tensor | Typed IO tensor source | Non-sample dimensions, axis names and encoder recipe |
| Variable-length series | Ragged source contract | Offsets, time coordinates, presence and the chosen processing policy |
| JSON/YAML definition | Configuration/parser route | Schema/version and resolved source references |

An array shape does not establish feature equivalence. Two equally shaped spectra with reordered wavelength columns or changed units are different inputs. Reload verifies source identity, feature order and coordinate schema independently of prediction values.

## Define identity and masks

Use unique sample IDs and declare the alignment of every source. Groups define the validation unit when related observations must stay together. Repetition and origin IDs express repeated or derived observations. A partition declares intended use: training, held-out test, prediction or calibration according to the profile.

Missing modalities, invalid feature values and unobserved targets are different cases. Keep their masks separate. A target-free prediction cohort is not a training cohort with invented zeros. Ragged offsets must describe real row boundaries and time coordinates; padding or aggregation requires an explicit supported recipe, not an implicit loader decision.

## Multimodal workflow

The complete native early-fusion example uses four named sources: NIR, image, series and mixed metadata. Encoders are fitted on training-only rows in each fold. Their learned scaling, PCA, category vocabulary and fusion model are exported together. Unknown metadata categories follow the declared one-hot policy. The frozen raw schema accompanies the fitted state.

For the executable recipe, see {doc}`/user_guide/data/methods_multimodal_u07`. For source alignment, grouped multimodal CV and local/global tuning, see {doc}`/user_guide/data/multimodal`. The SDK late-fusion partial-cohort profile is described in {doc}`/user_guide/data/multimodal_late_partial`; it is a distinct execution profile from dense native early fusion.

## Build a declaration step by step

1. Choose stable observation/sample IDs and the experimental validation unit.
2. Declare each source's role, representation, axes, dtype, units and coordinates.
3. Align by IDs and record missing-source masks; never rely on file ordering alone.
4. Attach targets and their masks, metadata and groups.
5. Assign train/test/calibration/predict partitions without leaking labels into fitting.
6. Validate the declaration and select a workflow profile that accepts it.

The {doc}`configuration reference </reference/configuration>` lists SDK fields, defaults and file forms. {doc}`/user_guide/data/loading_data` covers practical assembly; {doc}`/user_guide/data/heterogeneous_repetitions` covers repeated observations; {doc}`/user_guide/data/signal_types` describes signal semantics. The [IO reference](https://github.com/GBeurier/nirs4all-io) owns native source and public-dataset contracts; the [Formats reference](https://github.com/GBeurier/nirs4all-formats) owns reader parameters.
