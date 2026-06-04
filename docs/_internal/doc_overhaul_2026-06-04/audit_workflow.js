export const meta = {
  name: 'nirs4all-docs-audit',
  description: 'Exhaustive grounded audit of nirs4all docs + examples (20 segment readers → synthesis → completeness critic)',
  phases: [
    { title: 'Audit', detail: 'one grounded reader per doc/examples segment' },
    { title: 'Synthesize', detail: 'cross-cutting themes + prioritized defect register' },
    { title: 'Critique', detail: 'completeness critic: what coverage is missing' },
  ],
}

const ROOT = '/home/delete/nirs4all/nirs4all'
const CARD = `${ROOT}/docs/_internal/doc_overhaul_2026-06-04/GROUND_TRUTH.md`

// ---- Segment definitions (explicit file lists => deterministic, no silent gaps) ----
const DS = (p) => `${ROOT}/docs/source/${p}`
const EX = (p) => `${ROOT}/examples/${p}`

const SEGMENTS = [
  // ---------- Documentation (narrative .md) ----------
  { key: 'doc:front-door', kind: 'doc', title: 'Landing page + getting started',
    files: ['index.md','ai_onboarding.md','getting_started/index.md','getting_started/installation.md','getting_started/quickstart.md','getting_started/tutorial.md','getting_started/concepts.md'].map(DS),
    focus: 'This is the FIRST thing a new user sees. Assess: clarity of the front door, the install→first-run→next-step path, whether there is a single obvious starting point or competing entry points, version/license correctness, and whether the quickstart actually matches the real public API.' },
  { key: 'doc:concepts', kind: 'doc', title: 'concepts/',
    files: ['concepts/index.md','concepts/pipelines.md','concepts/datasets.md','concepts/cross_validation.md','concepts/branching_and_merging.md','concepts/generators.md','concepts/augmentation.md','concepts/predictions_and_deployment.md'].map(DS),
    focus: 'Conceptual/explanation layer. Assess overlap/redundancy with onboarding/ and user_guide/ (do the same concepts get re-explained in 3 places?), conceptual accuracy vs source, and progression.' },
  { key: 'doc:onboarding', kind: 'doc', title: 'onboarding/',
    files: ['onboarding/index.md','onboarding/mental_models.md','onboarding/data_workflow.md','onboarding/pipeline_workflow.md','onboarding/workspace_intro.md','onboarding/controllers_intro.md','onboarding/persona_paths.md'].map(DS),
    focus: 'A THIRD front door competing with getting_started/ and concepts/. persona_paths.md is KNOWN to reference nonexistent example files and a nonexistent nirs4all.load_dataset API — verify every file/API reference in this segment against the ground-truth card and source. Flag every broken reference.' },
  { key: 'doc:ug-pipelines', kind: 'doc', title: 'user_guide/pipelines/ + ug root',
    files: ['user_guide/index.md','user_guide/scoring_and_refit.md','user_guide/logging.md','user_guide/pipelines/index.md','user_guide/pipelines/writing_pipelines.md','user_guide/pipelines/branching.md','user_guide/pipelines/merging.md','user_guide/pipelines/stacking.md','user_guide/pipelines/generators.md','user_guide/pipelines/multi_source.md','user_guide/pipelines/cache_optimization.md'].map(DS),
    focus: 'Core how-to layer for pipelines. CRITICAL for the style mandate: check EVERY python code block for the "1 pipeline inline, no external variable, run()" form. Flag every block that assigns pipeline=[...] to a variable then calls run(pipeline,...). Check keyword/API accuracy vs source.' },
  { key: 'doc:ug-data', kind: 'doc', title: 'user_guide/data/ + troubleshooting/',
    files: ['user_guide/data/index.md','user_guide/data/loading_data.md','user_guide/data/signal_types.md','user_guide/data/aggregation.md','user_guide/data/sample_filtering.md','user_guide/data/synthetic_data.md','user_guide/troubleshooting/index.md','user_guide/troubleshooting/faq.md','user_guide/troubleshooting/dataset_troubleshooting.md','user_guide/troubleshooting/migration.md'].map(DS),
    focus: 'Data loading + troubleshooting. Verify the synthetic_data / generate() API matches the ground-truth card. Check whether troubleshooting/migration.md duplicates docs/source/migration/. Style mandate on code blocks.' },
  { key: 'doc:ug-preproc', kind: 'doc', title: 'user_guide/preprocessing/ + augmentation/',
    files: ['user_guide/preprocessing/index.md','user_guide/preprocessing/overview.md','user_guide/preprocessing/handbook.md','user_guide/preprocessing/cheatsheet.md','user_guide/preprocessing/snv.md','user_guide/preprocessing/resampler.md','user_guide/preprocessing/transfer_preprocessing_selector_cheatsheet.md','user_guide/augmentation/index.md','user_guide/augmentation/augmentations.md','user_guide/augmentation/sample_augmentation_guide.md','user_guide/augmentation/synthetic_nirs_generator.md'].map(DS),
    focus: 'Preprocessing + augmentation how-to. Check overlap between overview/handbook/cheatsheet (3 preprocessing summaries?), and between augmentation/ here and concepts/augmentation.md and reference/augmentations.md. Operator names must exist in nirs4all.operators.transforms / augmentation. Style mandate.' },
  { key: 'doc:ug-models', kind: 'doc', title: 'user_guide/models/ + visualization/',
    files: ['user_guide/models/index.md','user_guide/models/training.md','user_guide/models/deep_learning.md','user_guide/models/hyperparameter_tuning.md','user_guide/models/tabpfn_nirs.md','user_guide/visualization/index.md','user_guide/visualization/in_pipeline_charts.md','user_guide/visualization/prediction_charts.md','user_guide/visualization/pipeline_diagram.md','user_guide/visualization/shap.md'].map(DS),
    focus: 'Models + visualization how-to. Verify finetune_params/train_params and model class names vs source. SHAP/explain API vs ground-truth card. Style mandate.' },
  { key: 'doc:ug-predict-deploy', kind: 'doc', title: 'user_guide/predictions/ + deployment/',
    files: ['user_guide/predictions/index.md','user_guide/predictions/making_predictions.md','user_guide/predictions/understanding_predictions.md','user_guide/predictions/analyzing_results.md','user_guide/predictions/advanced_predictions.md','user_guide/predictions/exporting_models.md','user_guide/predictions/session_api.md','user_guide/deployment/index.md','user_guide/deployment/export_bundles.md','user_guide/deployment/prediction_model_reuse.md','user_guide/deployment/retrain_transfer.md'].map(DS),
    focus: 'Predict/deploy/retrain/session how-to — these are STABLE 0.9.x contracts. Verify predict()/retrain()/session()/load_session()/export() signatures and PredictResult members against the ground-truth card EXACTLY. Check overlap between predictions/exporting_models.md and deployment/export_bundles.md. Style mandate.' },
  { key: 'doc:ref-pipeline', kind: 'doc', title: 'reference/ (pipeline-centric)',
    files: ['reference/index.md','reference/pipeline_syntax.md','reference/pipeline_keywords.md','reference/generator_keywords.md','reference/combination_generator.md','reference/operator_catalog.md','reference/configuration.md','reference/configuration/index.md','reference/configuration/cache_config.md'].map(DS),
    focus: 'Reference layer for pipeline syntax/keywords/generators. This must be EXHAUSTIVE and EXACT. Cross-check the keyword table and generator keyword list against the project CLAUDE.md tables and source (controllers + generator). Flag missing or wrong keywords. Check redundancy between configuration.md and configuration/index.md.' },
  { key: 'doc:ref-operators', kind: 'doc', title: 'reference/ (operators/storage)',
    files: ['reference/models.md','reference/transforms.md','reference/splitters.md','reference/filters.md','reference/augmentations.md','reference/metrics.md','reference/storage.md','reference/workspace.md','reference/predictions_api.md','reference/cli.md'].map(DS),
    focus: 'Reference for operators, storage, CLI. Spot-check that listed operators/models/splitters/filters actually exist in nirs4all.operators.* and that CLI subcommands match nirs4all/cli. Flag stale/removed/renamed items. Note completeness gaps vs the actual operator set.' },
  { key: 'doc:developer', kind: 'doc', title: 'developer/ + migration/',
    files: ['developer/index.md','developer/architecture.md','developer/pipeline_architecture.md','developer/controllers.md','developer/artifacts.md','developer/artifacts_internals.md','developer/outputs_vs_artifacts.md','developer/caching.md','developer/metadata.md','developer/synthetic.md','developer/testing.md','migration/duckdb_to_sqlite.md','migration/storage_migration_webapp.md'].map(DS),
    focus: 'Internals/contributor docs + migration notes. Verify architecture descriptions vs the real module layout (controllers registry, storage SQLite+Parquet, bundle). Flag references to removed subsystems (e.g. duckdb if fully migrated to sqlite). Is migration/ still relevant or historical?' },
  { key: 'doc:examples-pages', kind: 'doc', title: 'docs examples/ pages',
    files: ['examples/index.md','examples/developer.md','examples/user/getting_started.md','examples/user/data_handling.md','examples/user/preprocessing.md','examples/user/models.md','examples/user/cross_validation.md','examples/user/deployment.md','examples/user/explainability.md'].map(DS),
    focus: 'These doc pages describe/index the example .py files. Cross-check that every example file they reference actually exists at the path given (the examples tree is user/{01..07} and developer/{01..06}). Flag broken example links, wrong counts ("50+/67/progressive"), and stale descriptions.' },

  // ---------- Examples (executable .py — also integration tests) ----------
  { key: 'ex:user-start-data', kind: 'example', title: 'user/01_getting_started + 02_data_handling',
    files: ['user/01_getting_started/U01_hello_world.py','user/01_getting_started/U02_basic_regression.py','user/01_getting_started/U03_basic_classification.py','user/01_getting_started/U04_visualization.py','user/02_data_handling/U01_flexible_inputs.py','user/02_data_handling/U02_multi_datasets.py','user/02_data_handling/U03_multi_source.py','user/02_data_handling/U04_wavelength_handling.py','user/02_data_handling/U05_synthetic_data.py','user/02_data_handling/U06_synthetic_advanced.py'].map(EX),
    focus: 'The on-ramp examples. STYLE MANDATE IS PRIMARY HERE: every example must use "1 inline pipeline, dataset(s), run()" with NO external pipeline variable. Flag each violation with file + the offending lines. Also assess: progression/teaching value, header docstring quality, whether they over-import or show too much at once, and API correctness vs ground-truth card.' },
  { key: 'ex:user-preproc-models', kind: 'example', title: 'user/03_preprocessing + 04_models',
    files: ['user/03_preprocessing/U01_preprocessing_basics.py','user/03_preprocessing/U02_feature_augmentation.py','user/03_preprocessing/U03_sample_augmentation.py','user/03_preprocessing/U04_signal_conversion.py','user/03_preprocessing/U05_orthogonalization.py','user/03_preprocessing/U06_wavelet_denoise.py','user/04_models/U01_multi_model.py','user/04_models/U02_hyperparameter_tuning.py','user/04_models/U03_stacking_ensembles.py','user/04_models/U04_pls_variants.py','user/04_models/U05_advanced_finetuning.py','user/04_models/U06_tabpfn_nirs.py'].map(EX),
    focus: 'Preprocessing + models examples. Style mandate (inline pipeline). API/operator correctness vs source. Consistency of structure across files (do they all follow the same skeleton?).' },
  { key: 'ex:user-cv-deploy-explain', kind: 'example', title: 'user/05_cv + 06_deployment + 07_explainability',
    files: ['user/05_cross_validation/U01_cv_strategies.py','user/05_cross_validation/U02_group_splitting.py','user/05_cross_validation/U03_sample_filtering.py','user/05_cross_validation/U04_aggregation.py','user/05_cross_validation/U05_tagging_analysis.py','user/05_cross_validation/U06_exclusion_strategies.py','user/06_deployment/U01_save_load_predict.py','user/06_deployment/U02_export_bundle.py','user/06_deployment/U03_workspace_management.py','user/06_deployment/U04_sklearn_integration.py','user/07_explainability/U01_shap_basics.py','user/07_explainability/U02_shap_sklearn.py','user/07_explainability/U03_feature_selection.py'].map(EX),
    focus: 'CV/deploy/explain examples. Style mandate. Verify predict/export/explain usage vs ground-truth card. These touch the stable 0.9.x contract — be exact.' },
  { key: 'ex:dev-pipelines-gen', kind: 'example', title: 'developer/01_advanced_pipelines + 02_generators',
    files: ['developer/01_advanced_pipelines/D01_branching_basics.py','developer/01_advanced_pipelines/D02_branching_advanced.py','developer/01_advanced_pipelines/D03_merge_basics.py','developer/01_advanced_pipelines/D04_merge_sources.py','developer/01_advanced_pipelines/D05_meta_stacking.py','developer/01_advanced_pipelines/D06_separation_branches.py','developer/01_advanced_pipelines/D07_value_mapping.py','developer/02_generators/D01_generator_syntax.py','developer/02_generators/D02_generator_advanced.py','developer/02_generators/D03_generator_iterators.py','developer/02_generators/D04_nested_generators.py','developer/02_generators/D05_synthetic_custom_components.py','developer/02_generators/D06_synthetic_testing.py','developer/02_generators/D07_synthetic_wavenumber_procedural.py','developer/02_generators/D08_synthetic_application_domains.py','developer/02_generators/D09_synthetic_instruments.py'].map(EX),
    focus: 'Advanced pipeline + generator examples. Style mandate (note: branching/generator pipelines are large — assess whether "fully inline" is realistic here or whether a documented exception is warranted; report your judgment). API correctness for branch/merge/generator keywords.' },
  { key: 'ex:dev-dl-transfer-internals', kind: 'example', title: 'developer/03_dl + 04_transfer + 05_advanced + 06_internals',
    files: ['developer/03_deep_learning/D01_pytorch_models.py','developer/03_deep_learning/D02_jax_models.py','developer/03_deep_learning/D03_tensorflow_models.py','developer/03_deep_learning/D04_framework_comparison.py','developer/04_transfer_learning/D01_transfer_analysis.py','developer/04_transfer_learning/D02_retrain_modes.py','developer/04_transfer_learning/D03_pca_geometry.py','developer/05_advanced_features/D01_metadata_branching.py','developer/05_advanced_features/D02_concat_transform.py','developer/05_advanced_features/D03_repetition_transform.py','developer/06_internals/D01_session_workflow.py','developer/06_internals/D02_custom_controllers.py','developer/06_internals/D03_cache_performance.py','developer/06_internals/D04_parallel_branches.py','developer/06_internals/D05_binary_search_sampler.py'].map(EX),
    focus: 'DL/transfer/internals examples. Style mandate. Verify retrain modes ("full"/"transfer"/"finetune"), session workflow, custom controller registration vs source. DL backends are lazy-loaded — note any import-time assumptions.' },
  { key: 'ex:reference-infra', kind: 'example', title: 'examples reference/ + infra + README',
    files: ['reference/R01_pipeline_syntax.py','reference/R02_generator_reference.py','reference/R03_all_keywords.py','reference/R05_synthetic_environmental.py','reference/R06_synthetic_validation.py','reference/R07_synthetic_fitter.py','pipeline_samples/test_all_pipelines.py','pipeline_samples/export_canonical.py','README.md','run.sh','run_ci_examples.sh','ci_example_launcher.py','example_utils.py'].map(EX),
    focus: 'Reference examples + the examples-as-integration-tests machinery. Note the MISSING R04 (gap in numbering — is it intentional or a deletion artifact?). Assess: how examples are run as CI tests, whether README counts match reality, whether the runner enforces anything about style, and whether R03_all_keywords stays in sync with the real keyword set. Style mandate on R-files.' },

  // ---------- Root / meta / build ----------
  { key: 'root:meta', kind: 'root', title: 'README + INSTALLATION + root meta',
    files: [`${ROOT}/README.md`,`${ROOT}/INSTALLATION.md`,`${ROOT}/CLAUDE.md`,`${ROOT}/AGENTS.md`,`${ROOT}/CONTRIBUTING.md`,`${ROOT}/Roadmap.md`,`${ROOT}/CHANGELOG.md`],
    focus: 'Repo front matter. PRIMARY CHECK: license consistency (LICENSE file = AGPL-3.0 per commit 7bd07280) and version (0.9.1) across README/CLAUDE/etc. Verify the README quickstart matches the real public API and the enforced inline-pipeline style. Does README point to the right docs/examples entry point? Is CHANGELOG current to 0.9.1?' },
  { key: 'build:rtd', kind: 'build', title: 'docs build config + RTD',
    files: [`${ROOT}/docs/source/conf.py`,`${ROOT}/docs/Makefile`,`${ROOT}/docs/build_docs.sh`,`${ROOT}/docs/readthedocs.requirements.txt`,`${ROOT}/docs/backlog.md`,`${ROOT}/docs/suggestions.md`,`${ROOT}/docs/source/migration/duckdb_to_sqlite.md`],
    focus: 'Sphinx/RTD build health. Read conf.py: extensions, myst, autodoc, theme, intersphinx. The api/ tree has 357 autodoc .rst stubs — are they generated or hand-written, and are they wired into a toctree (or orphaned)? Check the landing toctree in index.md for orphaned/missing pages. Read backlog.md + suggestions.md: are these stale internal notes that should not ship? Report concrete build-correctness risks.' },
]

const SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['segment','files_reviewed','health_score','defects','structural_issues','style_violations','strengths','summary'],
  properties: {
    segment: { type: 'string' },
    files_reviewed: { type: 'array', items: { type: 'string' } },
    health_score: { type: 'integer', minimum: 1, maximum: 5, description: '5=best-in-class, 1=broken/misleading' },
    defects: { type: 'array', items: {
      type: 'object', additionalProperties: false,
      required: ['severity','type','location','description','fix'],
      properties: {
        severity: { type: 'string', enum: ['critical','high','medium','low'] },
        type: { type: 'string', enum: ['stale','broken_ref','wrong_api','inconsistency','missing','clarity','redundancy','style','build'] },
        location: { type: 'string', description: 'file path, ideally with line/anchor' },
        description: { type: 'string' },
        evidence: { type: 'string', description: 'the exact source/ground-truth fact that proves it is a defect' },
        fix: { type: 'string' },
      },
    } },
    structural_issues: { type: 'array', items: { type: 'string' }, description: 'IA/progression/redundancy/overlap problems spanning files in this segment' },
    style_violations: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['file','description'],
      properties: { file: { type: 'string' }, description: { type: 'string', description: 'external-pipeline-variable or other deviation from "1 inline pipeline + run()"' } },
    } },
    strengths: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string', description: '3-6 sentence verdict on this segment' },
  },
}

function readerPrompt(s) {
  return `You are a meticulous documentation auditor for the nirs4all Python library (a NIRS/spectroscopy ML library, v0.9.1). Audit the "${s.title}" segment.

FIRST, Read the ground-truth card: ${CARD}
It is authoritative for the public API, signatures, version (0.9.1), license (AGPL-3.0), and known defects. Re-read it; do not rely on memory.

THEN audit exactly these files (Read each fully):
${s.files.map(f => '  - ' + f).join('\n')}

GROUND EVERY CLAIM. When you assert a symbol/path/signature is wrong, broken, stale, or missing, verify it against the ACTUAL repo: Read/Grep the source under ${ROOT}/nirs4all/ (the package), and check example/file paths actually exist under ${ROOT}/examples/ or ${ROOT}/docs/. Never speculate. A defect must cite the source/ground-truth fact that proves it (the "evidence" field).

Audit dimensions (this is "${s.kind}" content):
1. CORRECTNESS — stale version/license, wrong API names/signatures, broken cross-references to files or symbols that do not exist, operators/keywords that no longer exist or are renamed, code that would error against the real 0.9.1 API.
2. STRUCTURE/IA — overlap and duplication with OTHER doc areas (e.g. the same concept explained in getting_started/ AND concepts/ AND onboarding/ AND user_guide/), missing progression/fil conducteur, no clear single entry point, orphaned or unreachable pages, ordering that does not teach.
3. CLARITY for newcomers — is information well structured for someone entering the library, or a wall of undifferentiated text? Are there missing "next step" signposts?
4. STYLE UNIFORMITY (MANDATE) — the user requires every pipeline to be written INLINE in one shot inside run(), with NO external pipeline variable, for readability:
       result = nirs4all.run(pipeline=[ ...inline... ], dataset=...)
   Flag EVERY code block / example that instead does \`pipeline = [...]\` then \`run(pipeline, ...)\`, or builds the pipeline across multiple statements/variables. Put each in style_violations with the file and the offending construct. (For genuinely huge generator/branching pipelines, note whether a documented exception is warranted — but still report them.)
5. COMPLETENESS — what a best-in-class doc for this topic would have that is absent here.

Score health_score 1-5 (5 = best-in-class, publish as-is; 1 = actively misleading/broken). Be exhaustive but precise — every defect must be real and evidenced. Return the structured object. Your response IS data, not a message.`
}

phase('Audit')
const findings = await parallel(
  SEGMENTS.map(s => () => agent(readerPrompt(s), { schema: SCHEMA, phase: 'Audit', label: s.key, agentType: 'Explore' }))
)
const ok = findings.filter(Boolean)
log(`Segment audit complete: ${ok.length}/${SEGMENTS.length} readers returned findings`)

// ---- Cross-cutting synthesis (barrier justified: themes/dedup need ALL segments at once) ----
phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['executive_summary','cross_cutting_themes','prioritized_register','style_mandate_rollup','health_table','top_quick_wins'],
  properties: {
    executive_summary: { type: 'string', description: '1-2 paragraph overall verdict on doc+examples health' },
    cross_cutting_themes: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['theme','severity','segments_affected','description','recommendation'],
      properties: {
        theme: { type: 'string' },
        severity: { type: 'string', enum: ['critical','high','medium','low'] },
        segments_affected: { type: 'array', items: { type: 'string' } },
        description: { type: 'string' },
        recommendation: { type: 'string' },
      },
    }, description: 'patterns spanning multiple segments: e.g. competing front doors, license/version drift count, style violations N/M segments, concept duplication map' },
    prioritized_register: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['rank','severity','title','where','why','fix'],
      properties: {
        rank: { type: 'integer' },
        severity: { type: 'string', enum: ['critical','high','medium','low'] },
        title: { type: 'string' }, where: { type: 'string' }, why: { type: 'string' }, fix: { type: 'string' },
      },
    }, description: 'the consolidated, de-duplicated, ranked defect list (merge identical defects reported by multiple readers into one row with all locations)' },
    style_mandate_rollup: { type: 'string', description: 'how widespread are external-pipeline-variable violations across docs+examples; quantify (segments/files affected) and name the worst offenders' },
    health_table: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['segment','score','headline'],
      properties: { segment: { type: 'string' }, score: { type: 'integer' }, headline: { type: 'string' } },
    } },
    top_quick_wins: { type: 'array', items: { type: 'string' }, description: 'high-value, low-effort fixes to do first' },
  },
}
const synthesis = await agent(
  `You are the lead documentation architect synthesizing an exhaustive audit of nirs4all's docs+examples. Below is the FULL set of ${ok.length} grounded segment findings as JSON. Read the ground-truth card too: ${CARD}.

Produce a cross-cutting synthesis:
- Merge defects that multiple readers reported into single register rows (dedup by root cause, keep all locations).
- Identify repo-wide THEMES (competing front doors getting_started/concepts/onboarding; concept duplication across layers; license/version drift; style-mandate violations; reference completeness gaps; examples-as-tests coverage; RTD/autodoc orphans).
- Quantify wherever possible (e.g. "license drift in N files", "style violations in M/20 segments").
- Build a prioritized register ranked by (severity x reach x how-misleading-to-a-newcomer).
- Roll up the style mandate compliance.
- Build a health table (one row per segment with its score).
- List top quick wins.

Be specific and quantified. Your response IS data.

=== SEGMENT FINDINGS (JSON) ===
${JSON.stringify(ok)}`,
  { schema: SYNTH_SCHEMA, phase: 'Synthesize', label: 'synthesis' }
)

// ---- Completeness critic ----
phase('Critique')
const CRITIC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['coverage_gaps','missed_or_underweighted','false_or_unproven_claims','verdict'],
  properties: {
    coverage_gaps: { type: 'array', items: { type: 'string' }, description: 'doc/example areas or audit dimensions not adequately covered' },
    missed_or_underweighted: { type: 'array', items: { type: 'string' }, description: 'important problems the synthesis under-weighted or missed' },
    false_or_unproven_claims: { type: 'array', items: { type: 'string' }, description: 'any audit finding that looks unproven/over-stated and should be re-verified before acting' },
    verdict: { type: 'string', description: 'is the audit exhaustive enough to base a restructuring on? what one more pass would add most value?' },
  },
}
const critic = await agent(
  `You are a skeptical completeness critic. An exhaustive audit of nirs4all's docs+examples was just produced. The segment list (what WAS covered) and the synthesis are below. The ground-truth card is at ${CARD}.

Your job: find what the audit MISSED or got wrong. Consider: doc areas or files not in any segment; audit dimensions skipped (accessibility, search, i18n FR/EN duplication, code-block runnability, intersphinx, screenshots/assets, notebook predictions.ipynb, versioned docs, contribution flow); findings that are asserted without source evidence and could be false; whether the style-mandate sweep was actually applied to every code block. Be concrete and adversarial. Your response IS data.

=== SEGMENTS COVERED ===
${SEGMENTS.map(s => s.key + ': ' + s.files.length + ' files').join('\n')}

=== SYNTHESIS (JSON) ===
${JSON.stringify(synthesis)}`,
  { schema: CRITIC_SCHEMA, phase: 'Critique', label: 'completeness-critic' }
)

return { segments: ok, synthesis, critic, counts: { requested: SEGMENTS.length, returned: ok.length } }
