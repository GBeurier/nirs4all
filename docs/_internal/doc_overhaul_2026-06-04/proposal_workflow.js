export const meta = {
  name: 'nirs4all-docs-restructuring-proposal',
  description: 'Design a best-in-class docs restructuring: 3 competing IA candidates → judge panel → synthesis → grounded page-by-page migration map',
  phases: [
    { title: 'Design', detail: '3 independent target-IA candidates from different philosophies' },
    { title: 'Judge', detail: 'panel scores all candidates on distinct lenses' },
    { title: 'Synthesize', detail: 'winning IA + grafts + style guide + safety-net + roadmap' },
    { title: 'MigrationMap', detail: 'every existing page assigned keep/merge/rewrite/delete, grounded in the real tree' },
  ],
}

const ROOT = '/home/delete/nirs4all/nirs4all'
const AUDIT = `${ROOT}/docs/_internal/doc_overhaul_2026-06-04/02_audit.md`
const CARD = `${ROOT}/docs/_internal/doc_overhaul_2026-06-04/GROUND_TRUTH.md`

const COMMON = `You are designing a best-in-class documentation architecture for **nirs4all**, a Python NIRS/spectroscopy ML library (v0.9.1, AGPL-3.0). The PUBLIC API is run/predict/explain/retrain/session/generate (+ result objects). The mandated code style is "1 pipeline written INLINE inside run(pipeline=[...], dataset=...), no external variable."

REQUIRED READING (Read these first, do not rely on memory):
- ${AUDIT}  — the exhaustive Phase-2 audit. Your design MUST solve the problems it documents (competing front doors getting_started/concepts/onboarding/ai_onboarding; api/ vs reference/ duplication; 648 inline-style violations; metadata drift; no doc safety-net; weekly-only examples CI; broken-ref suppression).
- ${CARD}  — the verified public API surface.
- The real tree: run \`find ${ROOT}/docs/source -name '*.md' -not -path '*/api/*'\` and \`find ${ROOT}/examples -name '*.py' -not -path '*/_internal/*'\` to see what actually exists (119 narrative md pages, ~80 example files, 357 autodoc stubs, 28 assets).

Benchmark against the gold standard of OSS library docs and name which patterns you borrow: Diátaxis (Tutorials / How-to / Reference / Explanation as distinct modes), FastAPI (tutorial-first, one concept per page, every snippet runnable), scikit-learn (User Guide + API Reference + executable examples gallery), Polars/Pydantic (modern mkdocs-material clarity), and a single unambiguous "Start here".`

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['philosophy','one_line_pitch','top_level_nav','front_door_resolution','section_design','examples_placement','reference_strategy','style_guide_stance','safety_net','pros','cons','newcomer_path'],
  properties: {
    philosophy: { type: 'string' },
    one_line_pitch: { type: 'string' },
    top_level_nav: { type: 'array', items: { type: 'string' }, description: 'ordered top-level sections of the new site' },
    front_door_resolution: { type: 'string', description: 'exactly how getting_started/concepts/onboarding/ai_onboarding competition is resolved into ONE canonical entry path' },
    section_design: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['section','purpose','contains'],
      properties: { section: { type: 'string' }, purpose: { type: 'string' }, contains: { type: 'string', description: 'which existing content lands here' } },
    } },
    examples_placement: { type: 'string', description: 'how examples/*.py relate to the docs (gallery? embedded? appendix?) and how the U/D/R taxonomy + missing R04 + weekly CI are handled' },
    reference_strategy: { type: 'string', description: 'how api/ (4 md + 357 stubs) and reference/ are unified into ONE reference home' },
    style_guide_stance: { type: 'string', description: 'how the inline-pipeline mandate is presented and enforced, incl. the documented exceptions' },
    safety_net: { type: 'string', description: 'doc-snippet execution, per-PR examples, nitpick tightening, metadata-from-source, CI grep gates' },
    pros: { type: 'array', items: { type: 'string' } },
    cons: { type: 'array', items: { type: 'string' } },
    newcomer_path: { type: 'string', description: 'the literal click-path a brand-new user takes from landing to first successful run() to second task' },
  },
}

phase('Design')
const CANDIDATES = [
  { key: 'A:strict-diataxis', philosophy: 'STRICT DIÁTAXIS — four clean, non-overlapping modes (Tutorials = learning-oriented; How-to = task-oriented; Reference = information-oriented; Explanation = understanding-oriented). Every page belongs to exactly one quadrant; cross-links between modes are explicit. Resolve every competing front door by assigning it to a single quadrant.' },
  { key: 'B:tutorial-first-fastapi', philosophy: 'TUTORIAL-FIRST (FastAPI model) — a single, long, strictly progressive guided tutorial is the spine of the site (each page adds exactly one concept, every snippet runnable and copy-pasteable, inline pipelines throughout). How-to recipes and a compact API reference are appendices. Optimize ruthlessly for time-to-first-success and a linear fil conducteur.' },
  { key: 'C:task-persona-first', philosophy: 'TASK/PERSONA-FIRST — organize by what users actually come to do (Regression, Classification, Preprocessing selection, Cross-validation, Deployment/serving, Transfer/retrain, Explainability), each a self-contained track from data→pipeline→run→predict. A thin shared "Concepts" layer underneath; reference auto-generated. Meets users at their goal, not at the architecture.' },
]
const designs = await parallel(CANDIDATES.map(c => () => agent(
  `${COMMON}\n\n=== YOUR ASSIGNED PHILOSOPHY ===\n${c.philosophy}\n\nDesign a COMPLETE target information architecture for nirs4all docs under this philosophy. Be concrete: name the top-level nav, show where the major existing content areas land, resolve the front-door competition explicitly, place the examples, unify the reference, and take a clear stance on the style mandate and the safety-net. Optimize for a newcomer reaching their first successful run() fast AND for long-term maintainability (single source of truth per concept). Return the structured object; your response IS data.`,
  { schema: DESIGN_SCHEMA, phase: 'Design', label: c.key }
)))
const validDesigns = designs.filter(Boolean)
log(`Designed ${validDesigns.length}/3 IA candidates`)

phase('Judge')
const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens','scores','ranking','best_ideas_to_graft','verdict'],
  properties: {
    lens: { type: 'string' },
    scores: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['candidate','score','rationale'],
      properties: { candidate: { type: 'string' }, score: { type: 'integer', minimum: 1, maximum: 10 }, rationale: { type: 'string' } },
    } },
    ranking: { type: 'array', items: { type: 'string' }, description: 'candidate keys best-to-worst on this lens' },
    best_ideas_to_graft: { type: 'array', items: { type: 'string' }, description: 'specific ideas worth taking from any candidate regardless of overall winner' },
    verdict: { type: 'string' },
  },
}
const LENSES = [
  'NEWCOMER ONBOARDING — how fast and unambiguously does a brand-new user get from landing to first successful run() and then to their second task? Is there one obvious path?',
  'MAINTAINABILITY & SINGLE-SOURCE-OF-TRUTH — does it eliminate the competing front doors and api/↔reference/ duplication, and is each concept documented in exactly one canonical place? How much ongoing drift will it allow?',
  'FIT TO nirs4all\'S ACTUAL SURFACE — does it match the real API (run/predict/explain/retrain/session/generate), the pipeline/generator/branching feature set, and the examples-as-tests reality, without inventing structure the library does not support?',
  'ENFORCEABILITY & SAFETY-NET — how well does it operationalize the inline-style mandate and the CI/doc-execution safety net so the rewrite stays correct?',
]
const judgments = await parallel(LENSES.map((lens, i) => () => agent(
  `${COMMON}\n\nYou are JUDGE #${i + 1}. Evaluate ALL THREE candidate documentation architectures below ONLY through this lens:\n\n=== LENS ===\n${lens}\n\nScore each candidate 1-10 on this lens with specific rationale grounded in the audit's problems. Rank them. Then, independent of the ranking, list the single best ideas worth grafting from ANY candidate. Be critical and concrete. Your response IS data.\n\n=== CANDIDATES (JSON) ===\n${JSON.stringify(validDesigns)}`,
  { schema: JUDGE_SCHEMA, phase: 'Judge', label: `judge:lens${i + 1}` }
)))
const validJudgments = judgments.filter(Boolean)

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['chosen_architecture','rationale','target_nav_tree','front_door','grafted_ideas','style_guide','examples_plan','reference_plan','safety_net_plan','execution_roadmap','success_metrics'],
  properties: {
    chosen_architecture: { type: 'string', description: 'winner (may be a hybrid) + why it won the panel' },
    rationale: { type: 'string' },
    target_nav_tree: { type: 'string', description: 'the full proposed nav as an indented tree, top-level sections down to page level' },
    front_door: { type: 'string', description: 'the single canonical Start-here path, concretely' },
    grafted_ideas: { type: 'array', items: { type: 'string' } },
    style_guide: { type: 'string', description: 'the inline-pipeline style guide: the rule, the canonical example, the documented exceptions, and the CI lint that enforces it (with the exact pattern to flag)' },
    examples_plan: { type: 'string', description: 'reorganized examples taxonomy, relation to docs (gallery/embed), R04 fix, weekly→per-PR CI, doctest of doc snippets' },
    reference_plan: { type: 'string', description: 'unify api/ + reference/ into one home; autodoc strategy; metadata-from-source' },
    safety_net_plan: { type: 'string', description: 'concrete CI gates: doc-snippet execution, version/license grep, nitpick tightening, RTD/CI fail_on_warning alignment' },
    execution_roadmap: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['phase','goal','tasks','effort','depends_on'],
      properties: { phase: { type: 'string' }, goal: { type: 'string' }, tasks: { type: 'array', items: { type: 'string' } }, effort: { type: 'string' }, depends_on: { type: 'string' } },
    }, description: 'ordered phases; quick wins (metadata, broken API, style sweep) before structural IA migration' },
    success_metrics: { type: 'array', items: { type: 'string' }, description: 'measurable definition of done (e.g. 0 style violations, 0 CeCILL tokens, mean health >=4, all snippets execute in CI)' },
  },
}
const synthesis = await agent(
  `${COMMON}\n\nYou are the lead documentation architect. The panel has scored 3 candidate architectures. Synthesize the WINNING architecture (a hybrid is allowed and often best — e.g. a Diátaxis backbone with a FastAPI-style progressive tutorial as the Tutorials arm and persona-named how-to tracks). Graft the best ideas the judges flagged. Produce a complete, actionable restructuring blueprint: the target nav tree, the single front door, the enforceable style guide, the examples plan, the unified reference plan, the safety-net CI plan, and a phased execution roadmap that front-loads the audit's quick wins before the structural migration. Define success metrics. Ground everything in the audit. Your response IS data.\n\n=== CANDIDATES (JSON) ===\n${JSON.stringify(validDesigns)}\n\n=== JUDGE PANEL (JSON) ===\n${JSON.stringify(validJudgments)}`,
  { schema: SYNTH_SCHEMA, phase: 'Synthesize', label: 'synthesis' }
)

phase('MigrationMap')
const MIGRATION_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['dispositions','deletions','merges','new_pages','summary'],
  properties: {
    dispositions: { type: 'array', items: {
      type: 'object', additionalProperties: false, required: ['current_path','disposition','target','note'],
      properties: {
        current_path: { type: 'string' },
        disposition: { type: 'string', enum: ['KEEP','MOVE','MERGE','REWRITE','SPLIT','DELETE','MOVE_TO_INTERNAL'] },
        target: { type: 'string', description: 'destination section/page in the new IA (or reason for delete)' },
        note: { type: 'string' },
      },
    }, description: 'one row per existing docs page / example group — be comprehensive, cover every top-level area' },
    deletions: { type: 'array', items: { type: 'string' }, description: 'pages/files to remove and why' },
    merges: { type: 'array', items: { type: 'string' }, description: 'sets of pages collapsing into one canonical page' },
    new_pages: { type: 'array', items: { type: 'string' }, description: 'pages that must be authored fresh to fill gaps the audit found' },
    summary: { type: 'string' },
  },
}
const migration = await agent(
  `${COMMON}\n\nYou are producing the PAGE-BY-PAGE MIGRATION MAP that turns today's tree into the chosen target IA below. Walk the REAL tree (run \`find ${ROOT}/docs/source -name '*.md'\` including the api/ pages, and \`find ${ROOT}/examples -name '*.py'\`) so you cover every existing page — do NOT work from memory. Assign each page (or coherent group, e.g. "all of user_guide/preprocessing/*") a disposition: KEEP / MOVE / MERGE / REWRITE / SPLIT / DELETE / MOVE_TO_INTERNAL, with its destination in the new IA and a one-line note. Be comprehensive across ALL areas: getting_started, concepts, onboarding, ai_onboarding, user_guide/*, reference/*, api/*, developer/*, migration/*, examples/* (incl. predictions.ipynb), and root meta. Fold in the audit's specific fixes (merge api/storage↔reference/storage, delete/convert predictions.ipynb, kill competing front doors, move Phase/roadmap leakage to _internal). List deletions, merges, and brand-new pages to author. Your response IS data.\n\n=== CHOSEN TARGET IA (JSON) ===\n${JSON.stringify(synthesis)}`,
  { schema: MIGRATION_SCHEMA, phase: 'MigrationMap', label: 'migration-map' }
)

return { designs: validDesigns, judgments: validJudgments, synthesis, migration }
