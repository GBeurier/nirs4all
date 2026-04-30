# Synthetic Realism Remediation Roadmap

Date: 2026-04-29
Scope: bench-only documentation. No code, no model training authorization, no production integration, no change under `nirs4all/`.

## Decision Snapshot
Status: `NO-GO`.

Audits synthesized:
- `reports/real_synthetic_scorecards.md`
- `reports/adversarial_auc.md`
- `reports/transfer_validation.md`
- `reports/minimal_ablation_attribution.md`
- `reports/encoder_tabpfn_gate.md`
- `reports/nirs_icl_gate_precheck.md`
- `reports/integration_gate_status.md`
- `docs/04_SCIENTIFIC_VALIDATION_PROTOCOL.md`
- `docs/05_INTEGRATION_GATE.md`

Current evidence:
- B2 is not a realism pass; thresholds are provisional smoke gates.
- B3 standalone adversarial AUC audit is `NO-GO`.
- Raw authoritative lane (`uncalibrated_raw`): 71 compared rows, 71 AUC smoke failures, 71 stretch failures, 6 blocked evidence gaps, 0 missing AUC rows.
- Named raw high-AUC sentinel failures: BEER, DIESEL, CORN.
- Secondary sentinel pressure: MILK, soil-like matrices, fruit-like matrices.
- Evidence gaps: COLZA, TABLET, WOOD wavelength support/overlap.
- B4 is `BLOCKED_BY_REALISM_GATE`: 0 synthetic samples, 0 fitted models, 0 real-only baselines, 0 TSTR/RTSR routes.
- B5 is `BLOCKED_REPORT_ONLY`; it summarizes failure groups but is not causal evidence.
- C is `BLOCKED_BY_UPSTREAM_REALISM_GATE`; no encoder, TabPFN, checkpoint, or benchmark.
- D is `BLOCKED_BY_UPSTREAM_REALISM_GATE`; no task sampling, ICL baseline, PFN training, or benchmark.
- Integration gate remains `NO-GO`.

## Scientific Position
Priority is the mechanistic generator. The synthetic prior must become harder to separate from real spectra because chemistry, optical mode, scattering, path length, targets, and instrument artifacts are more realistic.

Calibration is diagnostic only. Marginal calibration, SNV, robust affine scaling, quantile mapping, and high-pass residual scaling cannot prove realism while raw spectra fail.

Machine learning and deep learning are forbidden until mechanistic gates are exhausted and the blocker is scientifically documented. No encoder, TabPFN, PFN, ICL task generation, learned residual model, or neural generator may be used as a cache-misere for failed raw realism.

Strict rule: improve mechanisms first; evaluate raw first; use statistical or learned lanes only after documented escalation; reopen B4 only from the best validated and documented tier; reopen C/D only after B4 and attribution are defensible; do not integrate.

## Tiered Remediation Strategy
The remediation strategy is progressive. A higher tier is not a shortcut around a lower tier. It is an auditable fallback only when the lower tier has reached a documented scientific blocker under frozen gates.

### Palier 1: `uncalibrated_raw`
Goal: progress as far as possible with a pure mechanistic generator.

Allowed:
- component banks, chemical priors, optical modes, scattering/path-length coupling, instrument physics, bounded drift/noise, target observation policy, and non-oracle metadata.
- real-vs-real baselines, repeated seeds, robust adversaries, and failure attribution.

Forbidden as proof:
- fitting to real marginal statistics, PCA, covariance, PSD, correlation length, noise manifolds, or any learned residual/noise model to pass B2/B3.
- label, split, target, downstream performance, row identity, or dataset fingerprint information.
- claiming a pass from calibrated, transformed, or statistic-matched spectra while `uncalibrated_raw` fails.

Exit condition:
- Stay in Palier 1 until either B2/B3 pass in `uncalibrated_raw`, or a written blocker shows that the remaining gap is plausibly non-mechanistic/noise-structure limited after R1-R9 evidence is complete.

### Palier 2: `mechanistic_statistical_noise`
Goal: if Palier 1 is scientifically blocked, add non-supervised capture of real noise/structure while keeping the generator mechanistic.

Allowed only in a separate lane:
- PCA/covariance structure, correlation length, PSD, instrument residual structure, noise manifolds, and other unsupervised structure summaries.
- summaries computed without labels, splits, performance metrics, target values, row identity, or B4/C/D feedback.
- explicit manifests recording source data, preprocessing, summary type, fitted degrees of freedom, random seed, and non-oracle audit.

Forbidden:
- using labels, train/test splits, target leakage, downstream scores, row matching, real spectrum replay, or any B4/C/D result to tune the noise/structure capture.
- replacing the `uncalibrated_raw` result with a Palier 2 result without labeling the lane and the escalation reason.

Exit condition:
- Palier 2 may support B4/C/D only if its lane passes frozen B2/B3 evidence, its non-oracle audit passes, and Palier 1 blocker documentation remains attached.
- If Palier 2 fails after documented attempts across sentinels and cohort gates, escalation to Palier 3 requires a separate scientific decision memo.

### Palier 3: `hybrid_ml_dl_diagnostic`
Goal: if Palier 2 is truly blocked, evaluate hybrid ML/DL approaches as audited diagnostics, never as automatic integration.

Allowed only after Palier 2 blockage:
- hybrid residuals, neural noise models, representation models, or generator components with explicit degrees of mechanistic, statistical, and learned contribution.
- per-artifact reporting of learned inputs, labels excluded, split/performance exclusions, training corpus, capacity, seed variance, and failure modes.

Forbidden:
- automatic production integration.
- hiding a learned realism substitute behind a mechanistic label.
- using B4/C/D performance to tune realism generation or to justify B2/B3 pass.

Exit condition:
- Palier 3 can only produce a human-reviewed diagnostic candidate. It does not authorize integration, and it does not erase the need to report the highest lower-tier evidence that failed.

## Tier Passage Conditions And Anti-Goodhart Rules
Passage from one tier to the next requires:
- frozen B2/B3 thresholds and schemas from R0;
- completed evidence-gap handling for the affected rows;
- sentinel-first failure notes for BEER, DIESEL, CORN, plus MILK/soil/fruit regression status where relevant;
- written blocker explaining why additional lower-tier mechanistic work is not the next scientifically reasonable step;
- non-oracle audit proving no labels, splits, target values, downstream performance, row identity, or dataset fingerprints were used;
- lane manifest naming the active lane exactly as `uncalibrated_raw`, `mechanistic_statistical_noise`, or `hybrid_ml_dl_diagnostic`.

Anti-Goodhart controls:
- Do not tune to one scalar AUC while ignoring coverage, blockers, seed instability, real-vs-real separability, adversary variation, or stretch failures.
- Do not move thresholds in the artifact that claims improvement.
- Do not promote a lane because it performs well downstream if it fails the appropriate realism gate.
- Do not hide blocked rows behind improved compared-row averages.
- Do not use B4/C/D feedback to select B2/B3 generator variants.
- Every claim must state the best validated tier, the failed lower-tier evidence, and the lane that produced the claim.

Downstream rule:
- B4, C, and D may rely only on the best tier that is both validated and documented at the time of the run.
- If only Palier 1 is validated, B4/C/D use `uncalibrated_raw` evidence only.
- If Palier 2 is validated, B4/C/D may use `mechanistic_statistical_noise` evidence only with the Palier 1 blocker and Palier 2 non-oracle audit attached.
- If Palier 3 exists, B4/C/D treat it as human-reviewed diagnostic evidence, not an automatic integration path.

## Evaluation Lanes To Lock
1. `uncalibrated_raw` is the primary gate.
   - Raw adversarial AUC smoke threshold remains `<= 0.85`.
   - Raw stretch target remains `<= 0.70`.
   - Any unresolved raw blocker fails the gate.
2. `mechanistic_statistical_noise` is the Palier 2 lane.
   - It is separate from `uncalibrated_raw`.
   - It may use unsupervised PCA/covariance/correlation length/PSD/noise-manifold summaries only after Palier 1 blockage is documented.
   - It cannot use labels, splits, target values, downstream performance, row identity, or oracle source information.
   - It cannot override a failed `uncalibrated_raw` result without the explicit Palier 1 blocker and lane label.
3. `hybrid_ml_dl_diagnostic` is the Palier 3 lane.
   - It is separate from both lower tiers.
   - It is allowed only after documented Palier 2 blockage.
   - It must report explicit mechanistic, statistical, and learned degrees for each artifact.
   - It remains diagnostic and human-reviewed; it never authorizes automatic integration.
4. Calibrated diagnostics remain second-order evidence, not a lane that can pass the gate.
   - SNV and marginal calibration may explain failure modes.
   - They cannot override raw AUC failures, missing AUC, blocked rows, or tier passage requirements.
   - Tables must label these outputs as diagnostic.
5. Manifest and coverage are mandatory.
   - Every row records dataset, source, task, preset, wavelength support, mode, component family, target policy, seed, and blocker reason.
   - Every row records tier and lane when Palier 2 or Palier 3 evidence exists.
   - Reports expose attempted rows, compared rows, blocked rows, missing AUC rows, and reason counts.
6. Real-vs-real baseline is required.
   - Estimate real-vs-real separability under matched sample count, split policy, instrument/domain grouping, and preprocessing.
   - Interpret real/synthetic AUC against this baseline.
7. Repeated seeds are required.
   - No pass on a single seed.
   - Sentinel and cohort reports must include seed variance.
   - Seed instability is a failure mode.
8. Robust adversaries are required.
   - Keep the existing AUC route and add checks across adversary families, wavelength windows, derivatives, sample caps, and feature subsets.
   - A pass must survive adversary variation without threshold loosening.

## R0. Freeze Gates And Lanes
Goal: prevent moving targets before remediation.

Deliverables:
- Lane manifest with `uncalibrated_raw` authoritative.
- Frozen B2/B3 schema separating `uncalibrated_raw`, `mechanistic_statistical_noise`, `hybrid_ml_dl_diagnostic`, calibrated diagnostics, real-vs-real, repeated-seed, and robust-adversary results.
- Blocker taxonomy: high AUC, wavelength unknown, wavelength overlap, mode mismatch, path-length mismatch, target leakage risk, oracle risk.
- Dashboard carrying 71/71 smoke failures, 71/71 stretch failures, and 6 evidence gaps.

Gate:
- B2/B3 fail closed if lane, coverage, seed, or blocker metadata are absent.
- No threshold change in the same artifact that claims improvement.
- No Palier 2 or Palier 3 result is considered unless the lower-tier blocker memo and non-oracle audit are present.

## R1. Close Wavelength Evidence Gaps
Goal: resolve COLZA, TABLET, and WOOD before claiming cohort progress.

Deliverables:
- COLZA memo for `C_woOutlier`, `N_wOutlier`, `N_woOutlier`.
- TABLET memo for `Escitalopramt_310_Zhao`.
- WOOD memo for `WOOD_Density_402_Olale`, `WOOD_N_402_Olale`.
- Per-row decision: physical grid recovered, valid overlap established, or scientifically excluded.

Gate:
- The 6 unresolved raw evidence gaps become 0.
- No `np.arange` index fallback is accepted as physical wavelength evidence.
- No cross-domain remap is allowed only to manufacture overlap.

## R2. Sentinel-First Remediation
Goal: fix named failures before broad tuning.

Deliverables:
- BEER, DIESEL, CORN scorecards with raw, diagnostic, real-vs-real, repeated-seed, and robust-adversary columns.
- Secondary sentinel scorecards for MILK, soil-like, and fruit-like matrices.
- Failure notes mapping each sentinel to component, optical mode, scatter/path length, noise/drift, or target-policy causes.
- For R2h/BERRY only: provenance must state that `cloudy_berry_percent_transmittance_readout` uses fixed cloudy-berry optical prior constants for apparent percent-transmittance/intensity raw readouts. It must explicitly record no real-stat capture, no PCA/covariance/quantile/marginal capture, no threshold calibration from sentinels, and no downstream feedback.

Gate:
- BEER, DIESEL, and CORN must stop being named raw smoke failures before broad cohort claims.
- MILK, soil, and fruit become regression sentinels before B4 reopens.
- A BERRY percent/intensity readout is not a gate pass by itself. Its amplitude remains a validation risk until repeated seeds, robust adversaries, and real-vs-real context are attached.

## R3. Enrich Component Bank And Presets
Goal: replace generic approximations with plausible matrix chemistry.

Deliverables:
- Component families for water, ethanol, sugars, starch, proteins, fats, hydrocarbons, minerals, cellulose/lignin, APIs, excipients, soil organic/mineral fractions.
- Presets for beverage, diesel/fuel, corn/grain/oil/starch, milk/dairy, soil/mineral, fruit/plant tissue, wood/cellulose, and tablets.
- Per-preset wavelength range, optical mode support, dynamic range, nuisance variables, and negative controls.

Gate:
- Generic fallback components are rare, explicit, and reported.
- Presets pass prior checks before B2 scoring.
- Improvements appear in `uncalibrated_raw`, not only after calibration.

## R4. Add Real Optical Modes
Goal: separate reflectance, transmittance, transflectance, and ATR instead of using one abstract spectrum type.

Deliverables:
- Mode metadata and generator branches for reflectance/transmittance/transflectance/ATR.
- Mode-specific wavelength ranges, transforms, saturation, baseline shapes, detector assumptions, and preprocessing validity.
- Dataset-to-mode audit for BEER, DIESEL, CORN, MILK, soil, fruit, TABLET, and WOOD.
- For raw percent/intensity datasets, document whether generated absorbance is being exposed through an apparent transmission/intensity readout. This is an optical readout hypothesis, not PCA, real-fit calibration, or statistical matching.

Gate:
- B2 rows record mode and mode confidence.
- Mode-unknown rows stay diagnostic unless a defensible default is documented.
- Mode-specific sentinels improve in raw lane without source-oracle shortcuts.

## R5. Couple Scattering And Path Length
Goal: make scatter, particle size, turbidity, packing, and path length co-vary with matrix and mode physics.

Deliverables:
- Coupled scatter/path-length parameters per matrix family and optical mode.
- Mechanisms for multiplicative scatter, additive baseline, particle curvature, effective path length, saturation, and wavelength-dependent attenuation.
- Sentinel ablations: scatter-only, path-length-only, and coupled variants.

Gate:
- Coupled variants reduce raw separability on sentinels.
- Derivatives, baseline curvature, SNR, PCA overlap, and nearest-neighbor ratio do not become new blockers.

## R6. Model Instrument Effects
Goal: represent instruments as structured measurement processes.

Deliverables:
- Instrument profiles for resolution, wavelength jitter, detector response, shot/read noise, correlated noise, stray light, dark current, lamp drift, temperature drift, and edge effects.
- Mode-aware bounds and seed metadata for each profile.
- Robust adversary analysis separating chemical shape from instrument fingerprints.

Gate:
- Noise is not white-noise decoration.
- Stray light and drift are physically bounded.
- Repeated-seed variance remains reported and acceptable.

## R7. Use Realistic Non-Oracle Targets
Goal: stop synthetic labels from being perfect latent readouts.

Deliverables:
- Target policies for noisy lab references, replicate uncertainty, censoring, discretization, class imbalance, batch effects, and correlated analytes.
- Metadata separating latent composition from observed target.
- Audit proving no real y values, labels, splits, target metrics, or performance metrics are used.

Gate:
- No target route replays or infers real labels.
- B4 target matching cannot start until target policy is documented for the relevant sentinel/cohort family.

## R8. Run Mechanistic Ablations Exp08
Goal: identify which mechanistic changes move raw realism.

Deliverables:
- `exp08` design memo before implementation.
- Ablation matrix for component bank, optical modes, scatter/path coupling, instrument effects, noise/drift/stray light, and target policy.
- Sentinel-first results followed by cohort-scale results.

Gate:
- Ablations use frozen R0 gates.
- No threshold loosening, cross-domain remap, or calibrated-lane pass criteria.
- Claims distinguish association from causality.

## R9. Reopen B2/B3
Goal: regenerate realism evidence only after mechanisms and evidence gaps are addressed.

Deliverables:
- New B2 scorecards with all lanes.
- New B3 adversarial AUC audit consuming the new B2 CSV.
- Real-vs-real baseline, repeated-seed variance, robust-adversary report, and sentinel appendix.

Gate:
- Raw smoke failures: 0.
- Raw unresolved evidence gaps: 0.
- Raw missing AUC rows: 0.
- Stretch failures are resolved or explicitly accepted for a human-reviewed B4 reopening decision.
- Calibrated diagnostics cannot override raw failures.
- Palier 2 evidence is eligible only with documented Palier 1 blockage and a clean non-oracle audit.
- Palier 3 evidence is eligible only with documented Palier 2 blockage and explicit mechanistic/statistical/learned contribution accounting.

## R10. Reopen B4 Then C/D, No Integration
Goal: test synthetic usefulness only after B2/B3 are green.

Deliverables:
- Gate-first B4 transfer validation reading passing B2/B3 artifacts from the best validated and documented tier.
- Real-only baselines, TSTR/RTSR only where target policy and domain matching are valid, repeated splits or confidence intervals, and negative-result reporting.
- Regenerated B5 attribution after B4 is unblocked.
- Human-reviewed C and D prechecks after upstream gates pass.

Gate:
- No B4 run if B2/B3 are not green.
- B4, C, and D may use only the best validated and documented tier: `uncalibrated_raw`, then `mechanistic_statistical_noise` if Palier 1 is blocked, then `hybrid_ml_dl_diagnostic` only as audited diagnostic evidence if Palier 2 is blocked.
- No usefulness claim without real-only baseline comparison.
- No C/D training, task sampling, PFN, ICL, TabPFN, checkpoint, or benchmark until B4/B5 make upstream gates defensible.
- No production integration in this roadmap.

## Anti-Patterns
Rejected patterns:
- Calibration marginale comme preuve de realisme.
- SNV or calibrated diagnostics passing while raw fails.
- Threshold loosening to convert failure into pass.
- Moving thresholds and claiming improvement in the same artifact.
- Goodhart tuning to a single AUC while coverage, blockers, seed variance, real-vs-real baseline, or robust adversaries fail.
- Cross-domain remap for wavelength overlap.
- Stable-hash fallback as scientific matrix matching.
- Deep learning as cache-misere.
- Learned residuals, encoders, PFNs, ICL, or TabPFN as realism substitutes.
- PCA, covariance, PSD, correlation length, or noise manifolds used to pass the gate without Palier 1 blockage, separate lane, and non-oracle audit.
- Hybrid ML/DL reported without explicit mechanistic/statistical/learned degrees.
- Oracle sources: real labels, y values, splits, target metrics, performance metrics, row identity.
- Replaying real spectra, real marginals, or dataset fingerprints.
- Ignoring blocked rows because compared-row averages improved.
- Reporting top-line AUC without coverage and blocker taxonomy.

## Required Ordering
1. Freeze lanes/gates.
2. Close COLZA/TABLET/WOOD evidence gaps.
3. Fix BEER/DIESEL/CORN sentinels.
4. Lock MILK/soil/fruit regression sentinels.
5. Enrich component bank and presets.
6. Add optical modes.
7. Couple scattering and path length.
8. Add realistic instrument noise, stray light, and drift.
9. Add realistic non-oracle targets.
10. Run mechanistic `exp08` ablations.
11. Reopen B2/B3 in `uncalibrated_raw`.
12. Escalate to `mechanistic_statistical_noise` only if Palier 1 is scientifically blocked and documented.
13. Escalate to `hybrid_ml_dl_diagnostic` only if Palier 2 is scientifically blocked and documented.
14. Reopen B4 only from the best validated and documented tier.
15. Reopen C/D only after upstream gates pass and B4/B5 are defensible.
16. Keep integration closed.

## Definition Of Done
Done means:
- all raw evidence gaps resolved or scientifically excluded;
- BEER, DIESEL, and CORN no longer named raw smoke failures;
- MILK, soil, and fruit tracked as regression sentinels;
- raw adversarial AUC smoke gate passes on audited cohort;
- real-vs-real, repeated-seed, and robust-adversary evidence are attached;
- B4 reopens only from green B2/B3 evidence;
- C/D remain gate-first and human-reviewed;
- B4/C/D depend only on the best validated and documented tier;
- Palier 2 and Palier 3, if used, have lower-tier blocker memos, separate lane labels, and non-oracle audits;
- no production integration is performed by this remediation track.
