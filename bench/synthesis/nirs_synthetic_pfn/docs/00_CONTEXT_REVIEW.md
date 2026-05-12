# Context Review

Date: 2026-04-28

Reviewed input:

- `docs/_internal/synthetic/spectral_synthesis_inventory_and_pfn_prior_plan.md`

## Executive Assessment

The reviewed document is technically sound. The main conclusion is correct:
`nirs4all` already contains many strong synthesis blocks, but the blocks are not
yet connected as one scientific workflow. The limiting problem is integration
and validation, not the absence of physical effects.

The right operating model is to develop in `bench`, validate the method there,
then port only stable, minimal APIs into the library. That avoids turning
research assumptions into production contracts too early.

## Current Strengths

The existing package already has the core assets needed for the project:

- `SyntheticNIRSGenerator` with physical effects, instruments, detectors,
  baseline, scatter, noise, batch effects, environment, particle size, and edge
  artifacts;
- `SyntheticDatasetBuilder` with regression, classification, nonlinear targets,
  confounders, batches, partitions, products, aggregates, and multi-source data;
- component, domain, product, aggregate, instrument, detector, and benchmark
  registries;
- reconstruction experiments that expose candidate latent variables;
- validation functions including scorecards and adversarial real/synthetic tests;
- TabPFN and spectral latent feature experiments in `bench`.

This is enough to build a serious bench research line.

## Blocking Gaps

The important gaps are interface and contract gaps:

- `PriorSampler` samples a configuration, not an executable dataset or task.
- Domain names in the prior do not align cleanly with `APPLICATION_DOMAINS`.
- `measurement_mode` exists conceptually, but the main generator path does not
  yet make it a first-class physical branch.
- `SyntheticDatasetBuilder.fit_to()` and `generate.from_template()` use only a
  shallow subset of `RealDataFitter` output.
- Advanced environment/scattering/edge effects are not fully exposed at builder
  or task level.
- Supervised augmentation assumes label preservation; mixup-style operations can
  violate that contract.
- There is no canonical latent object, no multi-view rendering workflow, and no
  NIRS task object with context/query splits.

These gaps should be handled in bench first with adapters and tests, then moved
into `nirs4all` after validation.

## Strategic Decisions

1. Development happens in `bench/nirs_synthetic_pfn`.
2. The production package is read-only reference until integration gates pass.
3. The first milestone is an executable prior, not a neural model.
4. The first model proof is encoder plus TabPFN, not full NIRS-PFN.
5. Every phase must produce validation reports, not only code.
6. Full NIRS-ICL/PFN work starts only after the synthetic prior improves real
   downstream metrics or few-shot transfer.

## Scientific Risks

The main scientific risk is learning false invariances from synthetic data. A
generator can produce visually plausible spectra while teaching a model that
reflectance, transmittance, matrix effects, or target noise behave incorrectly.

The second risk is target triviality. If labels are always a direct component
concentration, models can succeed on synthetic tasks while failing on real NIRS
tasks where reference error, confounding, regimes, and analyte visibility matter.

The third risk is latent non-identifiability. A common latent representation is
valuable only if same-latent multi-view samples are verifiable and if the latent
keeps analyte information while removing nuisance information.

## Recommended Direction

The project should follow this order:

1. make prior samples executable and valid;
2. create standard metadata and task contracts;
3. run prior predictive checks and real/synthetic scorecards;
4. generate multi-view batches from common latents;
5. train or evaluate a spectral encoder;
6. compare encoder plus TabPFN with current PCA-based baselines;
7. only then design a full NIRS-ICL/PFN task prior.

