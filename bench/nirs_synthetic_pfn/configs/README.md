# Configs

Experiment and preset configs will live here once implementation starts.

Expected layout:

```text
configs/
  presets/
    grain_benchtop.yaml
    dairy_liquid_transmittance.yaml
    tablet_transflectance.yaml
  experiments/
    smoke_prior.yaml
    prior_scorecards.yaml
    encoder_tabpfn.yaml
```

Every config must include:

- seed;
- prior preset or prior distribution;
- dataset/task size;
- validation suite;
- output report path.

