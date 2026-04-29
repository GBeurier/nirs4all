# Adapters

Bench adapters live here.

Planned modules:

- `prior_adapter.py`: `PriorSampler` output to canonical validated records;
- `builder_adapter.py`: canonical prior records to dataset runs;
- `fitted_config_adapter.py`: `RealDataFitter.to_full_config()` to executable
  bench configs.

Adapters should document unsupported production fields instead of dropping them
silently.

