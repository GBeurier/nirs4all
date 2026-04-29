# Tests

Bench-only tests will live here.

Minimum intended command:

```bash
PYTHONPATH=bench/nirs_synthetic_pfn/src pytest bench/nirs_synthetic_pfn/tests -q
```

Initial test targets:

- prior canonicalization;
- prior-to-dataset adapter;
- task contract;
- multi-view latent ids;
- report schema;
- deterministic seeds.

