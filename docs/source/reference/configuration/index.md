# Configuration Reference

This section documents configuration objects used to customize nirs4all behavior.

## Available Configurations

::::{grid} 2
:gutter: 3

:::{grid-item-card} CacheConfig
:link: cache_config
:link-type: doc

Configure step-level caching and memory optimization for pipeline execution.
:::

::::

## Usage Pattern

Configuration objects are passed to API functions like `nirs4all.run()`:

```python
from nirs4all.config.cache_config import CacheConfig

result = nirs4all.run(
    pipeline=pipeline,
    dataset=dataset,
    cache=CacheConfig(step_cache_enabled=True),
)
```

Most configurations are optional and use sensible defaults.

## See Also

- {doc}`/user_guide/pipelines/cache_optimization` - Cache optimization guide
- {doc}`/reference/pipeline_syntax` - Pipeline syntax reference
