"""nicon_v2 — research bench for an improved 1-D CNN over `nirs4all` `nicon`/`decon`.

This package is intentionally narrow in scope:
    * dataset loading from the AOM-Ridge / TabPFN paper cohort,
    * baseline training (Ridge / PLS / nicon / decon),
    * the V1+ improved models added phase by phase.

Nothing here mutates the `nirs4all` library; we only import from it.
"""

__version__ = "0.0.1"
CODE_VERSION = "nicon_v2/0.0.1"
