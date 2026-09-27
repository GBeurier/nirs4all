"""Cross-language trained pipelines of n4m role steps (envelope version 8).

The envelope carries a portable recipe whose steps are generic n4m role
tokens (``"n4m:<catalog method id>"``, see
:mod:`nirs4all.pipeline.config.component_serialization`) and, for every
fitted step, its native N4ME state. Any n4m binding (Python, R, JS/WASM,
Rust) rebuilds the same pipeline from those bytes and predicts identically;
numerics stay in Methods.

The implementation is ``nirs4all_core.N4mRolePipeline``, a wrapper over the
native Methods role pipeline: recipe validation (sample filters, transformers
and selectors, then one regressor or classifier), multi-target routing, the
column-name check and the recipe/state consistency check are native. This
module keeps nirs4all's public names.
"""

from nirs4all_core import N4M_TRAINED_PIPELINE_SCHEMA as SCHEMA
from nirs4all_core import N4mRolePipeline as PortableN4MRolePipeline

__all__ = ["SCHEMA", "PortableN4MRolePipeline"]
