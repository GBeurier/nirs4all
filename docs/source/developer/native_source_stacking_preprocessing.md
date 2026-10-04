# Native source stacking with upstream preprocessing

Ordinary X transformations before `branch.by_source` are applied separately to
sources, in declaration order, before each source's branch encoders and head.
Both bare instantiated transforms and `{"preprocessing": transform}` are
accepted when reconstructible and, for stochastic operators, explicitly seeded.
A splitter may occur before or after these declarations. The transformations
are never fitted before splitting: their real learned state belongs to each
native training scope.

For `zero_with_indicator`, each base chain fits only rows where its source is
present and at least one target is observed. `per_target` regression clones and
fits the **whole** upstream/branch chain separately for each observed target.
Complete mono-target classification retains genuine full-vocabulary
probabilities. Native group constraints, inner OOF, outer scoring and REFIT
remain authoritative; absent observations do not enter an encoder. No synthetic
probability distribution is created for an absent source.

Whole-stack search paths `branches.<source>.<step_index>.<parameter>` address the
expanded per-source chain, starting with the upstream transforms. Existing paths
are unchanged when there is no upstream declaration. Different sources have
independent cloned operators and may choose different parameters. The selected
public recipe moves the prefix inside each source body exactly once. Constructor
parameters are part of the signed native graph and the true learned encoder/head
chain is part of each original REFIT capture and checksum. Export/load/replay
reuse those fitted chains and perform no FIT.

Target transforms, global `fit_on_all`/transfer preprocessing, structured joins,
augmentation, layout changes and unseeded stochastic preprocessing remain
explicitly outside this prefix path. Mixed per-source task types and additional
missing-source policies remain open MM05/MM08 work; this source change does not
mark the entire MM05 backlog complete. Experimental-unit weighting additionally
requires the existing exact `sample_weight` capability admission for every real
encoder and head; no weights are fabricated.

Source witnesses cover grouped nested native scopes against independent sklearn
fits, whole-stack search and installed cold replay for regression, Unicode and
non-contiguous int64 classes, including an entirely absent prediction source.
These witnesses are authored for the final qualification and have not run during
source authoring. Installed replay must set
`NIRS4ALL_PARTIAL_LATE_INSTALLED_PYTHON` and
`NIRS4ALL_REQUIRE_PARTIAL_LATE_INSTALLED=1` for a mandatory gate.
