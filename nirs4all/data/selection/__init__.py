"""Sample subset selection utilities.

Index-level helpers for picking a representative subset of samples from a
dataset (e.g. for real-time previews and lightweight visualisations). They
operate purely on sample *indices* and never copy or mutate the underlying
spectra — the caller subsets its own arrays with the returned indices.
"""

from .sampling import kmeans_sample, random_sample, stratified_sample

__all__ = ["kmeans_sample", "random_sample", "stratified_sample"]
