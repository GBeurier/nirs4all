"""FastAOM model family: chain-based PLS-Ridge and multi-kernel Ridge variants."""

from __future__ import annotations

from .single_chain_pls_ridge import SingleChainPLSRidge
from .hard_aom_chain_pls_ridge import HardAOMChainPLSRidge
from .sparse_multi_kernel_ridge import SparseMultiKernelRidge
from .soft_aom_chain_pls_ridge import SoftAOMChainPLSRidge
from .fast_aom_pls_ridge import FastAOMConfig, FastAOMPLSRidge

__all__ = [
    "SingleChainPLSRidge",
    "HardAOMChainPLSRidge",
    "SparseMultiKernelRidge",
    "SoftAOMChainPLSRidge",
    "FastAOMConfig",
    "FastAOMPLSRidge",
]
