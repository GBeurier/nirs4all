"""Sample strategy for statistical sampling.

This module handles _sample_ nodes that generate values using statistical
distributions - useful for random hyperparameter search.

Syntax:
    {"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 10}}
    {"_sample_": {"distribution": "log_uniform", "from": 0.001, "to": 1, "num": 10}}
    {"_sample_": {"distribution": "normal", "mean": 0, "std": 1, "num": 10}}
    {"_sample_": {"distribution": "choice", "values": [...], "num": 5}}

Examples:
    {"_sample_": {"distribution": "uniform", "from": 0.1, "to": 1.0, "num": 5}}
    -> [0.234, 0.567, 0.891, 0.123, 0.456]  (5 random uniform values)
"""

import math
import random
from typing import Any, Dict, FrozenSet, List, Optional

from .base import ExpansionStrategy, GeneratorNode, ExpandedResult
from .registry import register_strategy
from ..keywords import SAMPLE_KEYWORD, COUNT_KEYWORD, SEED_KEYWORD, PURE_SAMPLE_KEYS


@register_strategy
class SampleStrategy(ExpansionStrategy):
    """Strategy for handling _sample_ nodes.

    Generates values using statistical sampling from various distributions.
    Supports uniform, log-uniform, normal, and choice distributions.

    Supported distributions:
        - uniform: Uniform distribution between from and to
        - log_uniform: Log-uniform distribution (common for learning rates)
        - normal/gaussian: Normal distribution with mean and std
        - choice: Random selection from a list of values

    Attributes:
        keywords: {_sample_, count, seed}
        priority: 24 (between log_range and range)
    """

    keywords: FrozenSet[str] = PURE_SAMPLE_KEYS
    priority: int = 24

    SUPPORTED_DISTRIBUTIONS = {"uniform", "log_uniform", "normal", "gaussian", "choice"}

    @classmethod
    def handles(cls, node: GeneratorNode) -> bool:
        """Check if node is a pure sample node.

        Args:
            node: Dictionary node to check.

        Returns:
            True if node contains _sample_ and only sample-related keys.
        """
        if not isinstance(node, dict):
            return False
        return SAMPLE_KEYWORD in node and set(node.keys()).issubset(PURE_SAMPLE_KEYS)

    def expand(
        self,
        node: GeneratorNode,
        seed: Optional[int] = None,
        expand_nested: Optional[callable] = None
    ) -> ExpandedResult:
        """Expand a sample node to list of sampled values.

        Args:
            node: Sample specification node.
            seed: Optional seed for reproducible sampling.
            expand_nested: Not typically used for sample nodes.

        Returns:
            List of sampled values.

        Examples:
            >>> strategy.expand({"_sample_": {"distribution": "uniform", "from": 0, "to": 1, "num": 3}}, seed=42)
            [0.6394267984578837, 0.025010755222666936, 0.27502931836911926]
        """
        sample_spec = node[SAMPLE_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)
        node_seed = node.get(SEED_KEYWORD, seed)

        if not isinstance(sample_spec, dict):
            raise ValueError(
                f"_sample_ must be a dict, got {type(sample_spec).__name__}"
            )

        # Initialize RNG with seed
        if node_seed is not None:
            rng = random.Random(node_seed)
        else:
            rng = random

        distribution = sample_spec.get("distribution", "uniform")
        num = sample_spec.get("num", 10)

        # Apply count limit to num if specified
        if count_limit is not None:
            num = min(num, count_limit)

        # Generate samples based on distribution
        if distribution == "uniform":
            results = self._sample_uniform(sample_spec, num, rng)
        elif distribution == "log_uniform":
            results = self._sample_log_uniform(sample_spec, num, rng)
        elif distribution in ("normal", "gaussian"):
            results = self._sample_normal(sample_spec, num, rng)
        elif distribution == "choice":
            results = self._sample_choice(sample_spec, num, rng)
        else:
            raise ValueError(
                f"Unknown distribution: {distribution}. "
                f"Supported: {self.SUPPORTED_DISTRIBUTIONS}"
            )

        return results

    def _sample_uniform(
        self, spec: Dict[str, Any], num: int, rng: random.Random
    ) -> List[float]:
        """Sample from uniform distribution."""
        low = spec.get("from", 0)
        high = spec.get("to", 1)

        return [round(rng.uniform(low, high), 10) for _ in range(num)]

    def _sample_log_uniform(
        self, spec: Dict[str, Any], num: int, rng: random.Random
    ) -> List[float]:
        """Sample from log-uniform distribution."""
        low = spec.get("from", 0.001)
        high = spec.get("to", 1)

        if low <= 0 or high <= 0:
            raise ValueError("Log-uniform requires positive from and to values")

        log_low = math.log(low)
        log_high = math.log(high)

        results = []
        for _ in range(num):
            log_val = rng.uniform(log_low, log_high)
            results.append(round(math.exp(log_val), 10))
        return results

    def _sample_normal(
        self, spec: Dict[str, Any], num: int, rng: random.Random
    ) -> List[float]:
        """Sample from normal distribution."""
        mean = spec.get("mean", 0)
        std = spec.get("std", 1)

        return [round(rng.gauss(mean, std), 10) for _ in range(num)]

    def _sample_choice(
        self, spec: Dict[str, Any], num: int, rng: random.Random
    ) -> List[Any]:
        """Sample from a list of choices."""
        values = spec.get("values", [])

        if not values:
            return []

        return [rng.choice(values) for _ in range(num)]

    def count(self, node: GeneratorNode, count_nested: Optional[callable] = None) -> int:
        """Count sample results (simply returns num).

        Args:
            node: Sample specification node.
            count_nested: Not used.

        Returns:
            Number of samples to generate.
        """
        sample_spec = node[SAMPLE_KEYWORD]
        count_limit = node.get(COUNT_KEYWORD)

        if not isinstance(sample_spec, dict):
            return 0

        num = sample_spec.get("num", 10)

        # Apply count limit
        if count_limit is not None:
            return min(count_limit, num)
        return num

    def validate(self, node: GeneratorNode) -> List[str]:
        """Validate sample node specification.

        Args:
            node: Sample node to validate.

        Returns:
            List of error messages. Empty if valid.
        """
        errors = []
        sample_spec = node.get(SAMPLE_KEYWORD)

        if sample_spec is None:
            errors.append("Missing _sample_ key")
            return errors

        if not isinstance(sample_spec, dict):
            errors.append(f"_sample_ must be a dict, got {type(sample_spec).__name__}")
            return errors

        distribution = sample_spec.get("distribution", "uniform")
        if distribution not in self.SUPPORTED_DISTRIBUTIONS:
            errors.append(
                f"Unknown distribution: {distribution}. "
                f"Supported: {self.SUPPORTED_DISTRIBUTIONS}"
            )

        # Validate distribution-specific parameters
        if distribution == "uniform":
            for key in ("from", "to"):
                if key in sample_spec and not isinstance(sample_spec[key], (int, float)):
                    errors.append(f"uniform '{key}' must be numeric")

        elif distribution == "log_uniform":
            for key in ("from", "to"):
                if key in sample_spec:
                    val = sample_spec[key]
                    if not isinstance(val, (int, float)):
                        errors.append(f"log_uniform '{key}' must be numeric")
                    elif val <= 0:
                        errors.append(f"log_uniform '{key}' must be positive")

        elif distribution in ("normal", "gaussian"):
            for key in ("mean", "std"):
                if key in sample_spec and not isinstance(sample_spec[key], (int, float)):
                    errors.append(f"normal '{key}' must be numeric")
            if "std" in sample_spec and sample_spec["std"] < 0:
                errors.append("normal 'std' must be non-negative")

        elif distribution == "choice":
            values = sample_spec.get("values")
            if values is not None and not isinstance(values, list):
                errors.append("choice 'values' must be a list")

        # Validate num
        num = sample_spec.get("num")
        if num is not None and (not isinstance(num, int) or num < 0):
            errors.append("num must be a non-negative integer")

        # Validate count
        count = node.get(COUNT_KEYWORD)
        if count is not None and not isinstance(count, int):
            errors.append(f"count must be an integer, got {type(count).__name__}")

        return errors
