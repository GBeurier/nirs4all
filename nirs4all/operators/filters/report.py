"""
Filtering report generator for sample filtering operations.

Provides utilities to generate comprehensive reports about sample filtering,
including statistics, visualizations, and export capabilities.
"""

from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import json

if TYPE_CHECKING:
    from nirs4all.data.dataset import SpectroDataset
    from nirs4all.operators.filters.base import SampleFilter
    import polars as pl


@dataclass
class FilterResult:
    """
    Result of applying a single filter.

    Attributes:
        filter_name: Name/identifier of the filter
        reason: Exclusion reason string
        n_samples: Total samples evaluated
        n_excluded: Number of samples excluded by this filter
        n_kept: Number of samples kept
        exclusion_rate: Ratio of excluded to total
        excluded_indices: Indices of excluded samples
        stats: Additional filter-specific statistics
    """

    filter_name: str
    reason: str
    n_samples: int
    n_excluded: int
    n_kept: int
    exclusion_rate: float
    excluded_indices: List[int] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "filter_name": self.filter_name,
            "reason": self.reason,
            "n_samples": self.n_samples,
            "n_excluded": self.n_excluded,
            "n_kept": self.n_kept,
            "exclusion_rate": self.exclusion_rate,
            "excluded_indices": self.excluded_indices,
            "stats": self.stats,
        }


@dataclass
class FilteringReport:
    """
    Comprehensive report of sample filtering operations.

    This class aggregates results from multiple filters and provides
    methods for analysis, visualization, and export.

    Attributes:
        dataset_name: Name of the filtered dataset
        partition: Partition that was filtered (e.g., "train")
        timestamp: When the filtering was performed
        filter_results: List of individual filter results
        combined_mode: How filters were combined ("any" or "all")
        n_total_samples: Total samples before filtering
        n_final_excluded: Final number of excluded samples
        n_final_kept: Final number of kept samples
        cascade_to_augmented: Whether augmented samples were also excluded
        n_augmented_excluded: Number of augmented samples excluded via cascade
    """

    dataset_name: str
    partition: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    filter_results: List[FilterResult] = field(default_factory=list)
    combined_mode: str = "any"
    n_total_samples: int = 0
    n_final_excluded: int = 0
    n_final_kept: int = 0
    cascade_to_augmented: bool = True
    n_augmented_excluded: int = 0

    @property
    def final_exclusion_rate(self) -> float:
        """Calculate final exclusion rate after combining filters."""
        if self.n_total_samples == 0:
            return 0.0
        return self.n_final_excluded / self.n_total_samples

    def add_filter_result(self, result: FilterResult) -> None:
        """Add a filter result to the report."""
        self.filter_results.append(result)

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary dictionary of the filtering report.

        Returns:
            Dict containing summary statistics
        """
        return {
            "dataset_name": self.dataset_name,
            "partition": self.partition,
            "timestamp": self.timestamp,
            "n_filters": len(self.filter_results),
            "combined_mode": self.combined_mode,
            "n_total_samples": self.n_total_samples,
            "n_final_excluded": self.n_final_excluded,
            "n_final_kept": self.n_final_kept,
            "final_exclusion_rate": self.final_exclusion_rate,
            "cascade_to_augmented": self.cascade_to_augmented,
            "n_augmented_excluded": self.n_augmented_excluded,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert the full report to a dictionary."""
        return {
            **self.summary(),
            "filter_results": [r.to_dict() for r in self.filter_results],
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert report to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def print_report(self, verbose: int = 1) -> None:
        """
        Print the filtering report to console.

        Args:
            verbose: Verbosity level (0=minimal, 1=normal, 2=detailed)
        """
        print("\n" + "=" * 60)
        print("SAMPLE FILTERING REPORT")
        print("=" * 60)
        print(f"Dataset: {self.dataset_name}")
        print(f"Partition: {self.partition}")
        print(f"Timestamp: {self.timestamp}")
        print("-" * 60)

        print(f"\nTotal base samples: {self.n_total_samples}")
        print(f"Samples excluded: {self.n_final_excluded} ({self.final_exclusion_rate:.1%})")
        print(f"Samples kept: {self.n_final_kept}")

        if self.cascade_to_augmented and self.n_augmented_excluded > 0:
            print(f"Augmented samples also excluded: {self.n_augmented_excluded}")

        if len(self.filter_results) > 1:
            print(f"\nFilter combination mode: {self.combined_mode}")

        if verbose >= 1 and self.filter_results:
            print("\n" + "-" * 60)
            print("PER-FILTER BREAKDOWN:")
            print("-" * 60)

            for i, result in enumerate(self.filter_results):
                print(f"\n  Filter {i + 1}: {result.filter_name}")
                print(f"    Reason: {result.reason}")
                print(f"    Excluded: {result.n_excluded} ({result.exclusion_rate:.1%})")

                if verbose >= 2 and result.stats:
                    print("    Additional stats:")
                    for key, value in result.stats.items():
                        if isinstance(value, float):
                            print(f"      - {key}: {value:.4f}")
                        else:
                            print(f"      - {key}: {value}")

        print("\n" + "=" * 60 + "\n")


class FilteringReportGenerator:
    """
    Generator for creating comprehensive filtering reports.

    This class provides utilities for collecting filter statistics,
    generating reports, and exporting results.

    Example:
        >>> generator = FilteringReportGenerator(dataset)
        >>> report = generator.create_report(
        ...     filters=[YOutlierFilter(method="iqr")],
        ...     mode="any",
        ...     partition="train"
        ... )
        >>> report.print_report()
    """

    def __init__(self, dataset: 'SpectroDataset'):
        """
        Initialize the report generator.

        Args:
            dataset: Dataset to generate reports for
        """
        self.dataset = dataset

    def create_report(
        self,
        filters: List['SampleFilter'],
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        mode: str = "any",
        partition: str = "train",
        cascade_to_augmented: bool = True,
        dry_run: bool = True,
    ) -> FilteringReport:
        """
        Create a filtering report by applying filters to data.

        Args:
            filters: List of SampleFilter instances to apply
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,) or (n_samples, n_targets)
            sample_indices: Array of sample indices corresponding to X/y
            mode: Filter combination mode ("any" or "all")
            partition: Which partition is being filtered
            cascade_to_augmented: Whether augmented samples will be cascaded
            dry_run: If True, don't actually mark samples as excluded

        Returns:
            FilteringReport with all statistics and results
        """
        report = FilteringReport(
            dataset_name=self.dataset.name,
            partition=partition,
            combined_mode=mode,
            n_total_samples=len(X),
            cascade_to_augmented=cascade_to_augmented,
        )

        # Flatten y if needed
        y_flat = y.flatten() if y.ndim > 1 else y

        # Collect individual filter masks and results
        all_masks = []

        for filter_obj in filters:
            # Fit and get mask
            filter_obj.fit(X, y_flat)
            mask = filter_obj.get_mask(X, y_flat)
            all_masks.append(mask)

            # Get excluded indices
            excluded_mask = ~mask
            excluded_idx = sample_indices[excluded_mask].tolist()

            # Get filter stats
            stats = filter_obj.get_filter_stats(X, y_flat)

            # Create filter result
            result = FilterResult(
                filter_name=filter_obj.__class__.__name__,
                reason=filter_obj.exclusion_reason,
                n_samples=len(X),
                n_excluded=int((~mask).sum()),
                n_kept=int(mask.sum()),
                exclusion_rate=stats.get("exclusion_rate", 0.0),
                excluded_indices=excluded_idx,
                stats={k: v for k, v in stats.items()
                       if k not in ["n_samples", "n_excluded", "n_kept", "exclusion_rate", "reason"]},
            )
            report.add_filter_result(result)

        # Combine masks
        if len(all_masks) == 1:
            combined_mask = all_masks[0]
        else:
            stacked = np.stack(all_masks, axis=0)
            if mode == "any":
                # Exclude if ANY filter flags -> keep only if ALL keep
                combined_mask = np.all(stacked, axis=0)
            else:  # "all"
                # Exclude only if ALL flag -> keep if ANY keeps
                combined_mask = np.any(stacked, axis=0)

        # Final statistics
        final_excluded = (~combined_mask).sum()
        report.n_final_excluded = int(final_excluded)
        report.n_final_kept = int(combined_mask.sum())

        # Estimate augmented exclusions if cascade is enabled
        if cascade_to_augmented:
            final_excluded_indices = sample_indices[~combined_mask].tolist()
            augmented = self.dataset._indexer.get_augmented_for_origins(final_excluded_indices)  # noqa: SLF001
            report.n_augmented_excluded = len(augmented)

        return report

    def generate_from_indexer(
        self,
        partition: Optional[str] = "train"
    ) -> FilteringReport:
        """
        Generate a report from current indexer exclusion state.

        This method creates a report based on samples already marked as excluded
        in the indexer, rather than applying filters.

        Args:
            partition: Partition to report on (None for all partitions)

        Returns:
            FilteringReport based on current exclusion state
        """
        # Get exclusion summary from indexer
        summary = self.dataset._indexer.get_exclusion_summary()  # noqa: SLF001

        # Build selector
        selector = {"partition": partition} if partition else {}

        # Get total samples
        total_indices = self.dataset._indexer.x_indices(  # noqa: SLF001
            selector, include_augmented=False, include_excluded=True
        )
        n_total = len(total_indices)

        # Get excluded samples
        excluded_df = self.dataset._indexer.get_excluded_samples(selector)  # noqa: SLF001

        report = FilteringReport(
            dataset_name=self.dataset.name,
            partition=partition or "all",
            n_total_samples=n_total,
            n_final_excluded=summary["total_excluded"],
            n_final_kept=n_total - summary["total_excluded"],
        )

        # Group by reason
        if len(excluded_df) > 0:
            for reason, count in summary["by_reason"].items():
                result = FilterResult(
                    filter_name=reason,
                    reason=reason,
                    n_samples=n_total,
                    n_excluded=count,
                    n_kept=n_total - count,
                    exclusion_rate=count / n_total if n_total > 0 else 0.0,
                )
                report.add_filter_result(result)

        return report

    def compare_filters(
        self,
        filters: List['SampleFilter'],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compare multiple filters on the same data without applying them.

        Useful for understanding which filter is more aggressive or to
        find the overlap between filter decisions.

        Args:
            filters: List of filters to compare
            X: Feature array
            y: Target array

        Returns:
            Dictionary with comparison statistics:
                - individual: Per-filter stats
                - overlap: Samples flagged by multiple filters
                - unique: Samples flagged by only one filter
        """
        y_flat = y.flatten() if y.ndim > 1 else y
        n_samples = len(X)

        # Collect masks
        masks = {}
        for filter_obj in filters:
            filter_obj.fit(X, y_flat)
            mask = filter_obj.get_mask(X, y_flat)
            masks[filter_obj.exclusion_reason] = mask

        # Individual stats
        individual = {}
        for name, mask in masks.items():
            individual[name] = {
                "n_excluded": int((~mask).sum()),
                "exclusion_rate": float((~mask).sum() / n_samples),
            }

        # Overlap analysis
        if len(masks) >= 2:
            # Convert to exclusion masks (True = excluded)
            exclusion_masks = [~m for m in masks.values()]
            stacked = np.stack(exclusion_masks, axis=0)

            # Samples excluded by all filters
            all_exclude = np.all(stacked, axis=0)
            # Samples excluded by at least one filter
            any_exclude = np.any(stacked, axis=0)
            # Samples excluded by exactly one filter
            exactly_one = stacked.sum(axis=0) == 1

            overlap = {
                "excluded_by_all": int(all_exclude.sum()),
                "excluded_by_any": int(any_exclude.sum()),
                "excluded_by_exactly_one": int(exactly_one.sum()),
            }

            # Per-filter unique exclusions
            unique = {}
            filter_names = list(masks.keys())
            for i, name in enumerate(filter_names):
                # Samples excluded only by this filter
                only_this = exclusion_masks[i].copy()
                for j, other_mask in enumerate(exclusion_masks):
                    if i != j:
                        only_this = only_this & ~other_mask
                unique[name] = int(only_this.sum())
        else:
            overlap = {}
            unique = {}

        return {
            "n_samples": n_samples,
            "n_filters": len(filters),
            "individual": individual,
            "overlap": overlap,
            "unique_exclusions": unique,
        }
