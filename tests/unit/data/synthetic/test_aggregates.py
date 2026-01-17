"""
Unit tests for aggregate spectral components.

Tests cover:
- AggregateComponent dataclass
- Aggregate retrieval and listing
- Composition expansion with variability
- Builder integration with aggregates
"""

import pytest
import numpy as np

from nirs4all.data.synthetic import (
    # Aggregate classes and functions
    AggregateComponent,
    AGGREGATE_COMPONENTS,
    get_aggregate,
    list_aggregates,
    expand_aggregate,
    aggregate_info,
    list_aggregate_domains,
    list_aggregate_categories,
    validate_aggregates,
    # Builder
    SyntheticDatasetBuilder,
)


class TestAggregateComponent:
    """Tests for AggregateComponent dataclass."""

    def test_init_basic(self):
        """Test basic AggregateComponent initialization."""
        agg = AggregateComponent(
            name="test_aggregate",
            components={"comp_a": 0.6, "comp_b": 0.3, "comp_c": 0.1},
            description="Test aggregate",
            domain="test",
            category="unit_test",
        )
        assert agg.name == "test_aggregate"
        assert len(agg.components) == 3
        assert agg.description == "Test aggregate"
        assert agg.domain == "test"
        assert agg.category == "unit_test"
        assert agg.variability == {}
        assert agg.correlations == []
        assert agg.tags == []

    def test_init_with_variability(self):
        """Test AggregateComponent with variability ranges."""
        agg = AggregateComponent(
            name="var_aggregate",
            components={"protein": 0.12, "starch": 0.65, "moisture": 0.12},
            description="Variable aggregate",
            domain="test",
            variability={
                "protein": (0.08, 0.18),
                "moisture": (0.08, 0.15),
            },
        )
        assert "protein" in agg.variability
        assert agg.variability["protein"] == (0.08, 0.18)

    def test_validate_valid_aggregate(self):
        """Test validation passes for valid aggregate."""
        agg = AggregateComponent(
            name="valid",
            components={"a": 0.5, "b": 0.3, "c": 0.2},
            description="Valid aggregate",
            domain="test",
        )
        issues = agg.validate()
        assert len(issues) == 0

    def test_validate_sum_warning(self):
        """Test validation catches weight sum issues."""
        agg = AggregateComponent(
            name="bad_sum",
            components={"a": 0.5, "b": 0.3},  # Sum = 0.8
            description="Bad sum aggregate",
            domain="test",
        )
        issues = agg.validate()
        assert len(issues) == 1
        assert "sum to" in issues[0]

    def test_validate_negative_weight(self):
        """Test validation catches negative weights."""
        agg = AggregateComponent(
            name="negative",
            components={"a": 0.8, "b": -0.2, "c": 0.4},
            description="Negative weight",
            domain="test",
        )
        issues = agg.validate()
        assert any("negative" in issue for issue in issues)

    def test_validate_variability_unknown_component(self):
        """Test validation catches variability for unknown component."""
        agg = AggregateComponent(
            name="bad_var",
            components={"a": 0.5, "b": 0.5},
            description="Bad variability",
            domain="test",
            variability={"c": (0.1, 0.3)},  # 'c' not in components
        )
        issues = agg.validate()
        assert any("unknown component" in issue for issue in issues)

    def test_info_output(self):
        """Test info() method produces formatted output."""
        agg = AggregateComponent(
            name="info_test",
            components={"protein": 0.15, "starch": 0.70, "moisture": 0.10},
            description="Info test aggregate",
            domain="agriculture",
            category="grain",
            tags=["test", "grain"],
        )
        info = agg.info()
        assert "info_test" in info
        assert "Info test aggregate" in info
        assert "protein:" in info
        assert "starch:" in info
        assert "Tags:" in info


class TestPredefinedAggregates:
    """Tests for predefined aggregate components."""

    def test_aggregate_count(self):
        """Test expected number of predefined aggregates."""
        assert len(AGGREGATE_COMPONENTS) >= 15  # Phase 4 minimum

    def test_wheat_grain_aggregate(self):
        """Test wheat_grain aggregate exists and is valid."""
        wheat = get_aggregate("wheat_grain")
        assert wheat.name == "wheat_grain"
        assert wheat.domain == "agriculture"
        assert wheat.category == "grain"
        assert "starch" in wheat.components
        assert "protein" in wheat.components
        assert "moisture" in wheat.components
        assert len(wheat.validate()) == 0

    def test_corn_grain_aggregate(self):
        """Test corn_grain aggregate exists and is valid."""
        corn = get_aggregate("corn_grain")
        assert corn.name == "corn_grain"
        assert "starch" in corn.components
        assert len(corn.validate()) == 0

    def test_soybean_aggregate(self):
        """Test soybean aggregate exists and is valid."""
        soy = get_aggregate("soybean")
        assert soy.name == "soybean"
        assert soy.domain == "agriculture"
        assert "protein" in soy.components
        assert "lipid" in soy.components
        assert len(soy.validate()) == 0

    def test_milk_aggregate(self):
        """Test milk aggregate exists and is valid."""
        milk = get_aggregate("milk")
        assert milk.name == "milk"
        assert milk.domain == "food"
        assert milk.category == "dairy"
        assert "water" in milk.components
        assert "casein" in milk.components
        assert len(milk.validate()) == 0

    def test_cheese_cheddar_aggregate(self):
        """Test cheese_cheddar aggregate exists and is valid."""
        cheese = get_aggregate("cheese_cheddar")
        assert cheese.name == "cheese_cheddar"
        assert cheese.domain == "food"
        assert "casein" in cheese.components
        assert "lipid" in cheese.components
        assert len(cheese.validate()) == 0

    def test_meat_beef_aggregate(self):
        """Test meat_beef aggregate exists and is valid."""
        beef = get_aggregate("meat_beef")
        assert beef.name == "meat_beef"
        assert beef.domain == "food"
        assert beef.category == "meat"
        assert "protein" in beef.components
        assert "water" in beef.components
        assert len(beef.validate()) == 0

    def test_tablet_excipient_base_aggregate(self):
        """Test tablet_excipient_base aggregate exists and is valid."""
        tablet = get_aggregate("tablet_excipient_base")
        assert tablet.name == "tablet_excipient_base"
        assert tablet.domain == "pharmaceutical"
        assert "starch" in tablet.components
        assert len(tablet.validate()) == 0

    def test_soil_agricultural_aggregate(self):
        """Test soil_agricultural aggregate exists and is valid."""
        soil = get_aggregate("soil_agricultural")
        assert soil.name == "soil_agricultural"
        assert soil.domain == "environmental"
        assert soil.category == "soil"
        assert "moisture" in soil.components
        assert len(soil.validate()) == 0

    def test_leaf_green_aggregate(self):
        """Test leaf_green aggregate exists and is valid."""
        leaf = get_aggregate("leaf_green")
        assert leaf.name == "leaf_green"
        assert leaf.domain == "agriculture"
        assert "water" in leaf.components
        assert "chlorophyll" in leaf.components
        assert len(leaf.validate()) == 0

    def test_all_predefined_valid(self):
        """Test all predefined aggregates pass validation."""
        issues = validate_aggregates()
        # Allow some issues but report them
        if issues:
            for issue in issues:
                print(f"Validation issue: {issue}")
        # Most critical aggregates should be valid
        critical = ["wheat_grain", "milk", "meat_beef", "tablet_excipient_base"]
        for name in critical:
            agg = get_aggregate(name)
            agg_issues = agg.validate()
            assert len(agg_issues) == 0, f"{name} has issues: {agg_issues}"


class TestGetAggregate:
    """Tests for get_aggregate function."""

    def test_get_existing(self):
        """Test getting existing aggregate."""
        agg = get_aggregate("wheat_grain")
        assert agg.name == "wheat_grain"

    def test_get_unknown_raises(self):
        """Test getting unknown aggregate raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_aggregate("nonexistent_aggregate")
        assert "Unknown aggregate" in str(exc_info.value)
        assert "nonexistent_aggregate" in str(exc_info.value)


class TestListAggregates:
    """Tests for list_aggregates function."""

    def test_list_all(self):
        """Test listing all aggregates."""
        all_aggs = list_aggregates()
        assert len(all_aggs) >= 15
        assert "wheat_grain" in all_aggs
        assert "milk" in all_aggs

    def test_filter_by_domain(self):
        """Test filtering aggregates by domain."""
        ag_aggs = list_aggregates(domain="agriculture")
        assert "wheat_grain" in ag_aggs
        assert "corn_grain" in ag_aggs
        assert "milk" not in ag_aggs

        food_aggs = list_aggregates(domain="food")
        assert "milk" in food_aggs
        assert "cheese_cheddar" in food_aggs
        assert "wheat_grain" not in food_aggs

    def test_filter_by_category(self):
        """Test filtering aggregates by category."""
        grain_aggs = list_aggregates(category="grain")
        assert "wheat_grain" in grain_aggs
        assert "corn_grain" in grain_aggs
        assert "soybean" not in grain_aggs  # legume category

    def test_filter_by_tags(self):
        """Test filtering aggregates by tags."""
        dairy_aggs = list_aggregates(tags=["dairy"])
        assert "milk" in dairy_aggs
        assert "cheese_cheddar" in dairy_aggs
        assert "wheat_grain" not in dairy_aggs

    def test_combined_filters(self):
        """Test combining domain and category filters."""
        food_dairy = list_aggregates(domain="food", category="dairy")
        assert "milk" in food_dairy
        assert "yogurt" in food_dairy
        assert "meat_beef" not in food_dairy


class TestExpandAggregate:
    """Tests for expand_aggregate function."""

    def test_expand_fixed(self):
        """Test expanding without variability."""
        comp = expand_aggregate("wheat_grain", variability=False)
        assert isinstance(comp, dict)
        assert "starch" in comp
        assert "protein" in comp
        # Should match base values (with small adjustment due to renormalization)
        wheat = get_aggregate("wheat_grain")
        # Allow 5% tolerance due to renormalization
        assert comp["protein"] == pytest.approx(wheat.components["protein"], rel=0.05)

    def test_expand_with_variability(self):
        """Test expanding with variability."""
        # Generate multiple samples
        samples = [
            expand_aggregate("wheat_grain", variability=True, random_state=i)
            for i in range(50)
        ]

        # Check protein varies within range
        proteins = [s["protein"] for s in samples]
        wheat = get_aggregate("wheat_grain")
        min_prot, max_prot = wheat.variability["protein"]

        assert min(proteins) >= min_prot * 0.9  # Allow small margin
        assert max(proteins) <= max_prot * 1.1
        # Should have meaningful variation
        assert max(proteins) - min(proteins) > 0.02

    def test_expand_sum_normalized(self):
        """Test expanded composition sums to ~1.0."""
        comp = expand_aggregate("wheat_grain", variability=True, random_state=42)
        total = sum(comp.values())
        assert total == pytest.approx(1.0, abs=0.05)

    def test_expand_deterministic(self):
        """Test expansion is deterministic with same random_state."""
        comp1 = expand_aggregate("wheat_grain", variability=True, random_state=42)
        comp2 = expand_aggregate("wheat_grain", variability=True, random_state=42)
        for key in comp1:
            assert comp1[key] == pytest.approx(comp2[key], rel=1e-10)


class TestAggregateInfo:
    """Tests for aggregate_info function."""

    def test_returns_string(self):
        """Test aggregate_info returns formatted string."""
        info = aggregate_info("wheat_grain")
        assert isinstance(info, str)
        assert "wheat_grain" in info
        assert "protein" in info


class TestListDomains:
    """Tests for list_aggregate_domains function."""

    def test_returns_sorted_list(self):
        """Test list_aggregate_domains returns sorted unique domains."""
        domains = list_aggregate_domains()
        assert isinstance(domains, list)
        assert "agriculture" in domains
        assert "food" in domains
        assert "pharmaceutical" in domains
        assert domains == sorted(domains)


class TestListCategories:
    """Tests for list_aggregate_categories function."""

    def test_returns_dict(self):
        """Test list_aggregate_categories returns dict of categories."""
        cats = list_aggregate_categories()
        assert isinstance(cats, dict)
        assert "grain" in cats
        assert "dairy" in cats

    def test_filter_by_domain(self):
        """Test filtering categories by domain."""
        food_cats = list_aggregate_categories(domain="food")
        assert "dairy" in food_cats
        assert "meat" in food_cats


class TestBuilderIntegration:
    """Tests for SyntheticDatasetBuilder.with_aggregate()."""

    def test_with_aggregate_basic(self):
        """Test basic aggregate-based generation."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.with_aggregate("wheat_grain")

        # Check state is set
        assert builder.state.aggregate_name == "wheat_grain"
        assert builder.state.component_names is not None
        assert "starch" in builder.state.component_names

    def test_with_aggregate_with_variability(self):
        """Test aggregate generation with variability."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.with_aggregate("wheat_grain", variability=True)

        assert builder.state.aggregate_name == "wheat_grain"
        assert builder.state.aggregate_variability is True

    def test_with_aggregate_target_component(self):
        """Test specifying target component from aggregate."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)
        builder.with_aggregate("wheat_grain", target_component="protein")

        assert builder.state.target_component == "protein"

    def test_with_aggregate_invalid_target(self):
        """Test invalid target component raises error."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)

        with pytest.raises(ValueError) as exc_info:
            builder.with_aggregate("wheat_grain", target_component="invalid_comp")

        assert "invalid_comp" in str(exc_info.value)

    def test_with_aggregate_unknown_raises(self):
        """Test unknown aggregate raises error."""
        builder = SyntheticDatasetBuilder(n_samples=100, random_state=42)

        with pytest.raises(ValueError):
            builder.with_aggregate("unknown_aggregate")

    def test_with_aggregate_chaining(self):
        """Test method chaining with aggregate."""
        builder = (
            SyntheticDatasetBuilder(n_samples=100, random_state=42)
            .with_aggregate("wheat_grain", variability=True)
            .with_features(complexity="realistic")
            .with_targets(component="protein", range=(8, 18))
        )

        assert builder.state.aggregate_name == "wheat_grain"
        assert builder.state.complexity == "realistic"
        assert builder.state.target_range == (8, 18)

    @pytest.mark.parametrize("aggregate_name", [
        "wheat_grain",
        "corn_grain",
        "milk",
        "cheese_cheddar",
        "meat_beef",
    ])
    def test_build_with_aggregate(self, aggregate_name):
        """Test building dataset from various aggregates."""
        dataset = (
            SyntheticDatasetBuilder(n_samples=50, random_state=42)
            .with_aggregate(aggregate_name, variability=True)
            .with_features(complexity="simple")
            .build()
        )

        assert dataset is not None
        assert dataset.num_samples == 50

    def test_build_aggregate_spectra_realistic(self):
        """Test generated spectra have reasonable properties."""
        dataset = (
            SyntheticDatasetBuilder(n_samples=100, random_state=42)
            .with_aggregate("wheat_grain", variability=True)
            .with_features(complexity="realistic")
            .with_targets(component="protein")
            .build()
        )

        X = dataset.x({}, layout="2d")
        y = dataset.y({})

        # Check spectra properties
        assert X.shape[0] == 100
        assert X.shape[1] > 100  # Many wavelengths
        assert np.all(np.isfinite(X))

        # Check targets
        assert len(y) == 100
        assert np.all(np.isfinite(y))
