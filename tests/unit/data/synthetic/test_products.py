"""
Unit tests for synthetic product-level generation module (Phase 7).

Tests for:
    - VariationType enum
    - ComponentVariation dataclass
    - ProductTemplate dataclass
    - ProductGenerator class
    - CategoryGenerator class
    - Convenience functions
"""

import numpy as np
import pytest

from nirs4all.synthesis import (
    VariationType,
    ComponentVariation,
    ProductTemplate,
    PRODUCT_TEMPLATES,
    ProductGenerator,
    CategoryGenerator,
    list_product_templates,
    get_product_template,
    generate_product_samples,
    product_template_info,
    list_product_categories,
    list_product_domains,
)


# =============================================================================
# VariationType Enum Tests
# =============================================================================


class TestVariationType:
    """Tests for VariationType enum."""

    def test_all_types_exist(self):
        """Test that all required variation types exist."""
        assert hasattr(VariationType, "FIXED")
        assert hasattr(VariationType, "UNIFORM")
        assert hasattr(VariationType, "NORMAL")
        assert hasattr(VariationType, "LOGNORMAL")
        assert hasattr(VariationType, "CORRELATED")
        assert hasattr(VariationType, "COMPUTED")

    def test_types_are_distinct(self):
        """Test that each variation type has a unique value."""
        types = [
            VariationType.FIXED,
            VariationType.UNIFORM,
            VariationType.NORMAL,
            VariationType.LOGNORMAL,
            VariationType.CORRELATED,
            VariationType.COMPUTED,
        ]
        values = [t.value for t in types]
        assert len(values) == len(set(values))


# =============================================================================
# ComponentVariation Tests
# =============================================================================


class TestComponentVariation:
    """Tests for ComponentVariation dataclass."""

    def test_fixed_variation(self):
        """Test FIXED variation type."""
        comp = ComponentVariation("moisture", VariationType.FIXED, value=0.12)
        assert comp.component == "moisture"
        assert comp.variation_type == VariationType.FIXED
        assert comp.value == 0.12

    def test_fixed_requires_value(self):
        """Test that FIXED requires value parameter."""
        with pytest.raises(ValueError, match="FIXED variation requires 'value'"):
            ComponentVariation("moisture", VariationType.FIXED)

    def test_uniform_variation(self):
        """Test UNIFORM variation type."""
        comp = ComponentVariation(
            "protein", VariationType.UNIFORM,
            min_value=0.08, max_value=0.18
        )
        assert comp.min_value == 0.08
        assert comp.max_value == 0.18

    def test_uniform_requires_min_max(self):
        """Test that UNIFORM requires min/max values."""
        with pytest.raises(ValueError, match="UNIFORM variation requires"):
            ComponentVariation("protein", VariationType.UNIFORM, min_value=0.08)

    def test_uniform_min_less_than_max(self):
        """Test that min_value must be <= max_value."""
        with pytest.raises(ValueError, match="min_value must be <= max_value"):
            ComponentVariation(
                "protein", VariationType.UNIFORM,
                min_value=0.20, max_value=0.10
            )

    def test_normal_variation(self):
        """Test NORMAL variation type."""
        comp = ComponentVariation(
            "casein", VariationType.NORMAL,
            mean=0.028, std=0.003
        )
        assert comp.mean == 0.028
        assert comp.std == 0.003

    def test_normal_requires_mean_std(self):
        """Test that NORMAL requires mean and std."""
        with pytest.raises(ValueError, match="NORMAL variation requires"):
            ComponentVariation("casein", VariationType.NORMAL, mean=0.028)

    def test_normal_std_non_negative(self):
        """Test that std must be non-negative."""
        with pytest.raises(ValueError, match="std must be non-negative"):
            ComponentVariation(
                "casein", VariationType.NORMAL,
                mean=0.028, std=-0.003
            )

    def test_lognormal_variation(self):
        """Test LOGNORMAL variation type."""
        comp = ComponentVariation(
            "cholesterol", VariationType.LOGNORMAL,
            mean=0.01, std=0.015
        )
        assert comp.mean == 0.01
        assert comp.std == 0.015

    def test_correlated_variation(self):
        """Test CORRELATED variation type."""
        comp = ComponentVariation(
            "starch", VariationType.CORRELATED,
            correlated_with="protein", correlation=-0.85,
            min_value=0.55, max_value=0.72
        )
        assert comp.correlated_with == "protein"
        assert comp.correlation == -0.85

    def test_correlated_requires_source_and_coefficient(self):
        """Test that CORRELATED requires source component and correlation."""
        with pytest.raises(ValueError, match="CORRELATED variation requires"):
            ComponentVariation(
                "starch", VariationType.CORRELATED,
                correlated_with="protein"
            )

    def test_correlated_correlation_bounds(self):
        """Test that correlation must be between -1 and 1."""
        with pytest.raises(ValueError, match="correlation must be between"):
            ComponentVariation(
                "starch", VariationType.CORRELATED,
                correlated_with="protein", correlation=1.5
            )

    def test_computed_variation(self):
        """Test COMPUTED variation type."""
        comp = ComponentVariation(
            "water", VariationType.COMPUTED,
            compute_as="remainder"
        )
        assert comp.compute_as == "remainder"

    def test_computed_requires_compute_as(self):
        """Test that COMPUTED requires compute_as parameter."""
        with pytest.raises(ValueError, match="COMPUTED variation requires"):
            ComponentVariation("water", VariationType.COMPUTED)


# =============================================================================
# ProductTemplate Tests
# =============================================================================


class TestProductTemplate:
    """Tests for ProductTemplate dataclass."""

    def test_basic_template(self):
        """Test creating a basic product template."""
        template = ProductTemplate(
            name="test_product",
            description="Test product for unit tests",
            category="test",
            domain="testing",
            components=[
                ComponentVariation("protein", VariationType.UNIFORM,
                                   min_value=0.10, max_value=0.20),
                ComponentVariation("water", VariationType.COMPUTED,
                                   compute_as="remainder"),
            ],
            default_target="protein",
        )
        assert template.name == "test_product"
        assert len(template.components) == 2
        assert template.component_names == ["protein", "water"]

    def test_duplicate_components_rejected(self):
        """Test that duplicate component names are rejected."""
        with pytest.raises(ValueError, match="Duplicate components"):
            ProductTemplate(
                name="duplicate_test",
                description="Test",
                category="test",
                domain="testing",
                components=[
                    ComponentVariation("protein", VariationType.FIXED, value=0.1),
                    ComponentVariation("protein", VariationType.FIXED, value=0.2),
                ],
            )

    def test_invalid_correlation_reference(self):
        """Test that invalid correlation references are rejected."""
        with pytest.raises(ValueError, match="correlates with.*which is not in template"):
            ProductTemplate(
                name="invalid_corr",
                description="Test",
                category="test",
                domain="testing",
                components=[
                    ComponentVariation(
                        "starch", VariationType.CORRELATED,
                        correlated_with="protein", correlation=-0.5,
                        min_value=0.5, max_value=0.7
                    ),
                ],
            )

    def test_template_info(self):
        """Test template info() method."""
        template = get_product_template("milk_variable_fat")
        info = template.info()
        assert "ProductTemplate: milk_variable_fat" in info
        assert "lipid" in info
        assert "dairy" in info


# =============================================================================
# Predefined Templates Tests
# =============================================================================


class TestPredefinedTemplates:
    """Tests for predefined product templates."""

    def test_templates_exist(self):
        """Test that predefined templates exist."""
        assert len(PRODUCT_TEMPLATES) >= 15  # Phase 7 requires 15+ templates

    def test_required_templates_present(self):
        """Test that required templates from Phase 7 spec are present."""
        required = [
            "milk_variable_fat",
            "cheese_variable_moisture",
            "meat_variable_fat",
            "wheat_variable_protein",
            "tablet_variable_api",
            "food_cholesterol_variable",
        ]
        for name in required:
            assert name in PRODUCT_TEMPLATES, f"Required template '{name}' not found"

    def test_all_templates_valid(self):
        """Test that all templates have valid component names."""
        from nirs4all.synthesis import available_components

        available = set(available_components())

        for name, template in PRODUCT_TEMPLATES.items():
            for comp_name in template.component_names:
                assert comp_name in available, (
                    f"Template '{name}' uses unknown component '{comp_name}'"
                )

    def test_dairy_templates(self):
        """Test dairy product templates."""
        dairy = list_product_templates(category="dairy")
        assert len(dairy) >= 2
        assert "milk_variable_fat" in dairy
        assert "cheese_variable_moisture" in dairy

    def test_grain_templates(self):
        """Test grain product templates."""
        grain = list_product_templates(category="grain")
        assert len(grain) >= 3
        assert "wheat_variable_protein" in grain

    def test_nn_training_templates(self):
        """Test high-variability NN training templates."""
        nn_templates = list_product_templates(tags=["nn_training"])
        assert len(nn_templates) >= 5
        assert "food_cholesterol_variable" in nn_templates


# =============================================================================
# ProductGenerator Tests
# =============================================================================


class TestProductGenerator:
    """Tests for ProductGenerator class."""

    def test_create_generator(self):
        """Test creating a ProductGenerator."""
        gen = ProductGenerator("milk_variable_fat", random_state=42)
        assert gen.template.name == "milk_variable_fat"
        assert len(gen.library.components) == len(gen.template.components)

    def test_create_generator_with_template_object(self):
        """Test creating generator with ProductTemplate object."""
        template = get_product_template("wheat_variable_protein")
        gen = ProductGenerator(template, random_state=42)
        assert gen.template.name == "wheat_variable_protein"

    def test_generate_samples(self):
        """Test generating samples."""
        gen = ProductGenerator("milk_variable_fat", random_state=42)
        dataset = gen.generate(n_samples=100, target="lipid")

        assert dataset is not None
        # Check total samples using num_samples
        assert dataset.num_samples == 100
        # Check train/test split using selector
        X_train = dataset.x({"partition": "train"}, layout="2d")
        X_test = dataset.x({"partition": "test"}, layout="2d")
        assert len(X_train) == 80  # Default 80/20 split
        assert len(X_test) == 20

    def test_generate_with_concentrations(self):
        """Test generating samples with concentration return."""
        gen = ProductGenerator("wheat_variable_protein", random_state=42)
        dataset, concentrations = gen.generate(
            n_samples=50,
            target="protein",
            return_concentrations=True
        )

        assert concentrations.shape == (50, len(gen.template.components))
        assert np.all(concentrations >= 0)
        # Check row sums are approximately 1
        row_sums = concentrations.sum(axis=1)
        assert np.all(row_sums > 0.8)
        assert np.all(row_sums < 1.2)

    def test_generate_respects_variability(self):
        """Test that generated concentrations respect variability ranges."""
        gen = ProductGenerator("wheat_variable_protein", random_state=42)
        _, concentrations = gen.generate(
            n_samples=1000,
            target="protein",
            return_concentrations=True
        )

        # Protein should vary between 0.08 and 0.18
        protein_idx = gen.template.component_names.index("protein")
        protein_values = concentrations[:, protein_idx]

        assert protein_values.min() >= 0.07  # Allow small tolerance
        assert protein_values.max() <= 0.19

    def test_generate_with_custom_wavelengths(self):
        """Test generating with custom wavelength grid."""
        custom_wl = np.linspace(1000, 2000, 100)
        gen = ProductGenerator(
            "milk_variable_fat",
            wavelengths=custom_wl,
            random_state=42
        )
        dataset = gen.generate(n_samples=50)

        # Get X shape
        X = dataset.x({}, layout="2d")
        assert X.shape[1] == 100

    def test_generate_dataset_for_target_with_scaling(self):
        """Test generating dataset with target scaling."""
        gen = ProductGenerator("wheat_variable_protein", random_state=42)
        dataset = gen.generate_dataset_for_target(
            target="protein",
            n_samples=100,
            target_range=(0, 100)
        )

        # Get target values using the selector API
        y = dataset.y({})
        assert y.min() >= 0
        assert y.max() <= 100

    def test_generator_repr(self):
        """Test generator string representation."""
        gen = ProductGenerator("milk_variable_fat")
        repr_str = repr(gen)
        assert "ProductGenerator" in repr_str
        assert "milk_variable_fat" in repr_str


# =============================================================================
# CategoryGenerator Tests
# =============================================================================


class TestCategoryGenerator:
    """Tests for CategoryGenerator class."""

    def test_create_category_generator(self):
        """Test creating a CategoryGenerator."""
        gen = CategoryGenerator(
            ["milk_variable_fat", "meat_variable_fat"],
            random_state=42
        )
        assert len(gen.templates) == 2
        assert len(gen.generators) == 2

    def test_generate_combined_dataset(self):
        """Test generating combined dataset."""
        gen = CategoryGenerator(
            ["milk_variable_fat", "meat_variable_fat"],
            random_state=42
        )
        dataset = gen.generate(n_samples=200, target="lipid")

        assert dataset.num_samples == 200

    def test_generate_with_samples_per_template(self):
        """Test generating with custom samples per template."""
        # Note: Both templates must have the target component
        # milk_variable_fat has: lipid, casein, whey, lactose, water
        # cheese_variable_moisture has: moisture, lipid, casein, lactose
        gen = CategoryGenerator(
            ["milk_variable_fat", "cheese_variable_moisture"],
            random_state=42
        )
        dataset = gen.generate(
            n_samples=300,
            target="lipid",  # Both have lipid
            samples_per_template=[200, 100]
        )

        assert dataset.num_samples == 300

    def test_generate_without_shuffle(self):
        """Test generating without shuffling."""
        gen = CategoryGenerator(
            ["milk_variable_fat", "cheese_variable_moisture"],
            random_state=42
        )
        dataset = gen.generate(
            n_samples=100,
            target="lipid",
            shuffle=False
        )

        assert dataset is not None
        assert dataset.num_samples == 100

    def test_category_generator_repr(self):
        """Test CategoryGenerator string representation."""
        gen = CategoryGenerator(["milk_variable_fat", "meat_variable_fat"])
        repr_str = repr(gen)
        assert "CategoryGenerator" in repr_str
        assert "milk_variable_fat" in repr_str


# =============================================================================
# Convenience Functions Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_list_product_templates(self):
        """Test list_product_templates function."""
        all_templates = list_product_templates()
        assert len(all_templates) >= 15

    def test_list_product_templates_by_category(self):
        """Test filtering by category."""
        dairy = list_product_templates(category="dairy")
        assert all("dairy" == get_product_template(t).category for t in dairy)

    def test_list_product_templates_by_domain(self):
        """Test filtering by domain."""
        food = list_product_templates(domain="food")
        assert all("food" == get_product_template(t).domain for t in food)

    def test_list_product_templates_by_tags(self):
        """Test filtering by tags."""
        nn = list_product_templates(tags=["nn_training"])
        for t in nn:
            template = get_product_template(t)
            assert "nn_training" in template.tags

    def test_get_product_template(self):
        """Test get_product_template function."""
        template = get_product_template("milk_variable_fat")
        assert isinstance(template, ProductTemplate)
        assert template.name == "milk_variable_fat"

    def test_get_product_template_invalid(self):
        """Test get_product_template with invalid name."""
        with pytest.raises(ValueError, match="Unknown product template"):
            get_product_template("nonexistent_template")

    def test_generate_product_samples(self):
        """Test generate_product_samples convenience function."""
        dataset = generate_product_samples(
            "milk_variable_fat",
            n_samples=50,
            target="lipid",
            random_state=42
        )

        assert dataset is not None
        assert dataset.num_samples == 50

    def test_product_template_info(self):
        """Test product_template_info function."""
        info = product_template_info("wheat_variable_protein")
        assert "ProductTemplate: wheat_variable_protein" in info
        assert "protein" in info

    def test_list_product_categories(self):
        """Test list_product_categories function."""
        categories = list_product_categories()
        assert len(categories) >= 5
        assert "dairy" in categories
        assert "grain" in categories
        assert "meat" in categories

    def test_list_product_domains(self):
        """Test list_product_domains function."""
        domains = list_product_domains()
        assert len(domains) >= 3
        assert "food" in domains
        assert "agriculture" in domains
        assert "pharmaceutical" in domains


# =============================================================================
# Integration with generate API Tests
# =============================================================================


class TestGenerateAPIIntegration:
    """Tests for integration with nirs4all.generate API."""

    def test_generate_product(self):
        """Test nirs4all.generate.product() function."""
        import nirs4all

        dataset = nirs4all.generate.product(
            "milk_variable_fat",
            n_samples=50,
            target="lipid",
            random_state=42
        )

        assert dataset is not None
        assert dataset.num_samples == 50

    def test_generate_product_with_target_range(self):
        """Test generate.product with target scaling."""
        import nirs4all

        dataset = nirs4all.generate.product(
            "wheat_variable_protein",
            n_samples=100,
            target="protein",
            target_range=(0, 100),
            random_state=42
        )

        y = dataset.y({})
        assert y.min() >= 0
        assert y.max() <= 100

    def test_generate_category(self):
        """Test nirs4all.generate.category() function."""
        import nirs4all

        dataset = nirs4all.generate.category(
            ["milk_variable_fat", "meat_variable_fat"],
            n_samples=100,
            target="lipid",
            random_state=42
        )

        assert dataset is not None
        assert dataset.num_samples == 100


# =============================================================================
# Composition Sampling Tests
# =============================================================================


class TestCompositionSampling:
    """Tests for _sample_compositions method."""

    def test_correlation_preserved(self):
        """Test that negative correlations are preserved."""
        gen = ProductGenerator("wheat_variable_protein", random_state=42)
        _, concentrations = gen.generate(
            n_samples=500,
            return_concentrations=True
        )

        # Protein and starch should be negatively correlated
        protein_idx = gen.template.component_names.index("protein")
        starch_idx = gen.template.component_names.index("starch")

        correlation = np.corrcoef(
            concentrations[:, protein_idx],
            concentrations[:, starch_idx]
        )[0, 1]

        # Should be negative (allowing for sampling variance)
        assert correlation < 0

    def test_computed_remainder(self):
        """Test that computed remainder components work correctly."""
        gen = ProductGenerator("milk_variable_fat", random_state=42)
        _, concentrations = gen.generate(
            n_samples=100,
            return_concentrations=True
        )

        # Water should be computed as remainder
        water_idx = gen.template.component_names.index("water")
        water_values = concentrations[:, water_idx]

        # Water should be positive and reasonable for milk
        assert np.all(water_values > 0.5)
        assert np.all(water_values < 0.99)

    def test_lognormal_sampling(self):
        """Test lognormal distribution sampling."""
        gen = ProductGenerator("food_cholesterol_variable", random_state=42)
        _, concentrations = gen.generate(
            n_samples=1000,
            return_concentrations=True
        )

        # Cholesterol uses lognormal
        chol_idx = gen.template.component_names.index("cholesterol")
        chol_values = concentrations[:, chol_idx]

        # Should have right-skewed distribution (lognormal characteristic)
        assert np.mean(chol_values) > np.median(chol_values)
        assert np.all(chol_values >= 0)
