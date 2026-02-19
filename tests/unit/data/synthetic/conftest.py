"""
Pytest configuration for synthetic data generation tests.

Uses session-scoped fixtures where possible for faster test execution.
"""

import numpy as np
import pytest

from nirs4all.synthesis import (
    ComponentLibrary,
    NIRBand,
    SpectralComponent,
    SyntheticNIRSGenerator,
)

# =============================================================================
# Session-scoped fixtures (created once per test session)
# =============================================================================

@pytest.fixture(scope="session")
def shared_simple_generator():
    """Session-scoped simple generator (read-only use)."""
    return SyntheticNIRSGenerator(
        complexity="simple",
        random_state=42,
    )

@pytest.fixture(scope="session")
def shared_realistic_generator():
    """Session-scoped realistic generator (read-only use)."""
    return SyntheticNIRSGenerator(
        complexity="realistic",
        random_state=42,
    )

@pytest.fixture(scope="session")
def shared_predefined_library():
    """Session-scoped predefined library."""
    return ComponentLibrary.from_predefined(
        ["water", "protein", "lipid"],
        random_state=42,
    )

@pytest.fixture(scope="session")
def sample_wavelengths():
    """Session-scoped sample wavelength array."""
    return np.arange(1000, 2500, 2)

@pytest.fixture(scope="session")
def sample_band():
    """Session-scoped sample NIR band."""
    return NIRBand(
        center=1450,
        sigma=25,
        gamma=3,
        amplitude=0.8,
        name="O-H 1st overtone",
    )

@pytest.fixture(scope="session")
def sample_component():
    """Session-scoped sample spectral component."""
    return SpectralComponent(
        name="water",
        bands=[
            NIRBand(center=1450, sigma=25, gamma=3, amplitude=0.8),
            NIRBand(center=1940, sigma=30, gamma=4, amplitude=1.0),
        ],
        correlation_group=1,
    )

# =============================================================================
# Function-scoped fixtures (fresh per test - for tests that modify state)
# =============================================================================

@pytest.fixture
def simple_generator():
    """Create a simple complexity generator with fixed seed."""
    return SyntheticNIRSGenerator(
        complexity="simple",
        random_state=42,
    )

@pytest.fixture
def realistic_generator():
    """Create a realistic complexity generator with fixed seed."""
    return SyntheticNIRSGenerator(
        complexity="realistic",
        random_state=42,
    )

@pytest.fixture
def complex_generator():
    """Create a complex complexity generator with fixed seed."""
    return SyntheticNIRSGenerator(
        complexity="complex",
        random_state=42,
    )

@pytest.fixture
def predefined_library():
    """Create a library from predefined components."""
    return ComponentLibrary.from_predefined(
        ["water", "protein", "lipid"],
        random_state=42,
    )

@pytest.fixture
def random_library():
    """Create a library with random components."""
    library = ComponentLibrary(random_state=42)
    library.generate_random_library(n_components=3)
    return library
