"""Pytest configuration and fixtures for MecaPy tests."""

import pytest


@pytest.fixture
def sample_material():
    """Fixture providing sample material properties."""
    return {
        "name": "test_steel",
        "density": 7850,
        "elastic_modulus": 210e9,
        "poisson_ratio": 0.3,
        "yield_strength": 250e6,
    }


@pytest.fixture
def sample_beam():
    """Fixture providing a sample beam object."""
    from mecapy.beams import Beam
    return Beam(length=5.0, material="steel")


@pytest.fixture
def sample_gear():
    """Fixture providing a sample gear object."""
    from mecapy.gears import Gear
    return Gear(teeth=20, module=2.5, material="steel")


@pytest.fixture
def sample_shaft():
    """Fixture providing a sample shaft object."""
    from mecapy.shafts import Shaft
    return Shaft(diameter=25.0, length=500.0, material="steel")
