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


@pytest.fixture
def sample_disc_clutch():
    """Fixture providing a sample disc clutch object."""
    from mecapy.clutches import DiscClutch
    return DiscClutch(outer_diameter=300, inner_diameter=225, mu=0.25, n_faces=2)


@pytest.fixture
def sample_internal_shoe_brake():
    """Fixture providing the Shigley Ex. 16-2 internal shoe brake."""
    import math
    from mecapy.brakes import InternalShoeBrake
    return InternalShoeBrake(drum_radius=150, face_width=32,
                             pivot_distance=math.sqrt(112 ** 2 + 50 ** 2),
                             theta1=0, theta2=126, actuation_arm=212, mu=0.32)


@pytest.fixture
def sample_flywheel():
    """Fixture providing a sample flywheel object."""
    from mecapy.wheels import Flywheel
    return Flywheel(outer_radius=0.3, thickness=0.05, material="steel")


@pytest.fixture
def sample_flat_belt():
    """Fixture providing a sample flat belt object."""
    from mecapy.belts import FlatBelt
    return FlatBelt(width=50.0, thickness=5.0, driver_diameter=100.0,
                    driven_diameter=300.0, center_distance=800.0,
                    belt_material="polyamide")


@pytest.fixture
def sample_v_belt():
    """Fixture providing a sample V-belt object."""
    from mecapy.belts import VBelt
    return VBelt(section="B", driver_diameter=150.0, driven_diameter=300.0,
                center_distance=600.0)


@pytest.fixture
def sample_roller_chain():
    """Fixture providing a sample roller chain object."""
    from mecapy.chains import RollerChain
    return RollerChain(chain_number=60, driver_teeth=17, driven_teeth=51,
                       center_distance=500.0)
