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
def sample_ball_bearing():
    """Fixture providing a 6205-style deep-groove ball bearing."""
    from mecapy.bearings import Bearing
    return Bearing(bore_diameter=25.0, outer_diameter=52.0, width=15.0,
                   C10=35000.0, C0=14000.0)


@pytest.fixture
def sample_journal_bearing():
    """Fixture providing a journal bearing with P = 1 MPa and S = 0.5."""
    from mecapy.bearings import JournalBearing
    return JournalBearing(radius=25.0, clearance=0.025, length=50.0,
                          speed=25.0, load=2500.0, viscosity=20.0)


@pytest.fixture
def sample_bushing():
    """Fixture providing a boundary-lubricated cast-bronze bushing."""
    from mecapy.bearings import PlainBearing
    return PlainBearing(bore_diameter=25.0, length=25.0, load=1000.0,
                        speed=5.0, bushing_material="cast_bronze")


@pytest.fixture
def sample_thrust_bearing():
    """Fixture providing an eight-pad tapered-land thrust bearing."""
    from mecapy.bearings import ThrustBearing
    return ThrustBearing(inner_radius=50.0, outer_radius=100.0, n_pads=8,
                         speed=30.0, load=20000.0, viscosity=30.0)


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
