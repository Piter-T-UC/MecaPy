"""Unit conversion functions for MecaPy."""

from . import constants


def mm_to_m(value):
    """Convert millimeters to meters."""
    return value * constants.MM_TO_M


def m_to_mm(value):
    """Convert meters to millimeters."""
    return value * constants.M_TO_MM


def kpa_to_pa(value):
    """Convert kilopascals to pascals."""
    return value * constants.KPA_TO_PA


def mpa_to_pa(value):
    """Convert megapascals to pascals."""
    return value * constants.MPA_TO_PA


def pa_to_mpa(value):
    """Convert pascals to megapascals."""
    return value / constants.MPA_TO_PA


def pa_to_kpa(value):
    """Convert pascals to kilopascals."""
    return value / constants.KPA_TO_PA


def in_to_mm(value):
    """Convert inches to millimeters."""
    return value * constants.IN_TO_MM


def mm_to_in(value):
    """Convert millimeters to inches."""
    return value / constants.IN_TO_MM


def lbf_to_n(value):
    """Convert pound-force to newtons."""
    return value * constants.LBF_TO_N


def n_to_lbf(value):
    """Convert newtons to pound-force."""
    return value / constants.LBF_TO_N


def psi_to_mpa(value):
    """Convert pounds per square inch to megapascals."""
    return value * constants.PSI_TO_MPA


def mpa_to_psi(value):
    """Convert megapascals to pounds per square inch."""
    return value / constants.PSI_TO_MPA


def hp_to_kw(value):
    """Convert mechanical horsepower to kilowatts."""
    return value * constants.HP_TO_KW


def kw_to_hp(value):
    """Convert kilowatts to mechanical horsepower."""
    return value / constants.HP_TO_KW


def kg_to_newtons(mass):
    """Convert mass in kilograms to force in newtons."""
    return mass * constants.G


def newtons_to_kg(force):
    """Convert force in newtons to mass in kilograms."""
    return force / constants.G
