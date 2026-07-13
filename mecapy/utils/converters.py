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


def kg_to_newtons(mass):
    """Convert mass in kilograms to force in newtons."""
    return mass * constants.G


def newtons_to_kg(force):
    """Convert force in newtons to mass in kilograms."""
    return force / constants.G
