"""Physical constants and material properties for MecaPy."""

# Physical Constants
G = 9.81  # Gravitational acceleration (m/s^2)
PI = 3.14159265359

# Material Properties Database
#
# All values are in SI base units:
#   density              kg/m^3
#   elastic_modulus (E)  Pa
#   shear_modulus (G)    Pa
#   poisson_ratio        -
#   yield_strength       Pa
#   ultimate_strength    Pa
#   endurance_limit      Pa
#   hardness_brinell     HB
#   thermal_expansion    1/K
#   thermal_conductivity W/(m*K)
#   specific_heat        J/(kg*K)
#   elongation           % (at break)
MATERIALS = {
    "steel": {
        "density": 7850,
        "elastic_modulus": 210e9,
        "shear_modulus": 81e9,
        "poisson_ratio": 0.3,
        "yield_strength": 250e6,
        "ultimate_strength": 400e6,
        "endurance_limit": 200e6,
        "hardness_brinell": 120,
        "thermal_expansion": 11.7e-6,
        "thermal_conductivity": 50.0,
        "specific_heat": 486,
        "elongation": 25.0,
    },
    "aluminum": {
        "density": 2700,
        "elastic_modulus": 69e9,
        "shear_modulus": 26e9,
        "poisson_ratio": 0.33,
        "yield_strength": 110e6,
        "ultimate_strength": 150e6,
        "endurance_limit": 96e6,
        "hardness_brinell": 30,
        "thermal_expansion": 23.6e-6,
        "thermal_conductivity": 167.0,
        "specific_heat": 896,
        "elongation": 12.0,
    },
    "copper": {
        "density": 8960,
        "elastic_modulus": 110e9,
        "shear_modulus": 41e9,
        "poisson_ratio": 0.34,
        "yield_strength": 200e6,
        "ultimate_strength": 220e6,
        "endurance_limit": None,
        "hardness_brinell": 50,
        "thermal_expansion": 16.5e-6,
        "thermal_conductivity": 401.0,
        "specific_heat": 385,
        "elongation": 30.0,
    },
    "cast_iron": {
        "density": 7200,
        "elastic_modulus": 160e9,
        "shear_modulus": 64e9,
        "poisson_ratio": 0.25,
        "yield_strength": 180e6,
        "ultimate_strength": 240e6,
        "endurance_limit": 100e6,
        "hardness_brinell": 190,
        "thermal_expansion": 10.5e-6,
        "thermal_conductivity": 55.0,
        "specific_heat": 490,
        "elongation": 0.6,
    },
}

# Standard safety factors
SAFETY_FACTOR_STATIC = 2.0
SAFETY_FACTOR_DYNAMIC = 3.0

# Unit conversion factors
MM_TO_M = 1e-3
M_TO_MM = 1e3
KPA_TO_PA = 1e3
MPA_TO_PA = 1e6
