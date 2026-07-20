"""Physical constants and material properties for MecaPy."""

# Physical Constants
G = 9.81  # Gravitational acceleration (m/s^2)
PI = 3.14159265359
RPM_TO_RAD_S = 2 * 3.14159265359 / 60  # rev/min to rad/s

# Material Properties Database
MATERIALS = {
    "steel": {
        "density": 7850,  # kg/m^3
        "elastic_modulus": 210e9,  # Pa
        "poisson_ratio": 0.3,
        "yield_strength": 250e6,  # Pa (typical mild steel)
        "ultimate_strength": 400e6,  # Pa (typical mild steel, A36-like)
        "shear_modulus": 81e9,  # Pa
        "specific_heat": 500,  # J/(kg*K)
    },
    "aluminum": {
        "density": 2700,  # kg/m^3
        "elastic_modulus": 69e9,  # Pa
        "poisson_ratio": 0.33,
        "yield_strength": 110e6,  # Pa (typical aluminum)
        "ultimate_strength": 200e6,  # Pa
        "shear_modulus": 26e9,  # Pa
        "specific_heat": 900,  # J/(kg*K)
    },
    "copper": {
        "density": 8960,  # kg/m^3
        "elastic_modulus": 110e9,  # Pa
        "poisson_ratio": 0.34,
        "yield_strength": 200e6,  # Pa
        "ultimate_strength": 250e6,  # Pa
        "shear_modulus": 42e9,  # Pa
        "specific_heat": 385,  # J/(kg*K)
    },
    "cast_iron": {
        "density": 7200,  # kg/m^3
        "elastic_modulus": 160e9,  # Pa
        "poisson_ratio": 0.25,
        "yield_strength": 180e6,  # Pa
        "ultimate_strength": 250e6,  # Pa (gray iron, tensile)
        "shear_modulus": 64e9,  # Pa
        "specific_heat": 500,  # J/(kg*K)
    },
    "bronze": {
        "density": 8800,  # kg/m^3
        "elastic_modulus": 110e9,  # Pa
        "poisson_ratio": 0.34,
        "yield_strength": 170e6,  # Pa (typical gear bronze, e.g. C90700)
        "ultimate_strength": 300e6,  # Pa (C90700-like)
        "shear_modulus": 41e9,  # Pa
        "specific_heat": 380,  # J/(kg*K)
    },
}

# Standard stress and strain values
SAFETY_FACTOR_STATIC = 2.0
SAFETY_FACTOR_DYNAMIC = 3.0

# Unit conversion factors
MM_TO_M = 1e-3
M_TO_MM = 1e3
KPA_TO_PA = 1e3
MPA_TO_PA = 1e6
KPA_TO_MPA = 1e-3
IN_TO_MM = 25.4
LBF_TO_N = 4.4482216152605
PSI_TO_MPA = 6.894757293168e-3
HP_TO_KW = 0.7456998715823
