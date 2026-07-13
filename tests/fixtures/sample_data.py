"""Sample test data for MecaPy tests."""


# Sample beam configurations
BEAM_CONFIGURATIONS = {
    "cantilever": {
        "length": 3.0,
        "material": "steel",
        "section": {"width": 0.1, "height": 0.2},
    },
    "simply_supported": {
        "length": 5.0,
        "material": "aluminum",
        "section": {"diameter": 0.05},
    },
}

# Sample gear specifications
GEAR_SPECIFICATIONS = {
    "spur_gear": {
        "teeth": 20,
        "module": 2.5,
        "material": "steel",
        "pitch_diameter": 50,
    },
    "helical_gear": {
        "teeth": 30,
        "module": 3.0,
        "material": "cast_iron",
        "pitch_diameter": 90,
    },
}

# Sample shaft parameters
SHAFT_PARAMETERS = {
    "transmission_shaft": {
        "diameter": 25.0,
        "length": 500.0,
        "material": "steel",
        "rpm": 1500,
    },
    "drive_shaft": {
        "diameter": 30.0,
        "length": 1000.0,
        "material": "steel",
        "rpm": 1200,
    },
}

# Sample bearing specifications
BEARING_SPECIFICATIONS = {
    "ball_bearing": {
        "bore_diameter": 10.0,
        "outer_diameter": 26.0,
        "width": 8.0,
        "bearing_type": "ball",
    },
    "roller_bearing": {
        "bore_diameter": 20.0,
        "outer_diameter": 52.0,
        "width": 15.0,
        "bearing_type": "roller",
    },
}
