"""Flat-belt and V-belt material/section data.

Digitized from Shigley's *Mechanical Engineering Design* Ch. 17 (Table 17-2
region, flat-belt materials; Tables 17-9/17-11 region, standard V-belt
sections), metric-converted. Shigley's own tables are printed in US
customary units, so every row carries the original US value in a comment;
some fields (noted per row) are engineering-handbook estimates rather than
transcribed table cells and should be checked against a printed copy of the
book before using them for a final design.

Fields per flat-belt material:
    ``mu``               — dry coefficient of friction on a steel/cast-iron
                            pulley (dimensionless)
    ``density``           — belt material density in kg/m^3
    ``allowable_stress``  — allowable working stress in MPa (Shigley's Fa,
                            an allowable tension per unit width, divided by
                            the reference thickness used to derive it)
    ``min_pulley_diameter`` — smallest recommended pulley diameter in mm

Fields per V-belt section:
    ``width``              — top width in mm
    ``thickness``          — section height in mm
    ``min_sheave_diameter`` — smallest recommended sheave diameter in mm
    ``mass_per_length``    — belt mass per unit length in kg/m
    ``length_addend``      — mm added to the nominal/inside belt length to
                             get the pitch length Lp (Table 17-11 region)
"""

FLAT_BELT_MATERIALS = {
    # mu = 0.4 (leather on cast iron, Shigley Table 17-2 region); Fa = 30
    # lbf/in over an assumed single-ply thickness of 0.20 in gives an
    # allowable stress of 150 psi. density assumed 0.035 lbf/in^3.
    # ESTIMATED thickness/density/min diameter -- verify against the printed
    # table.
    "leather": {"mu": 0.4, "density": 969.0, "allowable_stress": 1.03,
                "min_pulley_diameter": 101.6},
    # Polyamide A-3 (Shigley Table 17-2, per Ex. 17-1/Prob. 17-6): t = 0.13
    # in, Fa = 100 lbf/in, density = 0.042 lbf/in^3, f = 0.8.
    # allowable_stress = Fa/t = 769 psi = 5.30 MPa.
    "polyamide": {"mu": 0.8, "density": 1162.6, "allowable_stress": 5.30,
                  "min_pulley_diameter": 76.2},
    # mu = 0.7, density = 1150 kg/m^3, allowable_stress = 4.0 MPa: typical
    # handbook values for a fabric/urethane flat belt, NOT transcribed from
    # a specific Shigley table row -- verify against the printed table.
    "urethane": {"mu": 0.7, "density": 1150.0, "allowable_stress": 4.0,
                 "min_pulley_diameter": 40.0},
}


def get_flat_belt_material(name):
    """
    Look up design values for a named flat-belt material.

    Args:
        name (str): Material name, one of the keys of
            :data:`FLAT_BELT_MATERIALS` (e.g. "polyamide"). Case-insensitive.

    Returns:
        dict: ``mu``, ``density`` (kg/m^3), ``allowable_stress`` (MPa) and
        ``min_pulley_diameter`` (mm) for the material.

    Raises:
        ValueError: If ``name`` is not a known flat-belt material.
    """
    key = name.lower() if isinstance(name, str) else name
    if key not in FLAT_BELT_MATERIALS:
        available = ", ".join(sorted(FLAT_BELT_MATERIALS))
        raise ValueError(f"Unknown flat belt material {name!r}. Available: {available}")
    return FLAT_BELT_MATERIALS[key]


# Standard classical V-belt cross sections (Shigley Table 17-9 region).
# width/thickness are the standard RMA/ARPM section dimensions (confirmed:
# E section thickness = 1.000 in in the Ch. 17 solutions manual). Every
# mass_per_length below is width*thickness*969 kg/m^3, where 969 kg/m^3 is
# back-calculated from the three sections whose centrifugal-tension factor
# Kc is confirmed in the solutions manual (B: Kc=0.965, C: Kc=1.716,
# D: Kc=3.498, via Fc[N] = Kc*0.172365*V[m/s]^2 -> density 962-997 kg/m^3);
# A and E are extrapolated from that same density, not independently
# confirmed -- verify against the printed table.
V_BELT_SECTIONS = {
    # width 0.500 in, thickness 0.312 in, min sheave dia 3.0 in
    "A": {"width": 12.70, "thickness": 7.925, "min_sheave_diameter": 76.2,
          "mass_per_length": 0.0976, "length_addend": 33.02},
    # width 0.660 in, thickness 0.406 in, min sheave dia 5.4 in (confirmed,
    # Prob. 17-18: d = 5.4 in with B85 belts)
    "B": {"width": 16.764, "thickness": 10.312, "min_sheave_diameter": 137.16,
          "mass_per_length": 0.1663, "length_addend": 45.72},
    # width 0.910 in, thickness 0.531 in, min sheave dia 9.0 in (Prob. 17-20:
    # an 11 in sheave "exceeds C-section minimum diameter")
    "C": {"width": 23.114, "thickness": 13.487, "min_sheave_diameter": 228.6,
          "mass_per_length": 0.2959, "length_addend": 73.66},
    # width 1.250 in, thickness 0.750 in, min sheave dia 13.0 in (Prob.
    # 17-20: 11 in sheave "precludes D- and E-section belts")
    "D": {"width": 31.75, "thickness": 19.05, "min_sheave_diameter": 330.2,
          "mass_per_length": 0.6030, "length_addend": 83.82},
    # width 1.500 in, thickness 1.000 in (confirmed, Prob. 17-21: E210 belt
    # thickness = 1 in), min sheave dia 21.6 in
    "E": {"width": 38.10, "thickness": 25.40, "min_sheave_diameter": 548.64,
          "mass_per_length": 0.9387, "length_addend": 114.30},
}


def get_v_belt_section(section):
    """
    Look up standard dimensions for a named V-belt section.

    Args:
        section (str): Section letter, one of the keys of
            :data:`V_BELT_SECTIONS` (e.g. "B"). Case-insensitive.

    Returns:
        dict: ``width`` (mm), ``thickness`` (mm), ``min_sheave_diameter``
        (mm), ``mass_per_length`` (kg/m) and ``length_addend`` (mm).

    Raises:
        ValueError: If ``section`` is not a known V-belt section.
    """
    key = section.upper() if isinstance(section, str) else section
    if key not in V_BELT_SECTIONS:
        available = ", ".join(sorted(V_BELT_SECTIONS))
        raise ValueError(f"Unknown V-belt section {section!r}. Available: {available}")
    return V_BELT_SECTIONS[key]


# Effective groove friction for a V-belt at the standard 36-degree groove
# angle (Shigley Eq. 17-24), confirmed throughout the Ch. 17 solutions
# manual: exp(f_e * phi) computed with f_e = 0.5123 exactly reproduces the
# worked V-belt examples (e.g. Prob. 17-17: exp[0.5123(2.9570)] = 4.5489).
V_BELT_EFFECTIVE_MU = 0.5123
