"""ANSI standard roller chain data (Shigley Table 17-19 region).

Digitized from the ANSI B29.1 standard roller chain series that Shigley's
*Mechanical Engineering Design* Table 17-19 reproduces. Values are
cross-checked against two independent chain manufacturer catalogs (Renold
Jeffrey, AmesWeb) that quote the same ANSI standard; every row carries the
original US-customary value in a comment for a final check against a
printed copy of the book.

Fields per chain number:
    ``pitch``               — chain pitch in mm
    ``width``                — minimum inside width in mm
    ``min_tensile_strength`` — minimum (average) ultimate tensile strength
                               in N
    ``mass_per_length``      — chain mass per unit length in kg/m
    ``kr``                   — Eq. 17-33 roller-bushing wear constant
                               (29 for chain no. 25/35, 3.4 for no. 41,
                               17 for no. 40 and no. 50 upward)
"""

ANSI_ROLLER_CHAINS = {
    # pitch 0.250 in, width 0.122 in, tensile 780 lbf, weight 0.09 lbf/ft
    "25": {"pitch": 6.350, "width": 3.099, "min_tensile_strength": 3469.6,
           "mass_per_length": 0.1339, "kr": 29},
    # pitch 0.375 in, width 0.184 in, tensile 1760 lbf, weight 0.24 lbf/ft
    "35": {"pitch": 9.525, "width": 4.674, "min_tensile_strength": 7828.9,
           "mass_per_length": 0.3572, "kr": 29},
    # pitch 0.500 in, width 0.309 in, tensile 3125 lbf, weight 0.40 lbf/ft
    "40": {"pitch": 12.700, "width": 7.849, "min_tensile_strength": 13900.7,
           "mass_per_length": 0.5953, "kr": 17},
    # pitch 0.500 in, width 0.246 in, tensile 1500 lbf, weight 0.28 lbf/ft
    # (lightweight series)
    "41": {"pitch": 12.700, "width": 6.248, "min_tensile_strength": 6672.3,
           "mass_per_length": 0.4167, "kr": 3.4},
    # pitch 0.625 in, width 0.370 in, tensile 4880 lbf, weight 0.67 lbf/ft
    "50": {"pitch": 15.875, "width": 9.398, "min_tensile_strength": 21707.3,
           "mass_per_length": 0.9971, "kr": 17},
    # pitch 0.750 in, width 0.495 in, tensile 7030 lbf, weight 0.99 lbf/ft
    # (confirmed: Shigley Prob. 17-24, Kr = 17 for this chain number)
    "60": {"pitch": 19.050, "width": 12.573, "min_tensile_strength": 31271.0,
           "mass_per_length": 1.4733, "kr": 17},
    # pitch 1.000 in, width 0.620 in, tensile 12,500 lbf, weight 1.86 lbf/ft
    "80": {"pitch": 25.400, "width": 15.748, "min_tensile_strength": 55602.7,
           "mass_per_length": 2.7680, "kr": 17},
    # pitch 1.250 in, width 0.744 in, tensile 19,530 lbf, weight 2.82 lbf/ft
    "100": {"pitch": 31.750, "width": 18.898, "min_tensile_strength": 86865.3,
            "mass_per_length": 4.1966, "kr": 17},
    # pitch 1.500 in, width 0.993 in, tensile 28,125 lbf, weight 3.83 lbf/ft
    "120": {"pitch": 38.100, "width": 25.222, "min_tensile_strength": 125106.1,
            "mass_per_length": 5.6996, "kr": 17},
    # pitch 1.750 in, width 0.993 in, tensile 38,280 lbf, weight 5.24 lbf/ft
    # (confirmed: Shigley Ex. 17-5 selects a triple-strand no. 140 chain;
    # pitch verified in Prob. 17-23)
    "140": {"pitch": 44.450, "width": 25.222, "min_tensile_strength": 170262.7,
            "mass_per_length": 7.7980, "kr": 17},
    # pitch 2.000 in, width 1.242 in, tensile 50,000 lbf, weight 6.99 lbf/ft
    "160": {"pitch": 50.800, "width": 31.547, "min_tensile_strength": 222411.0,
            "mass_per_length": 10.4022, "kr": 17},
    # pitch 2.250 in, width 1.397 in, tensile 63,280 lbf, weight 9.34 lbf/ft
    "180": {"pitch": 57.150, "width": 35.484, "min_tensile_strength": 281475.1,
            "mass_per_length": 13.8994, "kr": 17},
    # pitch 2.500 in, width 1.490 in, tensile 78,125 lbf, weight 11.59 lbf/ft
    "200": {"pitch": 63.500, "width": 37.846, "min_tensile_strength": 347517.4,
            "mass_per_length": 17.2477, "kr": 17},
    # pitch 3.000 in, width 1.864 in, tensile 112,500 lbf, weight 16.75 lbf/ft
    "240": {"pitch": 76.200, "width": 47.346, "min_tensile_strength": 500424.7,
            "mass_per_length": 24.9267, "kr": 17},
}

# Table 17-22 (multi-strand factor K2), confirmed throughout the Ch. 17
# solutions manual (e.g. Prob. 17-27: K2 = 1, 1.7, 2.5, 3.3 for 1 to 4
# strands).
STRAND_FACTORS = {1: 1.0, 2: 1.7, 3: 2.5, 4: 3.3}


def get_chain(chain_number):
    """
    Look up dimensions and strength data for a named ANSI roller chain.

    Args:
        chain_number (int or str): Chain number, one of the keys of
            :data:`ANSI_ROLLER_CHAINS` (e.g. 60 or "60").

    Returns:
        dict: ``pitch`` (mm), ``width`` (mm), ``min_tensile_strength`` (N),
        ``mass_per_length`` (kg/m) and ``kr``.

    Raises:
        ValueError: If ``chain_number`` is not a known chain number.
    """
    key = str(chain_number)
    if key not in ANSI_ROLLER_CHAINS:
        available = ", ".join(sorted(ANSI_ROLLER_CHAINS, key=int))
        raise ValueError(f"Unknown chain number {chain_number!r}. Available: {available}")
    return ANSI_ROLLER_CHAINS[key]
