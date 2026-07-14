"""Material class and database for MecaPy."""

from .utils import constants


class Material:
    """
    Engineering material with mechanical, thermal and physical properties.

    A ``Material`` bundles all the information about a material in one object.
    It also behaves like a read-only mapping (``material["yield_strength"]``)
    so it can be used interchangeably with the raw property dictionaries.

    Attributes:
        name (str): Material name.
        density (float): Density in kg/m^3.
        elastic_modulus (float): Young's modulus E in Pa.
        shear_modulus (float): Shear modulus G in Pa. Derived from E and
            Poisson's ratio when not supplied.
        poisson_ratio (float): Poisson's ratio.
        yield_strength (float): Yield strength in Pa.
        ultimate_strength (float): Ultimate tensile strength in Pa.
        endurance_limit (float): Fatigue/endurance limit in Pa (may be None).
        hardness_brinell (float): Brinell hardness (HB).
        thermal_expansion (float): Coefficient of thermal expansion in 1/K.
        thermal_conductivity (float): Thermal conductivity in W/(m*K).
        specific_heat (float): Specific heat capacity in J/(kg*K).
        elongation (float): Elongation at break in percent.
    """

    #: Property names carried by every Material instance.
    FIELDS = (
        "density",
        "elastic_modulus",
        "shear_modulus",
        "poisson_ratio",
        "yield_strength",
        "ultimate_strength",
        "endurance_limit",
        "hardness_brinell",
        "thermal_expansion",
        "thermal_conductivity",
        "specific_heat",
        "elongation",
    )

    def __init__(self, name, density=None, elastic_modulus=None,
                 shear_modulus=None, poisson_ratio=None, yield_strength=None,
                 ultimate_strength=None, endurance_limit=None,
                 hardness_brinell=None, thermal_expansion=None,
                 thermal_conductivity=None, specific_heat=None,
                 elongation=None):
        self.name = name
        self.density = density
        self.elastic_modulus = elastic_modulus
        self.poisson_ratio = poisson_ratio
        self.yield_strength = yield_strength
        self.ultimate_strength = ultimate_strength
        self.endurance_limit = endurance_limit
        self.hardness_brinell = hardness_brinell
        self.thermal_expansion = thermal_expansion
        self.thermal_conductivity = thermal_conductivity
        self.specific_heat = specific_heat
        self.elongation = elongation

        # Derive the shear modulus from E and nu when it is not provided.
        if shear_modulus is None and elastic_modulus and poisson_ratio is not None:
            shear_modulus = elastic_modulus / (2 * (1 + poisson_ratio))
        self.shear_modulus = shear_modulus

    @classmethod
    def from_dict(cls, name, properties):
        """Build a Material from a name and a property dictionary."""
        return cls(name=name, **properties)

    def to_dict(self):
        """Return the material properties as a plain dictionary."""
        return {field: getattr(self, field) for field in self.FIELDS}

    # -- Mapping-style access for backward compatibility ----------------
    def __getitem__(self, key):
        if key not in self.FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def get(self, key, default=None):
        """Dictionary-style ``get`` with a default."""
        return getattr(self, key, default)

    def __contains__(self, key):
        return key in self.FIELDS

    def __repr__(self):
        return (
            f"Material(name={self.name!r}, E={self.elastic_modulus}, "
            f"yield={self.yield_strength})"
        )


def get_material(material_name):
    """
    Get a :class:`Material` instance for a named material.

    Args:
        material_name (str): Name of the material.

    Returns:
        Material: The material object with all its properties.

    Raises:
        ValueError: If the material is not found in the database.
    """
    if material_name not in constants.MATERIALS:
        raise ValueError(f"Material '{material_name}' not found in database")
    return Material.from_dict(material_name, constants.MATERIALS[material_name])


def get_material_properties(material_name):
    """
    Get the raw property dictionary for a named material.

    Kept for backward compatibility; prefer :func:`get_material`.

    Args:
        material_name (str): Name of the material.

    Returns:
        dict: Dictionary of material properties.

    Raises:
        ValueError: If the material is not found in the database.
    """
    if material_name not in constants.MATERIALS:
        raise ValueError(f"Material '{material_name}' not found in database")
    return dict(constants.MATERIALS[material_name])


def get_available_materials():
    """Get the list of material names available in the database."""
    return list(constants.MATERIALS.keys())


def add_custom_material(name, properties):
    """
    Add a custom material to the database.

    Args:
        name (str): Material name.
        properties (dict): Dictionary of material properties. Accepts any of
            the keys listed in :attr:`Material.FIELDS`.
    """
    constants.MATERIALS[name] = dict(properties)
