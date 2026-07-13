"""Material database and properties for MecaPy."""

from .utils import constants


def get_material_properties(material_name):
    """
    Get properties for a specific material.

    Args:
        material_name (str): Name of the material

    Returns:
        dict: Dictionary of material properties

    Raises:
        ValueError: If material is not found in database
    """
    if material_name not in constants.MATERIALS:
        raise ValueError(f"Material '{material_name}' not found in database")
    return constants.MATERIALS[material_name]


def get_available_materials():
    """Get list of available materials in database."""
    return list(constants.MATERIALS.keys())


def add_custom_material(name, properties):
    """
    Add a custom material to the database.

    Args:
        name (str): Material name
        properties (dict): Dictionary of material properties
    """
    constants.MATERIALS[name] = properties
