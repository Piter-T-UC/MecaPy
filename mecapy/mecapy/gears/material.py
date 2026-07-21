"""Gear material: a :class:`~mecapy.materials.Material` plus AGMA rating data.

Units follow the gear/AGMA convention: strengths in MPa, hardness as a
Brinell number (HB). A :class:`GearMaterial` adds the two things the
AGMA 2101 allowable-stress fits need — core/surface Brinell hardness and
the AGMA quality grade — that would otherwise clutter the general-purpose
fatigue :class:`~mecapy.materials.Material`. Because it *is* a
``Material``, it drops straight into any gear constructor and works with
``MechaElement`` stress/safety unchanged; :class:`AGMARating` also reads
the hardness and grade off it automatically when they are not passed by
hand.
"""

from ..materials import Material
from .agma_data import allowable_bending_stress, allowable_contact_stress

__all__ = ["GearMaterial"]


class GearMaterial(Material):
    """A gear material carrying Brinell hardness and AGMA quality grade.

    Extends :class:`~mecapy.materials.Material` with the through-hardened
    steel data AGMA 2101 needs: Brinell hardness ``hardness_HB`` and the
    material quality ``grade`` (1 or 2). The allowable bending and contact
    stress numbers are exposed as recomputed properties, so changing the
    hardness or grade updates them immediately.

    Attributes:
        hardness_HB (float): Brinell hardness HB.
        grade (int): AGMA material quality grade, 1 or 2.
    """

    def __init__(self, hardness_HB, base_material="steel", grade=1,
                 **material_kwargs):
        """
        Initialize a gear material.

        Args:
            hardness_HB (float): Brinell hardness HB. Must be strictly
                positive (AGMA through-hardened fits are meaningful over
                roughly 150-400 HB).
            base_material (str): Database material for the non-strength
                properties and strength defaults (default "steel").
            grade (int): AGMA material quality grade, 1 or 2 (default 1).
            **material_kwargs: Any other :class:`~mecapy.materials.Material`
                argument (``ultimate_strength``, ``yield_strength``,
                ``surface_finish``, ``name``, ...), forwarded unchanged.

        Raises:
            ValueError: If ``hardness_HB`` is not strictly positive, the
                grade is not 1 or 2, or any forwarded ``Material``
                argument fails its own validation.
        """
        super().__init__(base_material=base_material, **material_kwargs)
        self.grade = grade
        self.hardness_HB = hardness_HB

    @property
    def hardness_HB(self):
        """float: Brinell hardness HB."""
        return self._hardness_HB

    @hardness_HB.setter
    def hardness_HB(self, value):
        if value is None or value <= 0:
            raise ValueError("Brinell hardness (hardness_HB) must be "
                             "strictly positive")
        self._hardness_HB = value

    @property
    def grade(self):
        """int: AGMA material quality grade, 1 or 2."""
        return self._grade

    @grade.setter
    def grade(self, value):
        if value not in (1, 2):
            raise ValueError("AGMA grade must be 1 or 2")
        self._grade = value

    @property
    def allowable_bending_stress(self):
        """float: Allowable bending stress number St in MPa (AGMA 2101
        through-hardened fit from ``hardness_HB`` and ``grade``)."""
        return allowable_bending_stress(self.hardness_HB, self.grade)

    @property
    def allowable_contact_stress(self):
        """float: Allowable contact stress number Sc in MPa (AGMA 2101
        through-hardened fit from ``hardness_HB`` and ``grade``)."""
        return allowable_contact_stress(self.hardness_HB, self.grade)

    def __repr__(self):
        return (
            f"GearMaterial(name={self.name!r}, "
            f"base_material={self.base_material!r}, "
            f"hardness_HB={self.hardness_HB}, grade={self.grade})"
        )
