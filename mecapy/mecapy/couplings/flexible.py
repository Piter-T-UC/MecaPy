"""Flexible coupling selection and checking module.

Units convention: torques in N*mm, speeds in rev/min, angular
misalignment in degrees, offsets in mm, torsional stiffness in
N*mm/rad.

Unlike the geometry-driven elements, a flexible coupling (jaw, disc,
gear, elastomeric, ...) is normally SELECTED from a catalog by its
ratings rather than sized from first principles, so this class models
the ratings — torque, speed and the three misalignment limits — and
checks a duty point against them.
"""

import math

from ..base import MechaElement


class FlexibleCoupling(MechaElement):
    """
    Catalog-style flexible coupling checked against its ratings.

    Attributes:
        torque_rating (float): Continuous torque rating in N*mm.
        max_speed_rpm (float): Maximum operating speed in rev/min.
        max_angular_misalignment (float): Allowable angular misalignment
            in degrees.
        max_parallel_offset (float): Allowable parallel (radial) offset
            in mm.
        max_axial_movement (float): Allowable axial movement in mm.
        torsional_stiffness (float): Torsional stiffness in N*mm/rad,
            or None.
    """

    def __init__(self, torque_rating, max_speed_rpm, max_angular_misalignment,
                 max_parallel_offset, max_axial_movement,
                 torsional_stiffness=None, material="steel", name=None):
        """
        Initialize a FlexibleCoupling object.

        Args:
            torque_rating (float): Continuous torque rating in N*mm.
            max_speed_rpm (float): Maximum operating speed in rev/min.
            max_angular_misalignment (float): Allowable angular
                misalignment in degrees.
            max_parallel_offset (float): Allowable parallel offset in mm.
            max_axial_movement (float): Allowable axial movement in mm.
            torsional_stiffness (float): Optional torsional stiffness in
                N*mm/rad, needed for the windup methods.
            material (str): Material type (default: "steel").
            name (str): Optional identifier for the coupling.

        Raises:
            ValueError: For a non-positive rating, speed or stiffness, or
                a negative misalignment limit.
        """
        super().__init__(name=name, material=material)
        if torque_rating <= 0:
            raise ValueError("Torque rating must be strictly positive")
        if max_speed_rpm <= 0:
            raise ValueError("Maximum speed must be strictly positive")
        if max_angular_misalignment < 0:
            raise ValueError("Angular misalignment limit must be non-negative")
        if max_parallel_offset < 0:
            raise ValueError("Parallel offset limit must be non-negative")
        if max_axial_movement < 0:
            raise ValueError("Axial movement limit must be non-negative")
        if torsional_stiffness is not None and torsional_stiffness <= 0:
            raise ValueError("Torsional stiffness must be strictly positive")

        self.torque_rating = torque_rating
        self.max_speed_rpm = max_speed_rpm
        self.max_angular_misalignment = max_angular_misalignment
        self.max_parallel_offset = max_parallel_offset
        self.max_axial_movement = max_axial_movement
        self.torsional_stiffness = torsional_stiffness

    # ---- Torque and speed checks ----

    @staticmethod
    def service_torque(nominal_torque, service_factor=1.0):
        """
        Design torque including a service factor.

        ::

            T_design = Ks * T_nominal

        Args:
            nominal_torque (float): Nominal (steady) torque in N*mm.
            service_factor (float): Service factor Ks for shock/duty
                (default 1.0; 1.5-3 typical for shock loads).

        Returns:
            float: Design torque in N*mm.

        Raises:
            ValueError: If the torque or the factor is not strictly
                positive.
        """
        if nominal_torque <= 0:
            raise ValueError("Nominal torque must be strictly positive")
        if service_factor <= 0:
            raise ValueError("Service factor must be strictly positive")
        return service_factor * nominal_torque

    def torque_safety_factor(self, torque):
        """
        Ratio of the torque rating to the applied torque.

        Args:
            torque (float): Applied (design) torque in N*mm.

        Returns:
            float: Rating / applied. Values greater than 1 mean the
            coupling is within its rating.

        Raises:
            ValueError: If ``torque`` is not strictly positive.
        """
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")
        return self.torque_rating / torque

    def speed_safety_factor(self, speed_rpm):
        """
        Ratio of the speed rating to the operating speed.

        Args:
            speed_rpm (float): Operating speed in rev/min.

        Returns:
            float: Rating / operating speed.

        Raises:
            ValueError: If ``speed_rpm`` is not strictly positive.
        """
        if speed_rpm <= 0:
            raise ValueError("Speed must be strictly positive")
        return self.max_speed_rpm / speed_rpm

    # ---- Misalignment checks ----

    def check_misalignment(self, angular=0.0, parallel=0.0, axial=0.0):
        """
        Check a misalignment condition against the coupling limits.

        Args:
            angular (float): Angular misalignment in degrees.
            parallel (float): Parallel (radial) offset in mm.
            axial (float): Axial movement in mm.

        Returns:
            bool: True when every component is within its limit.

        Raises:
            ValueError: If any component is negative.
        """
        self._check_misalignment_signs(angular, parallel, axial)
        return (angular <= self.max_angular_misalignment
                and parallel <= self.max_parallel_offset
                and axial <= self.max_axial_movement)

    def validate_misalignment(self, angular=0.0, parallel=0.0, axial=0.0):
        """
        Like :meth:`check_misalignment` but raises naming the violation.

        Args:
            angular (float): Angular misalignment in degrees.
            parallel (float): Parallel (radial) offset in mm.
            axial (float): Axial movement in mm.

        Raises:
            ValueError: If a component is negative, or exceeds its limit
                (the message names the violated limit).
        """
        self._check_misalignment_signs(angular, parallel, axial)
        if angular > self.max_angular_misalignment:
            raise ValueError(
                f"Angular misalignment {angular} deg exceeds the limit "
                f"{self.max_angular_misalignment} deg"
            )
        if parallel > self.max_parallel_offset:
            raise ValueError(
                f"Parallel offset {parallel} mm exceeds the limit "
                f"{self.max_parallel_offset} mm"
            )
        if axial > self.max_axial_movement:
            raise ValueError(
                f"Axial movement {axial} mm exceeds the limit "
                f"{self.max_axial_movement} mm"
            )

    # ---- Torsional windup ----

    def torsional_deflection(self, torque):
        """
        Torsional windup angle across the coupling.

        ::

            theta = T / k_t

        Args:
            torque (float): Applied torque in N*mm.

        Returns:
            float: Windup angle in rad.

        Raises:
            ValueError: If ``torsional_stiffness`` was not given, or
                ``torque`` is not strictly positive.
        """
        if torque <= 0:
            raise ValueError("Torque must be strictly positive")
        if self.torsional_stiffness is None:
            raise ValueError("torsional_stiffness is required for windup calculations")
        return torque / self.torsional_stiffness

    def windup_angle_degrees(self, torque):
        """
        Torsional windup angle in degrees.

        Args:
            torque (float): Applied torque in N*mm.

        Returns:
            float: Windup angle in degrees.

        Raises:
            ValueError: See :meth:`torsional_deflection`.
        """
        return math.degrees(self.torsional_deflection(torque))

    # ---- Internal helpers ----

    @staticmethod
    def _check_misalignment_signs(angular, parallel, axial):
        """Validate that misalignment components are non-negative."""
        if angular < 0 or parallel < 0 or axial < 0:
            raise ValueError("Misalignment components must be non-negative")
