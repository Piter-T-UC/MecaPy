"""Gear rack (rack-and-pinion) module.

A rack is the limiting case of a gear with infinite pitch radius: its
pitch "circle" is a straight line and the involute tooth flank becomes a
straight flank at the pressure angle. It meshes with a spur (or helical)
pinion of the same module and pressure angle.
"""

import math

from ..base import MechaElement


class Rack(MechaElement):
    """
    Straight gear rack.

    Attributes:
        module (float): Module in mm.
        pressure_angle (float): Pressure angle in degrees.
        length (float): Rack length in mm (None if not set).
        face_width (float): Face width in mm (None if not set).
        material (str): Material type.
    """

    def __init__(self, module=2.0, pressure_angle=20.0, length=10,
                 face_width=None, material="steel", name=None,
                 diametral_pitch=None):
        """
        Initialize a Rack object.

        Args:
            module (float): Module in mm. Give exactly one of ``module``
                or ``diametral_pitch``.
            pressure_angle (float): Pressure angle in degrees (default 20).
            length (float): Optional rack length in mm.
            face_width (float): Optional face width in mm.
            material (str): Material type (default: "steel").
            name (str): Optional identifier.
            diametral_pitch (float): US-customary alternative to
                ``module``, in teeth per inch.

        Raises:
            ValueError: For non-physical inputs.
        """
        super().__init__(name=name, material=material)
        if (module is None) == (diametral_pitch is None):
            raise ValueError(
                "Give exactly one of 'module' (mm) or 'diametral_pitch' (1/in)"
            )
        if diametral_pitch is not None:
            if diametral_pitch <= 0:
                raise ValueError("Diametral pitch must be strictly positive")
            module = 25.4 / diametral_pitch
        if module <= 0:
            raise ValueError("Module must be strictly positive")
        if not 0 < pressure_angle < 45:
            raise ValueError("Pressure angle must be between 0 and 45 degrees")
        if length is not None and length <= 0:
            raise ValueError("Length must be strictly positive")
        if face_width is not None and face_width <= 0:
            raise ValueError("Face width must be strictly positive")
        self.module = module
        self.pressure_angle = pressure_angle
        self.length = length
        self.face_width = face_width

    @property
    def circular_pitch(self):
        """float: Linear pitch between teeth in mm (pi * module)."""
        return math.pi * self.module

    @property
    def addendum(self):
        """float: Addendum in mm (equal to the module)."""
        return self.module

    @property
    def dedendum(self):
        """float: Dedendum in mm (1.25 * module)."""
        return 1.25 * self.module

    @property
    def n_teeth(self):
        """float: Number of teeth over the rack length.

        Raises:
            ValueError: If ``length`` was not set.
        """
        if self.length is None:
            raise ValueError("Rack length must be set to count teeth")
        return self.length / self.circular_pitch

    def travel_per_pinion_rev(self, pinion):
        """
        Linear rack travel for one revolution of the mating pinion.

        Args:
            pinion (CylindricalGear): Mating pinion.

        Returns:
            float: Travel in mm (pi * pinion pitch diameter).
        """
        return math.pi * pinion.pitch_diameter

    def linear_velocity(self, pinion, speed_rpm):
        """
        Rack linear velocity for a given pinion speed.

        Args:
            pinion (CylindricalGear): Mating pinion.
            speed_rpm (float): Pinion speed in rpm.

        Returns:
            float: Rack velocity in m/s.
        """
        return self.travel_per_pinion_rev(pinion) * speed_rpm / 60000.0

    def force_report(self, pinion, power_kw, speed_rpm):
        """
        Forces the rack reacts from its driving pinion.

        A rack is a linear (terminal) member: it carries a driving
        (tangential) force equal to the pinion tangential force and a
        separating force ``Ft * tan(alpha)``, but transmits no torque.

        Args:
            pinion (CylindricalGear): Driving pinion.
            power_kw (float): Transmitted power in kW.
            speed_rpm (float): Pinion rotational speed in rpm.

        Returns:
            dict: With keys ``Ft``, ``Fr`` (N), ``linear_velocity``
            (m/s) and ``torque`` (always ``None`` — a rack has no
            torque).

        Raises:
            ValueError: If power or speed is not strictly positive.
        """
        ft = pinion.tangential_force(power_kw, speed_rpm)
        fr = ft * math.tan(math.radians(self.pressure_angle))
        return {
            "Ft": ft,
            "Fr": fr,
            "torque": None,
            "linear_velocity": self.linear_velocity(pinion, speed_rpm),
        }

    def describe_forces(self, pinion, power_kw, speed_rpm):
        """
        Multi-line report of the forces the rack reacts.

        Args:
            pinion (CylindricalGear): Driving pinion.
            power_kw (float): Transmitted power in kW.
            speed_rpm (float): Pinion rotational speed in rpm.

        Returns:
            str: Formatted force report.
        """
        f = self.force_report(pinion, power_kw, speed_rpm)
        header = "Rack forces"
        if self.name:
            header += f" '{self.name}'"
        return "\n".join([
            header,
            "=" * 40,
            f"transmitted power (P) = {power_kw:.3f} kW",
            f"pinion speed (n) = {speed_rpm:.1f} rpm",
            f"rack linear velocity (v) = {f['linear_velocity']:.3f} m/s",
            f"driving force (Ft) = {f['Ft']:.1f} N",
            f"separating force (Fr) = {f['Fr']:.1f} N",
        ])

    def __repr__(self):
        return (
            f"Rack(module={self.module}, length={self.length}, "
            f"material={self.material!r})"
        )
