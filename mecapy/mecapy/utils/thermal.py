"""Energy and temperature-rise helpers for friction elements.

Standalone functions used by the clutch and brake modules to answer the
"how hot does it get?" questions: energy absorbed during a stop or a
clutch engagement, the resulting temperature rise of the friction
assembly, and the Newton-cooling decay back to ambient.

Units convention: this module is pure SI — energy in J, inertia in
kg*m^2, angular velocity in rad/s, torque in N*m, mass in kg, specific
heat in J/(kg*K), temperatures in degrees C, power in W. This differs
deliberately from the mm/N*mm/MPa convention of the geometry modules;
the clutch/brake classes document the one-line N*mm -> N*m bridge where
they meet.

Reference: Shigley's *Mechanical Engineering Design*, Ch. 16
(Secs. 16-8 and 16-9, energy considerations and temperature rise).
"""

import math


def stop_energy(inertia, omega_initial, omega_final=0.0):
    """
    Kinetic energy a brake must absorb to slow a rotating inertia.

    ::

        E = (1/2) * I * (w1^2 - w2^2)

    Args:
        inertia (float): Moment of inertia I of the braked system in kg*m^2.
        omega_initial (float): Initial angular velocity w1 in rad/s.
        omega_final (float): Final angular velocity w2 in rad/s (default 0,
            a complete stop). Must not exceed ``omega_initial``.

    Returns:
        float: Energy absorbed by the brake in J.

    Raises:
        ValueError: If ``inertia`` is not strictly positive, a velocity is
            negative, or ``omega_final`` exceeds ``omega_initial``.
    """
    if inertia <= 0:
        raise ValueError("Inertia must be strictly positive")
    if omega_initial < 0 or omega_final < 0:
        raise ValueError("Angular velocities must be non-negative")
    if omega_final > omega_initial:
        raise ValueError("omega_final must not exceed omega_initial")
    return 0.5 * inertia * (omega_initial ** 2 - omega_final ** 2)


def clutch_slip_energy(inertia_1, inertia_2, omega_1, omega_2):
    """
    Energy dissipated in the interface during a clutch engagement.

    Two inertias initially spinning at different speeds are brought to a
    common speed by the clutch; the slip energy is independent of the
    clutch torque (Shigley Sec. 16-8)::

        E = I1 * I2 * (w1 - w2)^2 / (2 * (I1 + I2))

    Args:
        inertia_1 (float): Driving-side inertia I1 in kg*m^2.
        inertia_2 (float): Driven-side inertia I2 in kg*m^2.
        omega_1 (float): Initial driving-side speed w1 in rad/s.
        omega_2 (float): Initial driven-side speed w2 in rad/s.

    Returns:
        float: Energy dissipated as heat in J.

    Raises:
        ValueError: If either inertia is not strictly positive.
    """
    if inertia_1 <= 0 or inertia_2 <= 0:
        raise ValueError("Inertias must be strictly positive")
    return (inertia_1 * inertia_2 * (omega_1 - omega_2) ** 2
            / (2 * (inertia_1 + inertia_2)))


def engagement_time(inertia_1, inertia_2, omega_1, omega_2, torque):
    """
    Time for a constant-torque clutch to bring two inertias to a common speed.

    ::

        t1 = I1 * I2 * (w1 - w2) / (T * (I1 + I2))

    Args:
        inertia_1 (float): Driving-side inertia I1 in kg*m^2.
        inertia_2 (float): Driven-side inertia I2 in kg*m^2.
        omega_1 (float): Initial driving-side speed w1 in rad/s.
        omega_2 (float): Initial driven-side speed w2 in rad/s.
        torque (float): Clutch torque T (assumed constant) in N*m.

    Returns:
        float: Slip duration in s (non-negative; uses ``abs(w1 - w2)``).

    Raises:
        ValueError: If an inertia or the torque is not strictly positive.
    """
    if inertia_1 <= 0 or inertia_2 <= 0:
        raise ValueError("Inertias must be strictly positive")
    if torque <= 0:
        raise ValueError("Torque must be strictly positive")
    return (inertia_1 * inertia_2 * abs(omega_1 - omega_2)
            / (torque * (inertia_1 + inertia_2)))


def temperature_rise(energy, mass, specific_heat=None, material=None):
    """
    Temperature rise of the assembly that soaks up the friction energy.

    Assumes the energy is absorbed uniformly by a mass with no loss to
    the surroundings during the (short) engagement (Shigley Eq. 16-56,
    SI form)::

        dT = E / (Cp * m)

    Args:
        energy (float): Friction energy absorbed in J.
        mass (float): Mass of the heat-absorbing parts in kg.
        specific_heat (float): Specific heat Cp in J/(kg*K). Give either
            this or ``material``, not both.
        material (str): Material name whose ``specific_heat`` is looked
            up in the material database (e.g. "steel", "cast_iron").

    Returns:
        float: Temperature rise in degrees C (equivalently K).

    Raises:
        ValueError: If ``mass`` is not strictly positive, ``energy`` is
            negative, or exactly one of ``specific_heat``/``material`` is
            not given.
    """
    if mass <= 0:
        raise ValueError("Mass must be strictly positive")
    if energy < 0:
        raise ValueError("Energy must be non-negative")
    if (specific_heat is None) == (material is None):
        raise ValueError("Give exactly one of specific_heat or material")
    if material is not None:
        # Imported here: mecapy.materials imports mecapy.utils, so a
        # module-level import back into materials would be circular.
        from ..materials import get_material_properties
        specific_heat = get_material_properties(material)["specific_heat"]
    if specific_heat <= 0:
        raise ValueError("Specific heat must be strictly positive")
    return energy / (specific_heat * mass)


def cooling_time_constant(mass, specific_heat, h_overall, area):
    """
    Newton-cooling time constant of a heated brake or clutch assembly.

    ::

        tau = m * Cp / (h * A)

    Args:
        mass (float): Mass of the assembly in kg.
        specific_heat (float): Specific heat Cp in J/(kg*K).
        h_overall (float): Overall heat-transfer coefficient (convection
            plus radiation) in W/(m^2*K).
        area (float): Heat-dissipating surface area in m^2.

    Returns:
        float: Time constant tau in s.

    Raises:
        ValueError: If any argument is not strictly positive.
    """
    if mass <= 0 or specific_heat <= 0 or h_overall <= 0 or area <= 0:
        raise ValueError("mass, specific_heat, h_overall and area must be "
                         "strictly positive")
    return mass * specific_heat / (h_overall * area)


def newton_cooling_temperature(time, temp_initial, temp_ambient, time_constant):
    """
    Assembly temperature during Newton cooling back to ambient.

    ::

        T(t) = T_amb + (T1 - T_amb) * exp(-t / tau)

    Args:
        time (float): Elapsed time t in s.
        temp_initial (float): Temperature T1 at t = 0 in degrees C.
        temp_ambient (float): Ambient temperature in degrees C.
        time_constant (float): Cooling time constant tau in s (see
            :func:`cooling_time_constant`).

    Returns:
        float: Temperature at time t in degrees C.

    Raises:
        ValueError: If ``time`` is negative or ``time_constant`` is not
            strictly positive.
    """
    if time < 0:
        raise ValueError("Time must be non-negative")
    if time_constant <= 0:
        raise ValueError("Time constant must be strictly positive")
    return temp_ambient + (temp_initial - temp_ambient) * math.exp(-time / time_constant)


def interface_power(torque, omega):
    """
    Instantaneous heat-generation rate at a slipping friction interface.

    ::

        H = T * w

    Args:
        torque (float): Friction torque in N*m.
        omega (float): Relative slip speed in rad/s.

    Returns:
        float: Power dissipated in W.
    """
    return torque * omega


def pv_value(pressure, velocity):
    """
    pV severity index of a friction lining.

    The product of contact pressure and rubbing velocity, used to gauge
    how hard a lining is being worked (wear and heat loading).

    Args:
        pressure (float): Contact pressure in Pa.
        velocity (float): Rubbing velocity in m/s.

    Returns:
        float: pV value in Pa*m/s (equivalently W/m^2).
    """
    return pressure * velocity
