"""Power-screw thread-form data.

Tabulated geometry factors for the common power-screw thread forms.
Units: ``thread_angle_deg`` is the thread HALF-angle measured in the
axial plane (the included flank angle is twice this); it feeds the
sec(alpha) friction correction in the torque equations.
``thread_height_factor`` is the engaged thread height h expressed as a
fraction of the pitch p (h = factor * p), used for the thread bearing,
bending and shear stresses.
"""

THREAD_FORMS = {
    "square": {"thread_angle_deg": 0.0, "thread_height_factor": 0.5},
    "acme": {"thread_angle_deg": 14.5, "thread_height_factor": 0.5},
    "trapezoidal": {"thread_angle_deg": 15.0, "thread_height_factor": 0.5},
    "buttress": {"thread_angle_deg": 7.0, "thread_height_factor": 0.663},
}


def get_thread_form(thread_form):
    """
    Look up geometry factors for a named power-screw thread form.

    Args:
        thread_form (str): Thread form, one of "square", "acme",
            "trapezoidal" or "buttress".

    Returns:
        dict: ``thread_angle_deg`` (half-angle) and
        ``thread_height_factor`` for the form.

    Raises:
        ValueError: If ``thread_form`` is not a known named form. Use
            ``thread_form="custom"`` with an explicit ``thread_angle``
            on :class:`~mecapy.shafts.power_screw.PowerScrew` for other
            angles; that case is handled by the class, not here.
    """
    if thread_form not in THREAD_FORMS:
        available = ", ".join(sorted(THREAD_FORMS))
        raise ValueError(
            f"Unknown thread form {thread_form!r}. Available: {available} "
            f"(or 'custom' with an explicit thread_angle)"
        )
    return THREAD_FORMS[thread_form]
