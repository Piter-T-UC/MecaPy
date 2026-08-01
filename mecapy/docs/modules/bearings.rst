Bearings
========

Four bearing families, each with its own governing model:
:class:`~mecapy.bearings.Bearing` for rolling contact,
:class:`~mecapy.bearings.JournalBearing` for a hydrodynamic plain journal,
:class:`~mecapy.bearings.PlainBearing` for a boundary-lubricated bushing and
:class:`~mecapy.bearings.ThrustBearing` for a tapered-land thrust pad::

    from mecapy.bearings import Bearing, JournalBearing

    # Rolling contact: catalog ratings in, life out
    bearing = Bearing(bore_diameter=25.0, outer_diameter=52.0, width=15.0,
                      C10=35000.0, C0=14000.0, name="6205")
    load = bearing.equivalent_load(radial_load=3000.0, axial_load=1400.0)
    bearing.life_hours(load, speed=1500.0)            # Shigley ch. 11
    bearing.iso_life_hours(load, 1500.0,              # ISO 281
                           sae_grade=30, temperature=70.0)
    bearing.static_safety_factor(3000.0, 1400.0)      # ISO 76

    # Hydrodynamic journal: solve the operating point, then check it
    journal = JournalBearing(radius=25.0, clearance=0.025, length=50.0,
                             speed=25.0, load=2500.0,
                             sae_grade=40, temperature=60.0)
    journal.solve_film_temperature(inlet_temperature=60.0)
    print(journal.describe())

**Units.** Geometry is in mm, forces in N, pressures in MPa, torques in N*mm and
power in W.  Rolling-bearing speeds are in **rpm** (the catalog convention) while
journal, bushing and thrust speeds are in **rev/s** (the Sommerfeld convention) —
the class docstrings state which, and a ``pint.Quantity`` may be passed for any
dimensional input to avoid the question entirely (see *Optional units* below).

Rolling-contact life
--------------------

:class:`~mecapy.bearings.Bearing` carries **two** life models on purpose, because
they answer the same question with different conventions and do not agree
numerically:

* **Shigley ch. 11** — :meth:`~mecapy.bearings.Bearing.life`,
  :meth:`~mecapy.bearings.Bearing.adjusted_life` and
  :meth:`~mecapy.bearings.Bearing.required_C10` use the 3-parameter Weibull
  distribution (x0 = 0.02, theta = 4.459, b = 1.483) for reliabilities other
  than 90%.
* **ISO 281:2007** — :meth:`~mecapy.bearings.Bearing.iso_life` applies the
  modified rating life ``L_nm = a1 * a_ISO * L10``, where ``a_ISO`` folds in the
  cleanliness of the installation (``e_C``), the fatigue load limit ``Cu`` and
  the film quality ``kappa = nu/nu1``.

Cleanliness dominates: at a fixed load the same bearing can differ by more than
an order of magnitude in life between ``high_cleanliness`` and
``severe_contamination``.  Combined loading (X/Y from Table 11-1) applies to the
ball family only; roller families under thrust are rejected rather than
silently approximated, and tapered rollers have their own induced-thrust
helpers (:meth:`~mecapy.bearings.Bearing.induced_thrust` and
:meth:`~mecapy.bearings.Bearing.tapered_pair_loads`).

Static rating and speed limits
------------------------------

:meth:`~mecapy.bearings.Bearing.static_safety_factor` is the ISO 76 check
``s0 = C0/P0`` with ``P0 = max(X0 Fr + Y0 Fa, Fr)`` — the floor matters, since a
lightly thrust-loaded deep-groove bearing is governed by its radial load alone.
:meth:`~mecapy.bearings.Bearing.speed_check` compares the ``n*dm`` product
against representative catalog limits; those are marked ``ESTIMATED`` in
:mod:`mecapy.bearings.iso281_data` and are a feasibility screen, not a
substitute for the bearing's own catalog page.

Hydrodynamic journal bearings
-----------------------------

:class:`~mecapy.bearings.JournalBearing` implements the Shigley ch. 12 route:
Petroff friction, the Sommerfeld number, the Raimondi-Boyd charts (digitized in
:mod:`mecapy.bearings.lubrication_data`, blended across l/d by Eq. 12-16), the
lubricant temperature rise and the Trumpler criteria.

Viscosity and temperature are coupled, so the operating point has to be *solved*
rather than assumed.  :meth:`~mecapy.bearings.JournalBearing.solve_film_temperature`
runs the damped fixed point ``T_avg = T_in + dT/2`` with ``mu(SAE, T)``
re-evaluated each pass and, by default, leaves the bearing at its own converged
state::

    solved = journal.solve_film_temperature(inlet_temperature=60.0)
    solved["temperature"], solved["viscosity"], solved["converged"]

Sizing and stability
--------------------

The chart inverse :func:`~mecapy.bearings.sommerfeld_for` turns a required film
ratio back into a Sommerfeld number, which is what the design helpers are built
on: :meth:`~mecapy.bearings.JournalBearing.viscosity_for_minimum_film` and
:meth:`~mecapy.bearings.JournalBearing.length_for_minimum_film` each size one
unknown for a target ``h0``.

Clearance is different, because ``h0(c)`` **peaks** at an intermediate clearance
— tightening it raises ``h0/c`` but shrinks ``c``.  So the useful answers are
:meth:`~mecapy.bearings.JournalBearing.optimum_clearance` and
:meth:`~mecapy.bearings.JournalBearing.clearance_window_for_minimum_film`, which
returns the interval a design can actually be manufactured to.

A lightly loaded journal rides near the centre of its clearance circle, where
the film's cross-coupled stiffness can drive a self-excited orbit at roughly half
shaft speed.  :attr:`~mecapy.bearings.JournalBearing.is_whirl_prone` flags an
eccentricity ratio below 0.6.  This is a rotordynamics rule of thumb, **not** a
Shigley result, and a real stability assessment needs the film's stiffness and
damping coefficients together with the rotor mass.

Boundary-lubricated bushings
----------------------------

When ``journal.is_thick_film()`` is False there is no hydrodynamic film to
compute, and the bearing belongs to :class:`~mecapy.bearings.PlainBearing`
instead: it is rated on the pressure ``P = W/(d l)``, the rubbing velocity
``V = pi d N`` and their product ``PV``, against the liner limits of Shigley
Table 12-8 (:mod:`mecapy.bearings.bushing_data`).  PV is usually what binds — a
bushing can be inside both its pressure and its velocity rating and still be
over its PV rating, which is what
:meth:`~mecapy.bearings.PlainBearing.pv_check` is for.

Hydrodynamic thrust bearings
----------------------------

:class:`~mecapy.bearings.ThrustBearing` models a ring of tapered-land pads as
plane sliders, whose Reynolds solution is exact in closed form — nothing here is
digitized.  Load capacity peaks at a taper ratio of about 2.19
(:func:`~mecapy.bearings.load_coefficient`), which is why tapered lands are cut
to roughly that ratio.  Side leakage is neglected, so capacity is optimistic for
short, wide pads.  One consequence of the closed form worth knowing: the
adiabatic temperature rise depends on pad pressure alone — thickening the oil
raises the friction and the flow by the same ``sqrt(mu)``.

Visualizing
-----------

Plots need the optional viz extra (``pip install -e ".[viz]"``); matplotlib is
imported lazily inside each method, and each returns the ``Figure``:

* :meth:`~mecapy.bearings.JournalBearing.plot_film` — the journal in its bore
  with the clearance exaggerated and the minimum film marked.
* :meth:`~mecapy.bearings.JournalBearing.plot_clearance_design` — the clearance
  trade-off, minimum film and power loss against clearance.
* :meth:`~mecapy.bearings.ThrustBearing.plot_pressure` — film pressure and
  thickness along one pad.

Each has a matplotlib-free counterpart returning the same data
(:meth:`~mecapy.bearings.JournalBearing.film_profile`,
:meth:`~mecapy.bearings.JournalBearing.clearance_sweep`,
:meth:`~mecapy.bearings.ThrustBearing.pressure_profile`), so the geometry can be
inspected and tested without a plotting backend.

Optional units
--------------

Every dimensional input accepts a ``pint.Quantity`` (``pip install -e ".[units]"``),
converted to the documented unit at the boundary; plain floats are assumed to be
in that unit already and behave exactly as before::

    from mecapy.utils.units import ureg

    Bearing(bore_diameter=1 * ureg.inch, outer_diameter=52.0, width=15.0,
            C10=35 * ureg.kN)
    JournalBearing(radius=1 * ureg.inch, clearance=0.025, length=50.0,
                   speed=1500 * ureg.rpm, load=2500 * ureg.N, viscosity=20.0)

Worked examples
---------------

``examples/bearing_design.py`` runs the ch. 11 rating-life procedure, the ISO 281
comparison and the full ch. 12 design pass (thermal solve, Trumpler criteria,
clearance window) and renders the textbook figures.
``examples/plain_and_thrust_bearings.py`` covers the PV design pass and the
tapered-land thrust pad.

Reference
---------

.. automodule:: mecapy.bearings.bearing
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.bearing_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.iso281_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.journal
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.lubrication_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.plain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.bushing_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.bearings.thrust
   :members:
   :undoc-members:
   :show-inheritance:
