Gears
=====

Gear types, transmissions and AGMA fatigue rating. Typical workflow:
build the gear geometry (:class:`~mecapy.gears.SpurGear`,
:class:`~mecapy.gears.HelicalGear`, ...), assemble a
:class:`~mecapy.gears.Transmission` to check compatibility and compute
ratios, then evaluate tooth bending and surface fatigue with
:class:`~mecapy.gears.AGMARating`::

    from mecapy.gears import SpurGear, Transmission

    pinion = SpurGear(17, module=2.5, face_width=40.0)
    gear = SpurGear(52, module=2.5, face_width=40.0)
    train = Transmission().add_stage(pinion, gear)
    rating = pinion.rate_agma(gear, power_kw=3.0,
                              pinion_speed_rpm=1800, hardness_HB=240)
    print(rating.summary())

Internal (ring) gears
---------------------

Any cylindrical gear can be an internal (ring) gear by passing
``internal=True``. Its teeth point inwards, so the tip circle lies
inside the pitch circle and the root circle outside it, and it meshes
only with an external cylindrical gear of fewer teeth. The mesh
quantities that are sums for an external pair become differences: the
center distance is ``m (z_ring - z_pinion) / 2``, and the pinion runs
*inside* the ring, which also means the mesh does **not** reverse the
direction of rotation. Internal helical gears mesh with the *same*
hand, not opposite hands::

    from mecapy.gears import SpurGear, Transmission

    pinion = SpurGear(20, module=2.0, speed_rpm=1800, power_kw=5.0)
    ring = SpurGear(80, module=2.0, internal=True)
    pinion.center_distance_with(ring)          # 60.0 mm
    train = Transmission().add_stage(pinion, ring)
    train.train_value                          # +0.25: same direction
    train.plot()                               # scaled drawing

Undercut does not apply to an internal gear; the equivalent failure
mode is trimming interference, checked with
:meth:`~mecapy.gears.CylindricalGear.has_trimming_interference_with`.
AGMA ratings of internal meshes use internal contact geometry but are
approximate — see :mod:`mecapy.gears.agma`.

Visualizing a train
-------------------

:meth:`~mecapy.gears.Transmission.plot` draws a parallel-axis train to
scale: every gear at its real size and center distance, ring gears as a
shaded annulus, and a curved arrow per gear for the direction of
rotation (from
:meth:`~mecapy.gears.Transmission.rotation_senses`). It needs
matplotlib, an optional dependency (``pip install -e ".[viz]"``). The
drawing geometry itself is available without matplotlib from
:meth:`~mecapy.gears.Transmission.stage_layout`. See
``examples/internal_gear_train.py``.

Base class
----------

.. automodule:: mecapy.gears.gear
   :members:
   :undoc-members:
   :show-inheritance:

Cylindrical gears (spur, helical, herringbone)
----------------------------------------------

.. automodule:: mecapy.gears.cylindrical
   :members:
   :undoc-members:
   :show-inheritance:

Rack and pinion
---------------

.. automodule:: mecapy.gears.rack
   :members:
   :undoc-members:
   :show-inheritance:

Bevel gears
-----------

.. automodule:: mecapy.gears.bevel
   :members:
   :undoc-members:
   :show-inheritance:

Worm drives
-----------

.. automodule:: mecapy.gears.worm
   :members:
   :undoc-members:
   :show-inheritance:

Planetary gear sets
-------------------

.. automodule:: mecapy.gears.planetary
   :members:
   :undoc-members:
   :show-inheritance:

Transmissions
-------------

.. automodule:: mecapy.gears.transmission
   :members:
   :undoc-members:
   :show-inheritance:

AGMA rating
-----------

.. automodule:: mecapy.gears.agma
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mecapy.gears.agma_data
   :members:
   :undoc-members:
   :show-inheritance:
