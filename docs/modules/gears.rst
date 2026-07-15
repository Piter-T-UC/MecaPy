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
