"""Tests for the energy / temperature-rise helpers (mecapy.utils.thermal)."""

import math

import pytest

from mecapy.utils import thermal


class TestStopEnergy:
    def test_full_stop(self):
        # E = 0.5 * 2 * 100^2
        assert thermal.stop_energy(2.0, 100.0) == pytest.approx(10000.0)

    def test_partial_stop(self):
        assert thermal.stop_energy(2.0, 100.0, 60.0) == pytest.approx(
            0.5 * 2.0 * (100.0 ** 2 - 60.0 ** 2))

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            thermal.stop_energy(0.0, 100.0)
        with pytest.raises(ValueError):
            thermal.stop_energy(2.0, -1.0)
        with pytest.raises(ValueError):
            thermal.stop_energy(2.0, 50.0, 60.0)


class TestClutchSlipEnergy:
    def test_equal_inertias_stop(self):
        # Classic result: I1 = I2, w2 = 0 -> E = I*w^2/4,
        # half the energy of rigidly stopping I1 alone.
        assert thermal.clutch_slip_energy(2.0, 2.0, 100.0, 0.0) == pytest.approx(
            2.0 * 100.0 ** 2 / 4)

    def test_no_slip_no_energy(self):
        assert thermal.clutch_slip_energy(1.0, 3.0, 50.0, 50.0) == 0.0

    def test_symmetric_in_speeds(self):
        assert thermal.clutch_slip_energy(1.0, 3.0, 80.0, 20.0) == pytest.approx(
            thermal.clutch_slip_energy(1.0, 3.0, 20.0, 80.0))

    def test_invalid_inertia(self):
        with pytest.raises(ValueError):
            thermal.clutch_slip_energy(0.0, 2.0, 100.0, 0.0)


class TestEngagementTime:
    def test_basic(self):
        # t = I1*I2*(w1-w2) / (T*(I1+I2))
        assert thermal.engagement_time(2.0, 2.0, 100.0, 0.0, 50.0) == pytest.approx(
            2.0 * 2.0 * 100.0 / (50.0 * 4.0))

    def test_invalid_torque(self):
        with pytest.raises(ValueError):
            thermal.engagement_time(2.0, 2.0, 100.0, 0.0, 0.0)


class TestTemperatureRise:
    def test_explicit_specific_heat(self):
        # 25 kJ into 4 kg at Cp = 500 -> 12.5 C
        assert thermal.temperature_rise(25e3, 4.0, specific_heat=500) == pytest.approx(12.5)

    def test_material_lookup(self):
        assert thermal.temperature_rise(25e3, 4.0, material="steel") == pytest.approx(12.5)

    def test_exactly_one_source_of_cp(self):
        with pytest.raises(ValueError):
            thermal.temperature_rise(25e3, 4.0)
        with pytest.raises(ValueError):
            thermal.temperature_rise(25e3, 4.0, specific_heat=500, material="steel")

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            thermal.temperature_rise(25e3, 0.0, specific_heat=500)
        with pytest.raises(ValueError):
            thermal.temperature_rise(-1.0, 4.0, specific_heat=500)


class TestNewtonCooling:
    def test_time_constant(self):
        assert thermal.cooling_time_constant(10.0, 500.0, 25.0, 0.4) == pytest.approx(
            10.0 * 500.0 / (25.0 * 0.4))

    def test_one_time_constant_decay(self):
        # After one time constant the excess over ambient drops to 1/e.
        temp = thermal.newton_cooling_temperature(300.0, 120.0, 20.0, 300.0)
        assert temp - 20.0 == pytest.approx((120.0 - 20.0) / math.e)

    def test_initial_and_final_temperatures(self):
        assert thermal.newton_cooling_temperature(0.0, 120.0, 20.0, 300.0) == pytest.approx(120.0)
        assert thermal.newton_cooling_temperature(1e9, 120.0, 20.0, 300.0) == pytest.approx(20.0)

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            thermal.cooling_time_constant(0.0, 500.0, 25.0, 0.4)
        with pytest.raises(ValueError):
            thermal.newton_cooling_temperature(-1.0, 120.0, 20.0, 300.0)
        with pytest.raises(ValueError):
            thermal.newton_cooling_temperature(10.0, 120.0, 20.0, 0.0)


class TestPowerAndPV:
    def test_interface_power(self):
        assert thermal.interface_power(200.0, 150.0) == pytest.approx(30000.0)

    def test_pv_value(self):
        assert thermal.pv_value(1e6, 20.0) == pytest.approx(2e7)
