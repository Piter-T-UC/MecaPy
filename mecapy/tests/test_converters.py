"""Tests for the unit conversion helpers."""

import pytest

from mecapy.utils import converters
from mecapy.utils.constants import G


class TestConverters:
    """Each converter and its inverse round-trip; a few known anchors."""

    @pytest.mark.parametrize("fwd, back, value", [
        (converters.mm_to_m, converters.m_to_mm, 1234.0),
        (converters.kpa_to_pa, converters.pa_to_kpa, 250.0),
        (converters.mpa_to_pa, converters.pa_to_mpa, 350.0),
        (converters.in_to_mm, converters.mm_to_in, 4.5),
        (converters.lbf_to_n, converters.n_to_lbf, 100.0),
        (converters.psi_to_mpa, converters.mpa_to_psi, 15000.0),
        (converters.hp_to_kw, converters.kw_to_hp, 5.0),
        (converters.kg_to_newtons, converters.newtons_to_kg, 12.0),
    ])
    def test_round_trip(self, fwd, back, value):
        """Applying a converter then its inverse returns the input."""
        assert back(fwd(value)) == pytest.approx(value)

    def test_known_anchors(self):
        """A handful of exact/known conversion values."""
        assert converters.mm_to_m(1000) == pytest.approx(1.0)
        assert converters.in_to_mm(1) == pytest.approx(25.4)
        assert converters.mpa_to_pa(1) == pytest.approx(1e6)
        assert converters.kg_to_newtons(1) == pytest.approx(G)
        assert converters.hp_to_kw(1) == pytest.approx(0.7457, abs=1e-3)
