"""Unit tests for the pure physics and economic engines.

Run from the project root::

    ./.venv/bin/python -m pytest tests/
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import numpy_financial as npf
import pytest

import economic_engine as econ
import physics_engine as phys
from models import SystemConfig, validate_config
from presets import PRESETS


# --------------------------------------------------------------------------- physics
def test_stefan_boltzmann_flux_known():
    cfg = SystemConfig(radiator_emissivity=0.9, T_sink_K=3.0)
    assert phys.radiator_flux(cfg, 318.0) == pytest.approx(521.8, abs=0.5)


def test_radiator_area_known_power():
    cfg = SystemConfig(radiator_emissivity=0.9, T_sink_K=3.0, T_surface_max_K=318.0, radiator_sides=2)
    area, T_used, ok = phys.required_radiator_area(1000.0, cfg)  # 1 MW
    assert ok and T_used == 318.0
    assert area == pytest.approx(958.2, abs=2.0)


def test_radiator_two_sided_halves_area():
    one = replace(SystemConfig(), radiator_sides=1)
    two = replace(SystemConfig(), radiator_sides=2)
    a1, _, _ = phys.required_radiator_area(500.0, one)
    a2, _, _ = phys.required_radiator_area(500.0, two)
    assert a2 == pytest.approx(a1 / 2, rel=1e-9)


def test_solar_area_known_power():
    cfg = SystemConfig(panel_efficiency=0.30, illumination_fraction=1.0)
    assert phys.required_solar_area(1000.0, cfg) == pytest.approx(2449.2, abs=2.0)


def test_solar_roundtrip():
    cfg = SystemConfig(panel_efficiency=0.31, illumination_fraction=0.97)
    area = phys.required_solar_area(742.0, cfg)
    assert phys.solar_generation(area, cfg) == pytest.approx(742.0, rel=1e-9)


def test_mass_monotonic_in_racks():
    small = phys.run_physics(replace(SystemConfig(), num_racks=10))
    large = phys.run_physics(replace(SystemConfig(), num_racks=40))
    assert large.total_dry_mass_kg > small.total_dry_mass_kg


def test_power_balance_includes_parasitic():
    cfg = SystemConfig(num_racks=10, power_per_rack_kW=100.0, parasitic_overhead_fraction=0.10)
    compute, load = phys.compute_power_balance(cfg)
    assert compute == pytest.approx(1000.0)
    assert load == pytest.approx(1100.0)


def test_survivability_flag_false_when_sink_ge_cap():
    cfg = replace(SystemConfig(), T_sink_K=320.0, T_surface_max_K=318.0)
    result = phys.run_physics(cfg)
    assert result.survivability_ok is False
    assert not np.isfinite(result.radiator_area_m2)


def test_length_m_ft_consistency():
    cfg = SystemConfig(radiator_width_m=5.0)
    length_m, length_ft = phys.radiator_length(100.0, cfg)
    assert length_m == pytest.approx(20.0)
    assert length_ft == pytest.approx(20.0 * 3.28084)


def test_degradation_decreases_output_and_noops_when_disabled():
    on = SystemConfig(enable_degradation=True, annual_degradation_pct=0.05)
    off = replace(on, enable_degradation=False)
    assert phys.apply_degradation(100.0, 5, on) < 100.0
    assert phys.apply_degradation(100.0, 5, off) == 100.0


# --------------------------------------------------------------------------- economics
def test_npv_matches_numpy_financial():
    flows = np.array([-1000.0, 300.0, 300.0, 300.0, 300.0, 300.0])
    assert econ.compute_npv(0.10, flows) == pytest.approx(137.236, abs=0.01)


def test_irr_none_for_all_negative():
    assert econ.compute_irr(np.array([-1000.0, -100.0, -100.0])) is None


def test_irr_known_series_matches_numpy_financial():
    flows = np.array([-1000.0, 600.0, 600.0])
    expected = npf.irr(flows)
    got = econ.compute_irr(flows)
    assert got is not None
    assert got == pytest.approx(float(expected), abs=1e-6)


def test_zero_crossing_none_when_never_recovers():
    series = np.linspace(-1000.0, -500.0, 60)  # always underwater
    assert econ.zero_crossing_period(series) is None


def test_zero_crossing_interpolation():
    series = np.array([-100.0, -50.0, 50.0])  # crosses zero halfway between idx 1 and 2
    assert econ.zero_crossing_period(series) == pytest.approx(1.5)


def test_capex_breakdown_sums_to_total():
    cfg = SystemConfig()
    physics = phys.run_physics(cfg)
    total, breakdown = econ.compute_capex(cfg, physics)
    assert sum(breakdown.values()) == pytest.approx(total)


def test_opex_insurance_is_annual_fraction():
    cfg = SystemConfig(insurance_pct_of_hardware=0.15)
    _, breakdown = econ.compute_opex(cfg, hardware_value=20_000_000.0)
    assert breakdown["Insurance"] == pytest.approx(20_000_000.0 * 0.15)


def test_bisection_finds_root_and_returns_none_without_sign_change():
    assert econ._bisect(lambda x: x - 5.0, 0.0, 10.0, tol=1e-6) == pytest.approx(5.0, abs=1e-3)
    assert econ._bisect(lambda x: x + 5.0, 0.0, 10.0) is None


# --------------------------------------------------------------------------- presets / integration
@pytest.mark.parametrize("name", list(PRESETS.keys()))
def test_preset_is_valid_config(name):
    errors, _ = validate_config(PRESETS[name])
    assert errors == []


@pytest.mark.parametrize("name", list(PRESETS.keys()))
def test_preset_is_thermally_survivable(name):
    assert phys.run_physics(PRESETS[name]).survivability_ok is True


@pytest.mark.parametrize("name", list(PRESETS.keys()))
def test_preset_full_pipeline(name):
    cfg = PRESETS[name]
    physics = phys.run_physics(cfg)
    result = econ.run_economics(cfg, physics)
    assert result.capex_total > 0
    assert len(result.cashflow_df) == cfg.lifespan_years + 1
    assert np.isfinite(result.npv_posttax)
    assert np.isfinite(result.npv_pretax)


# --------------------------------------------------------------------------- depreciation / tax / inflation
def test_depreciation_default_is_straight_line_over_life():
    cfg = replace(SystemConfig(), lifespan_years=10, overwrite_depreciation=False)
    dep = econ.depreciation_schedule(1000.0, cfg)
    assert len(dep) == 10
    assert np.allclose(dep, 100.0)


@pytest.mark.parametrize("scheme,kw", [
    ("SLN", {"depreciation_years_sln": 20}),
    ("DB", {"depreciation_rate_db": 0.10}),
    ("MACRS", {"macrs_years": 5}),
])
def test_depreciation_schemes_total_full_basis(scheme, kw):
    cfg = replace(SystemConfig(), lifespan_years=10, overwrite_depreciation=True,
                  depreciation_scheme=scheme, **kw)
    dep = econ.depreciation_schedule(1000.0, cfg)
    assert dep.sum() == pytest.approx(1000.0, abs=1e-3)  # terminal write-off recovers all basis
    assert np.all(dep >= 0)


def test_macrs_tables_sum_to_one():
    for years, table in econ.MACRS_TABLES.items():
        assert sum(table) == pytest.approx(1.0, abs=1e-3)


def test_declining_balance_front_loads_vs_straight_line():
    cfg_db = replace(SystemConfig(), lifespan_years=10, overwrite_depreciation=True,
                     depreciation_scheme="DB", depreciation_rate_db=0.30)
    cfg_sln = replace(SystemConfig(), lifespan_years=10, overwrite_depreciation=False)
    dep_db = econ.depreciation_schedule(1000.0, cfg_db)
    dep_sln = econ.depreciation_schedule(1000.0, cfg_sln)
    assert dep_db[0] > dep_sln[0]  # accelerated takes more in year 1


def test_tax_loss_carryforward_defers_tax():
    # Pin revenue to 1000/yr via per-FLOP mode; year-1 depreciation of 1200 forces a loss
    # (1000 - 100 - 1200 = -300) that a carry-forward can use against year-2's 900 gain.
    cfg = replace(
        SystemConfig(), lifespan_years=2, tax_rate=0.30, inflation_rate=0.0,
        overwrite_depreciation=False, enable_degradation=False,
        revenue_mode="per_flop", revenue_per_flop=1.0, flops_per_system=1000.0, utilization=1.0,
    )
    dep = np.array([1200.0, 0.0])
    with_cf = econ.build_financials(replace(cfg, enable_tax_loss_carryforward=True), 1000.0, 100.0, dep)
    without_cf = econ.build_financials(replace(cfg, enable_tax_loss_carryforward=False), 1000.0, 100.0, dep)
    assert without_cf["tax_paid"].sum() == pytest.approx(270.0)  # 0 then 0.30 * 900
    assert with_cf["tax_paid"].sum() == pytest.approx(180.0)  # 0 then 0.30 * (900 - 300)


def test_inflation_raises_nominal_cashflow():
    cfg0 = replace(SystemConfig(), inflation_rate=0.0)
    cfg5 = replace(SystemConfig(), inflation_rate=0.05)
    dep = np.zeros(cfg0.lifespan_years)
    df0 = econ.build_financials(cfg0, 1_000_000.0, 100_000.0, dep)
    df5 = econ.build_financials(cfg5, 1_000_000.0, 100_000.0, dep)
    # Final-year nominal revenue is higher under inflation.
    assert df5["nominal_revenue"].iloc[-1] > df0["nominal_revenue"].iloc[-1]


def test_real_rate_fisher_relation():
    assert econ.to_real_rate(0.10, 0.025) == pytest.approx((1.10 / 1.025) - 1.0)
    assert econ.to_real_rate(None, 0.025) is None
