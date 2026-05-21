"""Techno-economics for an orbital data center: CapEx, OpEx, revenue, cash flow and
the headline financial metrics, plus an equal-power terrestrial comparison.

Pure module: imports only ``constants``, ``models``, NumPy, Pandas and numpy-financial.
Never imports Streamlit. Operates on a ``SystemConfig`` plus the ``PhysicsResult`` that
fixes the mass and areas the costs scale from.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy_financial as npf
import pandas as pd

from constants import HOURS_PER_YEAR, KW_PER_MW, MONTHS_PER_YEAR
from models import EconomicsResult, PhysicsResult, SystemConfig


def compute_capex(cfg: SystemConfig, physics: PhysicsResult) -> tuple[float, dict[str, float]]:
    """Total up-front CapEx and its component breakdown.

    Launch scales with dry mass; hardware carries a space-qualification premium; solar and
    radiator scale with their areas; R&D + bus integration is a percentage of the built
    subsystems (everything except launch).
    """
    launch = physics.total_dry_mass_kg * cfg.launch_cost_per_kg
    hardware = cfg.num_racks * cfg.rack_unit_cost * cfg.space_qualification_premium
    solar = physics.solar_array_area_m2 * cfg.solar_cost_per_m2
    radiator = physics.radiator_area_m2 * cfg.radiator_cost_per_m2
    rd_integration = (hardware + solar + radiator) * cfg.rd_integration_overhead_pct

    breakdown = {
        "Launch": launch,
        "Hardware": hardware,
        "Solar": solar,
        "Radiator": radiator,
        "R&D + Integration": rd_integration,
    }
    return sum(breakdown.values()), breakdown


def compute_opex(cfg: SystemConfig, hardware_value: float) -> tuple[float, dict[str, float]]:
    """Annual OpEx and breakdown. Insurance is an annual premium on hardware value."""
    breakdown = {
        "Ground Station": cfg.ground_station_cost_annual,
        "Station-keeping": cfg.stationkeeping_cost_annual,
        "Insurance": hardware_value * cfg.insurance_pct_of_hardware,
        "Ops / Management": cfg.terrestrial_ops_cost_annual,
    }
    return sum(breakdown.values()), breakdown


def _degradation_factor(cfg: SystemConfig, year: int) -> float:
    if not cfg.enable_degradation:
        return 1.0
    return max((1.0 - cfg.annual_degradation_pct) ** year, 0.0)


def compute_revenue(cfg: SystemConfig, year: int = 0) -> float:
    """Annual revenue for a given operational ``year`` (0 = first year, no degradation yet)."""
    utilization = min(max(cfg.utilization, 0.0), 1.0)
    if cfg.revenue_mode == "per_gpu_hour":
        num_gpus = cfg.num_racks * cfg.gpus_per_rack
        base = cfg.revenue_per_gpu_hour * num_gpus * utilization * HOURS_PER_YEAR
    elif cfg.revenue_mode == "per_flop":
        base = cfg.revenue_per_flop * cfg.flops_per_system * utilization
    else:
        raise ValueError(f"Unknown revenue_mode: {cfg.revenue_mode!r}")
    return base * _degradation_factor(cfg, year)


def build_cashflows(cfg: SystemConfig, capex_total: float, opex_annual: float) -> pd.DataFrame:
    """Annual cash-flow table from year 0 (CapEx outflow) through the lifespan."""
    rows = [{"year": 0, "revenue": 0.0, "opex": 0.0, "net": -capex_total}]
    for year in range(1, cfg.lifespan_years + 1):
        revenue = compute_revenue(cfg, year - 1)
        rows.append({"year": year, "revenue": revenue, "opex": opex_annual, "net": revenue - opex_annual})
    df = pd.DataFrame(rows)
    df["cumulative"] = df["net"].cumsum()
    return df


def monthly_cashflow_series(cfg: SystemConfig, capex_total: float, opex_annual: float) -> np.ndarray:
    """Cumulative cash flow at monthly resolution, for an exact payback period."""
    cum = [-capex_total]
    for month in range(1, cfg.lifespan_years * MONTHS_PER_YEAR + 1):
        operational_year = (month - 1) // MONTHS_PER_YEAR
        monthly_net = (compute_revenue(cfg, operational_year) - opex_annual) / MONTHS_PER_YEAR
        cum.append(cum[-1] + monthly_net)
    return np.array(cum)


def compute_npv(rate: float, flows: np.ndarray) -> float:
    """Net present value; ``flows[0]`` is the year-0 (undiscounted) outflow."""
    return float(npf.npv(rate, flows))


def compute_irr(flows: np.ndarray) -> float | None:
    """Internal rate of return, or ``None`` when no real root exists (e.g. all-negative flows)."""
    try:
        irr = npf.irr(np.asarray(flows, dtype=float))
    except (ValueError, FloatingPointError):
        return None
    if irr is None or np.isnan(irr):
        return None
    return float(irr)


def compute_payback_months(monthly_cum: np.ndarray) -> float | None:
    """First month the cumulative cash flow turns non-negative, linearly interpolated."""
    if monthly_cum[0] >= 0:
        return 0.0
    for m in range(1, len(monthly_cum)):
        if monthly_cum[m] >= 0:
            prev, curr = monthly_cum[m - 1], monthly_cum[m]
            fraction = -prev / (curr - prev) if curr != prev else 0.0
            return (m - 1) + fraction
    return None


def terrestrial_baseline(cfg: SystemConfig, total_load_kW: float) -> tuple[float, float]:
    """Annual and full-lifespan cost of an equal-power ground data center.

    Energy cost applies PUE to the IT load. Land, water and ops are annual; terrestrial
    build cost scales with MW and is amortised across the lifespan.
    """
    mw = total_load_kW / KW_PER_MW
    annual_energy = total_load_kW * cfg.pue * HOURS_PER_YEAR * cfg.grid_cost_per_kWh
    annual_fixed = (
        cfg.terrestrial_land_cost_annual
        + cfg.terrestrial_water_cost_annual
        + cfg.terrestrial_ground_ops_annual
    )
    terrestrial_capex = cfg.terrestrial_capex_per_MW * mw
    annual_total = annual_energy + annual_fixed + terrestrial_capex / cfg.lifespan_years
    lifespan_total = annual_total * cfg.lifespan_years
    return annual_total, lifespan_total


def _bisect(f: Callable[[float], float], lo: float, hi: float, tol: float = 1.0, max_iter: int = 200) -> float | None:
    """Find a root of monotonic ``f`` in ``[lo, hi]``; return ``None`` if it doesn't cross zero."""
    f_lo, f_hi = f(lo), f(hi)
    if f_lo == 0:
        return lo
    if f_hi == 0:
        return hi
    if (f_lo > 0) == (f_hi > 0):
        return None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if (f_mid > 0) == (f_lo > 0):
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    return 0.5 * (lo + hi)


def _orbital_tco(cfg: SystemConfig, physics: PhysicsResult, capex_total: float, opex_annual: float) -> float:
    """Total cost of ownership over the lifespan: CapEx plus all OpEx (undiscounted)."""
    return capex_total + opex_annual * cfg.lifespan_years


def crossover_launch_cost(
    cfg: SystemConfig, physics: PhysicsResult, capex_ex_launch: float, opex_annual: float,
    terrestrial_tco: float, lo: float = 0.0, hi: float = 50_000.0,
) -> float | None:
    """Launch $/kg at which orbital TCO equals the terrestrial baseline (bisection)."""
    def f(launch: float) -> float:
        orbital = (capex_ex_launch + launch * physics.total_dry_mass_kg) + opex_annual * cfg.lifespan_years
        return orbital - terrestrial_tco

    return _bisect(f, lo, hi)


def crossover_power_cost(
    cfg: SystemConfig, physics: PhysicsResult, orbital_tco: float,
    lo: float = 0.0, hi: float = 5.0,
) -> float | None:
    """Grid $/kWh at which the terrestrial baseline rises to match orbital TCO (bisection)."""
    def f(grid: float) -> float:
        trial = SystemConfig(**{**cfg.__dict__, "grid_cost_per_kWh": grid})
        _, terrestrial = terrestrial_baseline(trial, physics.total_electrical_load_kW)
        return terrestrial - orbital_tco

    return _bisect(f, lo, hi, tol=1e-4)


def run_economics(cfg: SystemConfig, physics: PhysicsResult) -> EconomicsResult:
    """Orchestrator: full economic result for one config + physics pair."""
    capex_total, capex_breakdown = compute_capex(cfg, physics)
    hardware_value = capex_breakdown["Hardware"]
    opex_annual, opex_breakdown = compute_opex(cfg, hardware_value)
    revenue_annual = compute_revenue(cfg, 0)

    cashflow_df = build_cashflows(cfg, capex_total, opex_annual)
    flows = cashflow_df["net"].to_numpy()
    npv = compute_npv(cfg.discount_rate, flows)
    irr = compute_irr(flows)

    monthly = monthly_cashflow_series(cfg, capex_total, opex_annual)
    payback_months = compute_payback_months(monthly)

    _, terrestrial_tco = terrestrial_baseline(cfg, physics.total_electrical_load_kW)
    orbital_tco = _orbital_tco(cfg, physics, capex_total, opex_annual)

    capex_ex_launch = capex_total - capex_breakdown["Launch"]
    crossover_launch = crossover_launch_cost(
        cfg, physics, capex_ex_launch, opex_annual, terrestrial_tco
    )
    crossover_power = crossover_power_cost(cfg, physics, orbital_tco)

    return EconomicsResult(
        capex_total=capex_total,
        capex_breakdown=capex_breakdown,
        opex_annual=opex_annual,
        opex_breakdown=opex_breakdown,
        revenue_annual=revenue_annual,
        cashflow_df=cashflow_df,
        cumulative_cashflow=cashflow_df["cumulative"].to_numpy(),
        npv=npv,
        irr=irr,
        payback_months=payback_months,
        terrestrial_tco=terrestrial_tco,
        orbital_tco=orbital_tco,
        crossover_launch_cost=crossover_launch,
        crossover_power_cost=crossover_power,
    )
