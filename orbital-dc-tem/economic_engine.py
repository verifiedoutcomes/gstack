"""Techno-economics for an orbital data center: CapEx, OpEx, revenue, depreciation, tax,
inflation-adjusted cash flow and the headline financial metrics, plus an equal-power
terrestrial comparison.

Pure module: imports only ``constants``, ``models``, NumPy, Pandas and numpy-financial.
Never imports Streamlit. Operates on a ``SystemConfig`` plus the ``PhysicsResult`` that
fixes the mass and areas the costs scale from.

Conventions
-----------
- Revenue and operating costs are entered in **base-year (real) dollars**; they are
  inflated to nominal dollars over time at ``inflation_rate``.
- ``discount_rate`` is the **nominal** required return. The real rate is derived via the
  Fisher relation. NPV is identical in real and nominal terms under consistent
  discounting; inflation's economic bite shows up through **taxes** (depreciation is a
  fixed nominal deduction) and through the cash-flow trajectory and IRR.
- Depreciation is non-cash: it lowers taxable income but not cash flow directly. The
  whole depreciable basis is written off over the life (terminal write-off of any
  remaining book value at deorbit, no salvage), so the depreciation *scheme* changes the
  timing -- and thus the NPV of the tax shield -- not the total.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy_financial as npf
import pandas as pd

from constants import HOURS_PER_YEAR, KW_PER_MW
from models import EconomicsResult, PhysicsResult, SystemConfig

# MACRS GDS percentage tables (half-year convention), as fractions of basis. Keys are the
# property recovery class in years; each list runs one year longer than the class.
MACRS_TABLES: dict[int, list[float]] = {
    3: [0.3333, 0.4445, 0.1481, 0.0741],
    5: [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576],
    7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446],
    10: [0.1000, 0.1800, 0.1440, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328],
    15: [0.0500, 0.0950, 0.0855, 0.0770, 0.0693, 0.0623, 0.0590, 0.0590, 0.0591, 0.0590,
         0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0295],
    20: [0.0375, 0.07219, 0.06677, 0.06177, 0.05713, 0.05285, 0.04888, 0.04522, 0.04462,
         0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461, 0.04462, 0.04461,
         0.04462, 0.04461, 0.02231],
}


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
    """Annual OpEx (base-year dollars) and breakdown. Insurance is an annual premium."""
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
    """Annual revenue in base-year dollars for operational ``year`` (0 = first year)."""
    utilization = min(max(cfg.utilization, 0.0), 1.0)
    if cfg.revenue_mode == "per_gpu_hour":
        num_gpus = cfg.num_racks * cfg.gpus_per_rack
        base = cfg.revenue_per_gpu_hour * num_gpus * utilization * HOURS_PER_YEAR
    elif cfg.revenue_mode == "per_flop":
        base = cfg.revenue_per_flop * cfg.flops_per_system * utilization
    else:
        raise ValueError(f"Unknown revenue_mode: {cfg.revenue_mode!r}")
    return base * _degradation_factor(cfg, year)


def depreciation_schedule(basis: float, cfg: SystemConfig) -> np.ndarray:
    """Per-operating-year depreciation of ``basis`` over ``lifespan_years``.

    Without ``overwrite_depreciation`` the asset is straight-lined over the project life.
    Otherwise the chosen scheme applies: SLN over ``depreciation_years_sln``, declining
    balance at ``depreciation_rate_db``, or a MACRS class table. Any basis not yet
    recovered is written off in the final year (deorbit, no salvage), so every scheme
    totals the full basis -- only the timing differs.
    """
    n = cfg.lifespan_years
    dep = np.zeros(n)
    if n <= 0:
        return dep

    if not cfg.overwrite_depreciation:
        dep[:] = basis / n
        return dep

    scheme = cfg.depreciation_scheme.upper()
    if scheme == "SLN":
        years = max(int(cfg.depreciation_years_sln), 1)
        annual = basis / years
        tentative = [annual] * n
    elif scheme == "DB":
        rate = cfg.depreciation_rate_db
        tentative = []
        book = basis
        for _ in range(n):
            d = book * rate
            tentative.append(d)
            book -= d
    elif scheme == "MACRS":
        table = MACRS_TABLES.get(int(cfg.macrs_years))
        if table is None:
            raise ValueError(f"Unsupported MACRS recovery class: {cfg.macrs_years}")
        tentative = [basis * table[t] if t < len(table) else 0.0 for t in range(n)]
    else:
        raise ValueError(f"Unknown depreciation_scheme: {cfg.depreciation_scheme!r}")

    cumulative = 0.0
    for t in range(n):
        d = min(max(tentative[t], 0.0), basis - cumulative)
        dep[t] = d
        cumulative += d
    remaining = basis - cumulative
    if remaining > 1e-6:
        dep[-1] += remaining
    return dep


def build_financials(
    cfg: SystemConfig, capex_total: float, opex_annual: float, depreciation: np.ndarray
) -> pd.DataFrame:
    """Year-by-year cash-flow table with inflation and tax.

    Columns include nominal revenue/opex, depreciation, taxable income, tax paid and the
    loss-carryforward balance, plus pre-tax and post-tax cash flows in both nominal and
    real (base-year) dollars and their cumulative sums. Row 0 is the CapEx outflow.
    """
    n = cfg.lifespan_years
    infl = cfg.inflation_rate
    tax_rate = cfg.tax_rate

    rows = [{
        "year": 0, "nominal_revenue": 0.0, "nominal_opex": 0.0, "depreciation": 0.0,
        "taxable_income": 0.0, "tax_paid": 0.0, "loss_carryforward": 0.0,
        "pretax_nominal": -capex_total, "posttax_nominal": -capex_total,
    }]

    carryforward = 0.0
    for t in range(1, n + 1):
        infl_factor = (1.0 + infl) ** t
        nominal_revenue = compute_revenue(cfg, t - 1) * infl_factor
        nominal_opex = opex_annual * infl_factor
        dep = float(depreciation[t - 1]) if t - 1 < len(depreciation) else 0.0
        taxable = nominal_revenue - nominal_opex - dep

        if cfg.enable_tax_loss_carryforward:
            if taxable < 0:
                carryforward += -taxable
                taxable_after = 0.0
            else:
                offset = min(carryforward, taxable)
                carryforward -= offset
                taxable_after = taxable - offset
        else:
            taxable_after = max(taxable, 0.0)

        tax = tax_rate * max(taxable_after, 0.0)
        pretax_nominal = nominal_revenue - nominal_opex  # depreciation is non-cash
        posttax_nominal = pretax_nominal - tax

        rows.append({
            "year": t, "nominal_revenue": nominal_revenue, "nominal_opex": nominal_opex,
            "depreciation": dep, "taxable_income": taxable, "tax_paid": tax,
            "loss_carryforward": carryforward,
            "pretax_nominal": pretax_nominal, "posttax_nominal": posttax_nominal,
        })

    df = pd.DataFrame(rows)
    factors = np.power(1.0 + infl, df["year"].to_numpy())
    df["pretax_real"] = df["pretax_nominal"].to_numpy() / factors
    df["posttax_real"] = df["posttax_nominal"].to_numpy() / factors
    for col in ("pretax_nominal", "posttax_nominal", "pretax_real", "posttax_real"):
        df[f"cum_{col}"] = df[col].cumsum()
    return df


def compute_npv(rate: float, flows: np.ndarray) -> float:
    """Net present value; ``flows[0]`` is the year-0 (undiscounted) outflow."""
    return float(npf.npv(rate, flows))


def compute_irr(flows: np.ndarray) -> float | None:
    """Internal rate of return, or ``None`` when no real root exists (e.g. all-negative)."""
    try:
        irr = npf.irr(np.asarray(flows, dtype=float))
    except (ValueError, FloatingPointError):
        return None
    if irr is None or np.isnan(irr):
        return None
    return float(irr)


def to_real_rate(nominal_rate: float | None, inflation_rate: float) -> float | None:
    """Convert a nominal rate to real via the Fisher relation, or ``None``."""
    if nominal_rate is None:
        return None
    return (1.0 + nominal_rate) / (1.0 + inflation_rate) - 1.0


def zero_crossing_period(cumulative: np.ndarray) -> float | None:
    """Fractional period index where a cumulative series first turns non-negative.

    Returns ``None`` if it never recovers. Generic: pass an annual cumulative to get a
    fractional year, a monthly one to get a fractional month.
    """
    if cumulative[0] >= 0:
        return 0.0
    for i in range(1, len(cumulative)):
        if cumulative[i] >= 0:
            prev, curr = cumulative[i - 1], cumulative[i]
            fraction = -prev / (curr - prev) if curr != prev else 0.0
            return (i - 1) + fraction
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


def _orbital_tco(cfg: SystemConfig, capex_total: float, opex_annual: float) -> float:
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

    depreciation_basis = capex_total
    depreciation = depreciation_schedule(depreciation_basis, cfg)
    cashflow_df = build_financials(cfg, capex_total, opex_annual, depreciation)

    nominal_rate = cfg.discount_rate
    npv_pretax = compute_npv(nominal_rate, cashflow_df["pretax_nominal"].to_numpy())
    npv_posttax = compute_npv(nominal_rate, cashflow_df["posttax_nominal"].to_numpy())

    irr_pretax = compute_irr(cashflow_df["pretax_nominal"].to_numpy())
    irr_posttax = compute_irr(cashflow_df["posttax_nominal"].to_numpy())
    irr_pretax_real = to_real_rate(irr_pretax, cfg.inflation_rate)
    irr_posttax_real = to_real_rate(irr_posttax, cfg.inflation_rate)

    frac = zero_crossing_period(cashflow_df["cum_posttax_nominal"].to_numpy())
    payback_months = frac * 12.0 if frac is not None else None
    total_tax = float(cashflow_df["tax_paid"].sum())

    _, terrestrial_tco = terrestrial_baseline(cfg, physics.total_electrical_load_kW)
    orbital_tco = _orbital_tco(cfg, capex_total, opex_annual)

    capex_ex_launch = capex_total - capex_breakdown["Launch"]
    crossover_launch = crossover_launch_cost(cfg, physics, capex_ex_launch, opex_annual, terrestrial_tco)
    crossover_power = crossover_power_cost(cfg, physics, orbital_tco)

    return EconomicsResult(
        capex_total=capex_total,
        capex_breakdown=capex_breakdown,
        opex_annual=opex_annual,
        opex_breakdown=opex_breakdown,
        revenue_annual=revenue_annual,
        depreciation_basis=depreciation_basis,
        cashflow_df=cashflow_df,
        npv_pretax=npv_pretax,
        npv_posttax=npv_posttax,
        irr_pretax=irr_pretax,
        irr_posttax=irr_posttax,
        irr_pretax_real=irr_pretax_real,
        irr_posttax_real=irr_posttax_real,
        real_discount_rate=to_real_rate(nominal_rate, cfg.inflation_rate),
        payback_months=payback_months,
        total_tax=total_tax,
        terrestrial_tco=terrestrial_tco,
        orbital_tco=orbital_tco,
        crossover_launch_cost=crossover_launch,
        crossover_power_cost=crossover_power,
    )
