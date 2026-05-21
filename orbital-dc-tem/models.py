"""Shared data structures: the input config and the two result objects.

These dataclasses are the contract between the pure engines and the Streamlit UI.
``SystemConfig`` gives every field a sensible default so the app boots with no preset
selected. ``validate_config`` is a pure function (no Streamlit) so it can be reused and
tested independently of the UI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class SystemConfig:
    """All user-tunable inputs for one orbital data center configuration.

    Every field has a default; presets override them, and the UI binds each widget to a
    field of the same name via ``st.session_state``.
    """

    name: str = "Custom"

    # --- Compute infrastructure ---
    num_racks: int = 50
    power_per_rack_kW: float = 100.0
    gpus_per_rack: int = 72
    parasitic_overhead_fraction: float = 0.10  # comms/ADCS/thermal-loop draw on top of compute

    # --- Solar ---
    panel_efficiency: float = 0.30  # space-grade multi-junction (0.28-0.32)
    illumination_fraction: float = 1.0  # dawn-dusk SSO ~ continuous sun; <1 adds eclipse
    solar_areal_density_kg_m2: float = 1.75  # deployable array mass per m^2

    # --- Thermal / radiator ---
    radiator_emissivity: float = 0.90
    T_sink_K: float = 3.0  # deep-space sink; raise for LEO Earth-IR/albedo loading
    T_surface_max_K: float = 318.0  # 45 C chassis/junction survivability cap
    radiator_sides: int = 2  # 1 or 2 -- two-sided panels halve required footprint
    radiator_width_m: float = 10.0  # used to convert area to a length KPI
    radiator_areal_density_kg_m2: float = 8.0  # deployable radiator mass per m^2 (5-12)

    # --- Mass ---
    rack_mass_kg: float = 900.0
    structural_mass_multiplier: float = 1.5  # bus/structure factor on subsystem mass (1.0 = none)

    # --- Degradation (optional) ---
    enable_degradation: bool = True
    annual_degradation_pct: float = 0.025  # per-year decline in solar + compute output

    # --- Launch & orbit ---
    launch_cost_per_kg: float = 500.0
    altitude_km: float = 650.0
    lifespan_years: int = 10

    # --- CapEx params ---
    rack_unit_cost: float = 3_000_000.0
    space_qualification_premium: float = 1.5  # multiplier on hardware for rad-hardening/qual
    solar_cost_per_m2: float = 300.0
    radiator_cost_per_m2: float = 1_000.0
    rd_integration_overhead_pct: float = 0.20  # R&D + bus integration, % of built subsystems

    # --- OpEx params (annual) ---
    ground_station_cost_annual: float = 5_000_000.0
    stationkeeping_cost_annual: float = 2_000_000.0
    insurance_pct_of_hardware: float = 0.15  # ANNUAL, 15-20% of hardware value
    terrestrial_ops_cost_annual: float = 3_000_000.0

    # --- Revenue ---
    revenue_mode: str = "per_gpu_hour"  # "per_gpu_hour" | "per_flop"
    revenue_per_gpu_hour: float = 2.50
    revenue_per_flop: float = 0.0  # $ per FLOP-year (per_flop mode)
    flops_per_system: float = 0.0  # aggregate FLOPS (per_flop mode)
    utilization: float = 0.88  # 0-1 duty factor

    # --- Financial framework ---
    discount_rate: float = 0.10
    grid_cost_per_kWh: float = 0.07
    pue: float = 1.25
    terrestrial_land_cost_annual: float = 1_000_000.0
    terrestrial_water_cost_annual: float = 500_000.0
    terrestrial_ground_ops_annual: float = 4_000_000.0
    terrestrial_capex_per_MW: float = 10_000_000.0

    # --- Tax, depreciation & inflation ---
    tax_rate: float = 0.30
    enable_tax_loss_carryforward: bool = True
    overwrite_depreciation: bool = False  # off = straight-line over the project life
    depreciation_scheme: str = "SLN"  # "SLN" | "DB" | "MACRS"
    depreciation_years_sln: int = 20  # SLN recovery period
    depreciation_rate_db: float = 0.05  # declining-balance rate
    macrs_years: int = 5  # MACRS recovery class (3/5/7/10/15/20)
    inflation_rate: float = 0.025  # annual inflation from the base year


@dataclass
class PhysicsResult:
    """Outputs of the physics engine for one configuration."""

    compute_power_kW: float
    total_electrical_load_kW: float  # compute x (1 + parasitic); also = heat to reject
    solar_array_area_m2: float
    solar_mass_kg: float
    q_per_area_W_m2: float  # single-face radiative flux at the operating temperature
    radiator_area_m2: float  # physical panel area (effective area = panel x sides)
    radiator_length_m: float
    radiator_length_ft: float
    radiator_mass_kg: float
    compute_mass_kg: float
    structural_mass_kg: float
    total_dry_mass_kg: float
    T_radiator_used_K: float
    survivability_ok: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class EconomicsResult:
    """Outputs of the economic engine for one configuration.

    NPV is inflation-invariant under consistent discounting, so a single pre-tax and
    post-tax value is reported. IRR differs in real vs nominal terms, so both are given.
    """

    capex_total: float
    capex_breakdown: dict[str, float]
    opex_annual: float
    opex_breakdown: dict[str, float]
    revenue_annual: float
    depreciation_basis: float
    cashflow_df: pd.DataFrame
    npv_pretax: float
    npv_posttax: float
    irr_pretax: float | None  # nominal
    irr_posttax: float | None  # nominal
    irr_pretax_real: float | None
    irr_posttax_real: float | None
    real_discount_rate: float | None
    payback_months: float | None  # post-tax, nominal
    total_tax: float
    terrestrial_tco: float
    orbital_tco: float
    crossover_launch_cost: float | None
    crossover_power_cost: float | None


def validate_config(cfg: SystemConfig) -> tuple[list[str], list[str]]:
    """Check a config for hard errors and soft warnings.

    Returns ``(errors, warnings)``. Errors block the run (the engines would raise or
    produce nonsense); warnings indicate values that were silently clamped or are
    physically suspect. Pure -- no Streamlit, safe to unit test.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if cfg.panel_efficiency <= 0:
        errors.append("Solar panel efficiency must be greater than 0.")
    elif cfg.panel_efficiency > 0.5:
        warnings.append(
            f"Panel efficiency {cfg.panel_efficiency:.0%} exceeds any real space cell (~32% max)."
        )

    if cfg.radiator_width_m <= 0:
        errors.append("Radiator width must be greater than 0 m.")
    if cfg.radiator_emissivity <= 0:
        errors.append("Radiator emissivity must be greater than 0.")
    if cfg.radiator_sides not in (1, 2):
        errors.append("Radiator sides must be 1 or 2.")
    if cfg.illumination_fraction <= 0:
        errors.append("Illumination fraction must be greater than 0.")

    if cfg.lifespan_years <= 0:
        errors.append("Operational lifespan must be at least 1 year.")
    if cfg.num_racks <= 0:
        errors.append("Number of racks must be at least 1.")
    if cfg.power_per_rack_kW <= 0:
        errors.append("Power per rack must be greater than 0 kW.")

    for label, value in (
        ("launch cost", cfg.launch_cost_per_kg),
        ("rack mass", cfg.rack_mass_kg),
        ("discount rate", cfg.discount_rate),
        ("grid power cost", cfg.grid_cost_per_kWh),
    ):
        if value < 0:
            errors.append(f"{label.capitalize()} cannot be negative.")

    if cfg.structural_mass_multiplier < 1.0:
        warnings.append("Structural mass multiplier below 1.0 implies negative structure mass.")
    if cfg.annual_degradation_pct >= 0.5:
        warnings.append("Annual degradation >= 50%/yr will collapse output within a year or two.")

    # Tax / depreciation / inflation
    if not 0.0 <= cfg.tax_rate < 1.0:
        errors.append("Tax rate must be between 0% and 100%.")
    if cfg.inflation_rate <= -1.0:
        errors.append("Inflation rate must be greater than -100%.")
    elif cfg.inflation_rate >= 0.25:
        warnings.append(f"Inflation of {cfg.inflation_rate:.0%}/yr is very high.")
    if cfg.overwrite_depreciation:
        if cfg.depreciation_scheme not in ("SLN", "DB", "MACRS"):
            errors.append("Depreciation scheme must be SLN, DB or MACRS.")
        if cfg.depreciation_scheme == "SLN" and cfg.depreciation_years_sln < 1:
            errors.append("SLN expenditure years must be at least 1.")
        if cfg.depreciation_scheme == "DB" and not 0.0 < cfg.depreciation_rate_db <= 1.0:
            errors.append("Declining-balance rate must be between 0% and 100%.")
        if cfg.depreciation_scheme == "MACRS" and cfg.macrs_years not in (3, 5, 7, 10, 15, 20):
            errors.append("MACRS recovery class must be one of 3, 5, 7, 10, 15 or 20 years.")

    return errors, warnings
