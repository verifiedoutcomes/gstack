"""First-principles physics for an orbital data center.

Pure module: imports only ``constants``, ``models`` and the stdlib. It never imports
Streamlit so it can be unit-tested headlessly. All public functions operate on a
``SystemConfig`` and return plain numbers or a ``PhysicsResult``.

Key coupling the model makes explicit: essentially 100% of the electrical power drawn by
the compute hardware turns into waste heat, so the radiator must reject the *same* power
the solar array has to supply. Solar sizing and thermal sizing are not independent.
"""

from __future__ import annotations

from constants import FT_PER_M, G_ORBIT, SIGMA, WATTS_PER_KW
from models import PhysicsResult, SystemConfig


def compute_power_balance(cfg: SystemConfig) -> tuple[float, float]:
    """Return ``(compute_power_kW, total_electrical_load_kW)``.

    Total load adds a parasitic overhead (comms, attitude control, thermal loop) on top
    of raw compute. That total is both the power the array must generate and the heat the
    radiator must reject.
    """
    compute_power_kW = cfg.num_racks * cfg.power_per_rack_kW
    total_load_kW = compute_power_kW * (1.0 + cfg.parasitic_overhead_fraction)
    return compute_power_kW, total_load_kW


def required_solar_area(load_kW: float, cfg: SystemConfig) -> float:
    """Solar array area (m^2) needed to generate ``load_kW`` of electrical power.

    ``A = P / (eta . G_orbit . illumination)``.
    """
    illumination = min(cfg.illumination_fraction, 1.0)
    denominator = cfg.panel_efficiency * G_ORBIT * illumination
    if denominator <= 0:
        raise ValueError("Solar generation denominator is non-positive (check efficiency/illumination).")
    return (load_kW * WATTS_PER_KW) / denominator


def solar_generation(area_m2: float, cfg: SystemConfig) -> float:
    """Forward check: electrical power (kW) produced by a given array area."""
    illumination = min(cfg.illumination_fraction, 1.0)
    power_W = area_m2 * cfg.panel_efficiency * G_ORBIT * illumination
    return power_W / WATTS_PER_KW


def radiator_flux(cfg: SystemConfig, T_radiator: float) -> float:
    """Net single-face radiative heat flux (W/m^2) after environmental loading.

    Gross emission via Stefan-Boltzmann: ``epsilon . sigma . (T_radiator^4 - T_sink^4)``.
    A radiator in low Earth orbit also *absorbs* incident planetary IR (Earth radiates at
    an effective ~255 K, ~240 W/m^2) and albedo. By Kirchhoff's law the absorptivity for
    long-wave Earth IR matches the radiator's emissivity, so the absorbed load is
    ``epsilon . environmental_thermal_load_W_m2``. The result is the *net* rejection:

        q_net = epsilon . (sigma . (T_rad^4 - T_sink^4) - environmental_load)

    Real spacecraft radiators in LEO reject 100-350 W/m^2 net (NASA SST-SOA), and this
    formula recovers ~300 W/m^2 single-face at 318 K with 250 W/m^2 environmental load.
    Set ``environmental_thermal_load_W_m2 = 0`` for the deep-space ideal.
    """
    gross = SIGMA * (T_radiator**4 - cfg.T_sink_K**4)
    return cfg.radiator_emissivity * (gross - cfg.environmental_thermal_load_W_m2)


def required_radiator_area(P_dissipated_kW: float, cfg: SystemConfig) -> tuple[float, float, bool]:
    """Physical radiator panel area to reject ``P_dissipated_kW`` of heat.

    Returns ``(panel_area_m2, T_radiator_used_K, survivability_ok)``. The radiator is sized
    at the hottest allowed surface temperature (``T_surface_max_K``) -- the point of minimum
    area. ``radiator_sides`` (1 or 2) multiplies the radiating surface, so a two-sided panel
    needs half the footprint. Survivability fails only when the flux is non-positive (sink at
    or above the temperature cap), where no finite radiator could ever reject the heat.
    """
    T_used = cfg.T_surface_max_K
    q_single = radiator_flux(cfg, T_used)
    effective_flux = q_single * cfg.radiator_sides
    if effective_flux <= 0:
        return float("inf"), T_used, False
    area = (P_dissipated_kW * WATTS_PER_KW) / effective_flux
    return area, T_used, True


def radiator_length(area_m2: float, cfg: SystemConfig) -> tuple[float, float]:
    """Convert panel area to a deployed length, in metres and feet, for a fixed width."""
    if cfg.radiator_width_m <= 0:
        raise ValueError("Radiator width must be greater than 0 m.")
    length_m = area_m2 / cfg.radiator_width_m
    return length_m, length_m * FT_PER_M


def compute_mass_model(cfg: SystemConfig, solar_area_m2: float, radiator_area_m2: float) -> dict[str, float]:
    """Dry-mass breakdown driven by compute count, solar area and radiator area.

    Structure/bus mass is a multiplier on the summed subsystem mass: a multiplier of 1.0
    adds nothing, 1.5 adds 50% for structure, harness, avionics and propulsion dry mass.
    """
    compute_mass = cfg.num_racks * cfg.rack_mass_kg
    solar_mass = solar_area_m2 * cfg.solar_areal_density_kg_m2
    radiator_mass = radiator_area_m2 * cfg.radiator_areal_density_kg_m2
    subsystem_mass = compute_mass + solar_mass + radiator_mass
    structural_mass = subsystem_mass * (cfg.structural_mass_multiplier - 1.0)
    total_dry_mass = subsystem_mass + structural_mass
    return {
        "compute_mass_kg": compute_mass,
        "solar_mass_kg": solar_mass,
        "radiator_mass_kg": radiator_mass,
        "structural_mass_kg": structural_mass,
        "total_dry_mass_kg": total_dry_mass,
    }


def apply_degradation(value: float, year: int, cfg: SystemConfig) -> float:
    """Decline ``value`` by the annual degradation rate over ``year`` years (0 = no decay)."""
    if not cfg.enable_degradation:
        return value
    factor = (1.0 - cfg.annual_degradation_pct) ** year
    return value * max(factor, 0.0)


def run_physics(cfg: SystemConfig) -> PhysicsResult:
    """Orchestrator: assemble a full ``PhysicsResult`` from a config.

    Does not raise on a thermally impossible config -- instead it returns a result with
    ``survivability_ok=False`` and infinite radiator metrics, which the UI surfaces as an
    error. It does raise (via the helpers) on degenerate inputs like zero efficiency.
    """
    notes: list[str] = []
    compute_power_kW, total_load_kW = compute_power_balance(cfg)

    solar_area = required_solar_area(total_load_kW, cfg)
    radiator_area, T_used, survivable = required_radiator_area(total_load_kW, cfg)
    q_single = radiator_flux(cfg, T_used)

    if not survivable:
        notes.append(
            f"Thermal sink ({cfg.T_sink_K:.0f} K) is at or above the {cfg.T_surface_max_K:.0f} K "
            "surface cap -- no finite radiator can reject the heat."
        )
        return PhysicsResult(
            compute_power_kW=compute_power_kW,
            total_electrical_load_kW=total_load_kW,
            solar_array_area_m2=solar_area,
            solar_mass_kg=solar_area * cfg.solar_areal_density_kg_m2,
            q_per_area_W_m2=q_single,
            radiator_area_m2=float("inf"),
            radiator_length_m=float("inf"),
            radiator_length_ft=float("inf"),
            radiator_mass_kg=float("inf"),
            compute_mass_kg=cfg.num_racks * cfg.rack_mass_kg,
            structural_mass_kg=float("inf"),
            total_dry_mass_kg=float("inf"),
            T_radiator_used_K=T_used,
            survivability_ok=False,
            notes=notes,
        )

    length_m, length_ft = radiator_length(radiator_area, cfg)
    masses = compute_mass_model(cfg, solar_area, radiator_area)

    if length_ft > 1000.0:
        notes.append(
            f"Required radiator is {length_ft:,.0f} ft as a single panel -- likely better "
            "modeled as a constellation of smaller satellites."
        )

    return PhysicsResult(
        compute_power_kW=compute_power_kW,
        total_electrical_load_kW=total_load_kW,
        solar_array_area_m2=solar_area,
        solar_mass_kg=masses["solar_mass_kg"],
        q_per_area_W_m2=q_single,
        radiator_area_m2=radiator_area,
        radiator_length_m=length_m,
        radiator_length_ft=length_ft,
        radiator_mass_kg=masses["radiator_mass_kg"],
        compute_mass_kg=masses["compute_mass_kg"],
        structural_mass_kg=masses["structural_mass_kg"],
        total_dry_mass_kg=masses["total_dry_mass_kg"],
        T_radiator_used_K=T_used,
        survivability_ok=True,
        notes=notes,
    )
