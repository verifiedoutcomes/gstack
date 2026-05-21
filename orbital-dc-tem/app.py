"""Orbital Data Center -- Techno-Economic Model.

Streamlit frontend. This is the only module that imports Streamlit; it builds a
``SystemConfig`` from the sidebar widgets, runs the pure physics and economic engines,
and renders the results. Run with::

    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import economic_engine as econ
import physics_engine as phys
from models import SystemConfig, validate_config
from presets import PRESETS
from theme import CAPEX_COLORS, CUSTOM_CSS, DIVERGING_NPV, PALETTE, apply_theme, register_theme

st.set_page_config(page_title="Orbital Data Center TEM", layout="wide")
register_theme()
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PRESET_NAMES = list(PRESETS.keys())
DEFAULT_PRESET = PRESET_NAMES[0]
CONFIG_FIELDS = [f.name for f in fields(SystemConfig)]


# --------------------------------------------------------------------------- helpers
def fmt_money(x: float | None) -> str:
    if x is None or not np.isfinite(x):
        return "N/A"
    a = abs(x)
    if a >= 1e9:
        return f"${x / 1e9:,.2f}B"
    if a >= 1e6:
        return f"${x / 1e6:,.1f}M"
    if a >= 1e3:
        return f"${x / 1e3:,.0f}k"
    return f"${x:,.0f}"


def fmt_pct(x: float | None) -> str:
    return "N/A" if x is None or not np.isfinite(x) else f"{x * 100:,.1f}%"


def fmt_payback(months: float | None) -> str:
    if months is None:
        return "Never"
    if months >= 24:
        return f"{months / 12:,.1f} yr"
    return f"{months:,.0f} mo"


def seed_state(cfg: SystemConfig) -> None:
    for name in CONFIG_FIELDS:
        st.session_state[name] = getattr(cfg, name)


# --------------------------------------------------------------------------- state init
if "_initialized" not in st.session_state:
    seed_state(PRESETS[DEFAULT_PRESET])
    st.session_state["_last_preset"] = DEFAULT_PRESET
    st.session_state["_initialized"] = True


# --------------------------------------------------------------------------- sidebar
st.sidebar.title("Configuration")
preset_choice = st.sidebar.selectbox("Case-study preset", PRESET_NAMES + ["Custom"], key="preset_select")
if preset_choice != "Custom" and preset_choice != st.session_state["_last_preset"]:
    seed_state(PRESETS[preset_choice])
    st.session_state["_last_preset"] = preset_choice

with st.sidebar.expander("Launch & Orbital Dynamics", expanded=True):
    st.slider("Launch cost ($/kg)", min_value=50, max_value=5000, step=50, key="launch_cost_per_kg")
    st.number_input("Orbit altitude (km)", min_value=300.0, max_value=2000.0, step=10.0, key="altitude_km")
    st.number_input("Operational lifespan (years)", min_value=1, max_value=30, step=1, key="lifespan_years")

with st.sidebar.expander("Compute Infrastructure"):
    st.number_input("Number of racks", min_value=1, max_value=10_000, step=1, key="num_racks")
    st.number_input("Power per rack (kW)", min_value=1.0, max_value=1000.0, step=5.0, key="power_per_rack_kW")
    st.number_input("GPUs / TPUs per rack", min_value=1, max_value=512, step=1, key="gpus_per_rack")
    st.slider("Parasitic overhead", min_value=0.0, max_value=0.5, step=0.01, key="parasitic_overhead_fraction")
    st.selectbox("Revenue model", ["per_gpu_hour", "per_flop"], key="revenue_mode")
    st.number_input("Revenue ($/GPU-hour)", min_value=0.0, max_value=50.0, step=0.10, key="revenue_per_gpu_hour")
    st.number_input("Revenue ($/FLOP-year)", min_value=0.0, step=0.0, key="revenue_per_flop", format="%.2e")
    st.number_input("System FLOPS (per_flop mode)", min_value=0.0, step=0.0, key="flops_per_system", format="%.2e")
    st.slider("Utilization", min_value=0.0, max_value=1.0, step=0.01, key="utilization")

with st.sidebar.expander("Spacecraft Engineering"):
    st.slider("Solar cell efficiency", min_value=0.05, max_value=0.45, step=0.01, key="panel_efficiency")
    st.slider("Illumination fraction", min_value=0.5, max_value=1.0, step=0.01, key="illumination_fraction")
    st.slider("Radiator emissivity", min_value=0.1, max_value=1.0, step=0.01, key="radiator_emissivity")
    st.radio("Radiator sides", options=[1, 2], key="radiator_sides", horizontal=True)
    st.number_input("Radiator width (m)", min_value=0.5, max_value=100.0, step=0.5, key="radiator_width_m")
    st.number_input("Deep-space sink temp (K)", min_value=2.7, max_value=300.0, step=1.0, key="T_sink_K")
    st.number_input("Max surface temp (K)", min_value=273.0, max_value=400.0, step=1.0, key="T_surface_max_K")
    st.number_input("Solar areal density (kg/m^2)", min_value=0.1, max_value=20.0, step=0.25, key="solar_areal_density_kg_m2")
    st.number_input("Radiator areal density (kg/m^2)", min_value=1.0, max_value=30.0, step=0.5, key="radiator_areal_density_kg_m2")
    st.number_input("Rack mass (kg)", min_value=50.0, max_value=5000.0, step=50.0, key="rack_mass_kg")
    st.slider("Structural mass multiplier", min_value=1.0, max_value=3.0, step=0.05, key="structural_mass_multiplier")
    st.checkbox("Model annual degradation", key="enable_degradation")
    st.slider("Annual degradation", min_value=0.0, max_value=0.2, step=0.005, key="annual_degradation_pct")

with st.sidebar.expander("CapEx & Hardware Costs"):
    st.number_input("Rack hardware unit cost ($)", min_value=0.0, step=100_000.0, key="rack_unit_cost", format="%.0f")
    st.slider("Space-qualification premium", min_value=1.0, max_value=4.0, step=0.1, key="space_qualification_premium")
    st.number_input("Solar cost ($/m^2)", min_value=0.0, step=25.0, key="solar_cost_per_m2", format="%.0f")
    st.number_input("Radiator cost ($/m^2)", min_value=0.0, step=50.0, key="radiator_cost_per_m2", format="%.0f")
    st.slider("R&D + integration overhead", min_value=0.0, max_value=1.0, step=0.05, key="rd_integration_overhead_pct")

with st.sidebar.expander("Operating Costs (annual)"):
    st.number_input("Ground station ($/yr)", min_value=0.0, step=100_000.0, key="ground_station_cost_annual", format="%.0f")
    st.number_input("Station-keeping ($/yr)", min_value=0.0, step=100_000.0, key="stationkeeping_cost_annual", format="%.0f")
    st.slider("Insurance (% hardware/yr)", min_value=0.0, max_value=0.5, step=0.01, key="insurance_pct_of_hardware")
    st.number_input("Ops / management ($/yr)", min_value=0.0, step=100_000.0, key="terrestrial_ops_cost_annual", format="%.0f")

with st.sidebar.expander("Financial Framework"):
    st.slider("Discount rate", min_value=0.04, max_value=0.20, step=0.005, key="discount_rate")

with st.sidebar.expander("Terrestrial Baseline"):
    st.slider("Grid power ($/kWh)", min_value=0.0, max_value=0.5, step=0.005, key="grid_cost_per_kWh")
    st.slider("PUE", min_value=1.0, max_value=2.5, step=0.05, key="pue")
    st.number_input("Land ($/yr)", min_value=0.0, step=100_000.0, key="terrestrial_land_cost_annual", format="%.0f")
    st.number_input("Water / cooling ($/yr)", min_value=0.0, step=100_000.0, key="terrestrial_water_cost_annual", format="%.0f")
    st.number_input("Ground ops ($/yr)", min_value=0.0, step=100_000.0, key="terrestrial_ground_ops_annual", format="%.0f")
    st.number_input("Build cost ($/MW)", min_value=0.0, step=1_000_000.0, key="terrestrial_capex_per_MW", format="%.0f")


# --------------------------------------------------------------------------- build + validate
cfg = SystemConfig(**{name: st.session_state[name] for name in CONFIG_FIELDS})
errors, warnings = validate_config(cfg)

st.title("Orbital Data Center -- Techno-Economic Model")
st.caption(
    "First-principles physics + full DCF economics for compute in orbit, benchmarked "
    "against an equal-power terrestrial data center."
)

if errors:
    for e in errors:
        st.error(e)
    st.stop()
for w in warnings:
    st.warning(w)

physics = phys.run_physics(cfg)
if not physics.survivability_ok:
    st.error(
        f"Thermal survivability failure: with a {cfg.T_sink_K:.0f} K sink and a "
        f"{cfg.T_surface_max_K:.0f} K surface cap, no finite radiator can reject the heat. "
        "Lower the sink temperature or raise the surface cap."
    )
    st.stop()

economics = econ.run_economics(cfg, physics)


# --------------------------------------------------------------------------- KPI header
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Total CapEx", fmt_money(economics.capex_total))
k2.metric("System Mass", f"{physics.total_dry_mass_kg / 1000:,.1f} t")
k3.metric("Radiator Length", f"{physics.radiator_length_ft:,.0f} ft")
k4.metric("NPV", fmt_money(economics.npv))
k5.metric("IRR", fmt_pct(economics.irr))
k6.metric("Payback", fmt_payback(economics.payback_months))

for note in physics.notes:
    st.info(note)


# --------------------------------------------------------------------------- charts
def cashflow_fig() -> go.Figure:
    df = economics.cashflow_df
    cum_m = df["cumulative"].to_numpy() / 1e6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["year"], y=cum_m, mode="lines+markers",
            line=dict(color=PALETTE["accent"], width=3),
            marker=dict(size=7), name="Cumulative cash flow",
            hovertemplate="Year %{x}: $%{y:.1f}M<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PALETTE["muted"], line_width=1)
    if economics.payback_months is not None:
        be_year = economics.payback_months / 12
        fig.add_vline(x=be_year, line_dash="dot", line_color=PALETTE["positive"])
        fig.add_annotation(
            x=be_year, y=0, text=f"break-even ~{be_year:.1f} yr",
            showarrow=True, arrowhead=2, ax=40, ay=-40,
            font=dict(color=PALETTE["positive"], size=12),
        )
    fig.update_layout(title="Cumulative cash flow", xaxis_title="Year", yaxis_title="$M")
    return apply_theme(fig, height=380)


def capex_fig() -> go.Figure:
    fig = go.Figure()
    for (label, value), color in zip(economics.capex_breakdown.items(), CAPEX_COLORS):
        fig.add_trace(
            go.Bar(
                y=["CapEx"], x=[value / 1e6], name=label, orientation="h",
                marker_color=color,
                hovertemplate=f"{label}: $%{{x:.1f}}M<extra></extra>",
            )
        )
    fig.update_layout(
        title="CapEx breakdown", barmode="stack", xaxis_title="$M",
        yaxis=dict(showticklabels=False),
    )
    return apply_theme(fig, height=240)


@st.cache_data(show_spinner=False)
def advantage_grid(cfg_dict: dict, launch_lo: float, launch_hi: float, grid_lo: float, grid_hi: float, n: int):
    """Lifetime orbital cost advantage (terrestrial TCO - orbital TCO, $M) over a
    launch-cost x grid-cost grid. Physics is independent of both axes, so it runs once."""
    local = SystemConfig(**cfg_dict)
    local_phys = phys.run_physics(local)
    capex_total, breakdown = econ.compute_capex(local, local_phys)
    opex_annual, _ = econ.compute_opex(local, breakdown["Hardware"])
    capex_ex_launch = capex_total - breakdown["Launch"]
    launches = np.linspace(launch_lo, launch_hi, n)
    grids = np.linspace(grid_lo, grid_hi, n)
    z = np.zeros((n, n))
    for i, g in enumerate(grids):
        _, terr = econ.terrestrial_baseline(replace(local, grid_cost_per_kWh=g), local_phys.total_electrical_load_kW)
        orbital = capex_ex_launch + launches * local_phys.total_dry_mass_kg + opex_annual * local.lifespan_years
        z[i, :] = (terr - orbital) / 1e6
    return launches, grids, z


def sensitivity_fig() -> go.Figure:
    launch_hi = max(cfg.launch_cost_per_kg * 2, 3000)
    grid_hi = max(cfg.grid_cost_per_kWh * 2, 0.20)
    launches, grids, z = advantage_grid(dict(cfg.__dict__), 0.0, launch_hi, 0.0, grid_hi, 40)
    fig = go.Figure(
        go.Heatmap(
            x=launches, y=grids, z=z, colorscale=DIVERGING_NPV, zmid=0,
            colorbar=dict(title="$M"),
            hovertemplate="Launch $%{x:.0f}/kg, grid $%{y:.3f}/kWh<br>advantage $%{z:.0f}M<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[cfg.launch_cost_per_kg], y=[cfg.grid_cost_per_kWh], mode="markers+text",
            marker=dict(color=PALETTE["ink"], size=12, symbol="x"),
            text=["current"], textposition="top center",
            textfont=dict(color=PALETTE["ink"]), showlegend=False, hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Orbital cost advantage vs terrestrial (green = orbital cheaper)",
        xaxis_title="Launch cost ($/kg)", yaxis_title="Grid power ($/kWh)",
    )
    return apply_theme(fig, height=440)


left, right = st.columns([3, 2])
with left:
    st.plotly_chart(cashflow_fig(), use_container_width=True)
with right:
    st.plotly_chart(capex_fig(), use_container_width=True)

st.plotly_chart(sensitivity_fig(), use_container_width=True)


# --------------------------------------------------------------------------- terrestrial card
st.subheader("Orbital vs terrestrial")
num_gpus = cfg.num_racks * cfg.gpus_per_rack
gpu_hours_life = num_gpus * min(max(cfg.utilization, 0.0), 1.0) * 8760 * cfg.lifespan_years
orbital_per_hr = economics.orbital_tco / gpu_hours_life if gpu_hours_life > 0 else None
terr_per_hr = economics.terrestrial_tco / gpu_hours_life if gpu_hours_life > 0 else None
advantage = economics.terrestrial_tco - economics.orbital_tco

c1, c2, c3, c4 = st.columns(4)
c1.metric("Orbital lifetime TCO", fmt_money(economics.orbital_tco))
c2.metric("Terrestrial lifetime TCO", fmt_money(economics.terrestrial_tco))
c3.metric("Orbital $/GPU-hr", fmt_money(orbital_per_hr) if orbital_per_hr else "N/A")
c4.metric("Terrestrial $/GPU-hr", fmt_money(terr_per_hr) if terr_per_hr else "N/A")

if advantage > 0:
    st.success(
        f"Orbital is cheaper over the {cfg.lifespan_years}-year lifetime by "
        f"{fmt_money(advantage)} of total cost of ownership."
    )
else:
    st.warning(
        f"Terrestrial is cheaper by {fmt_money(-advantage)} over the "
        f"{cfg.lifespan_years}-year lifetime at these inputs."
    )

cross = []
if economics.crossover_launch_cost is not None:
    cross.append(f"orbital wins below **${economics.crossover_launch_cost:,.0f}/kg** launch cost")
if economics.crossover_power_cost is not None:
    cross.append(f"orbital wins above **${economics.crossover_power_cost:,.3f}/kWh** terrestrial grid power")
if cross:
    st.caption("Crossover points (holding everything else fixed): " + "; ".join(cross) + ".")
