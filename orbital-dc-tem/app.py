"""Orbital Data Center -- Techno-Economic Model.

Streamlit frontend. This is the only module that imports Streamlit; it builds a
``SystemConfig`` from the sidebar widgets, runs the pure physics and economic engines,
and renders the results. Run with::

    streamlit run app.py
"""

from __future__ import annotations

from dataclasses import fields, replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import economic_engine as econ
import physics_engine as phys
from models import SystemConfig, validate_config
from presets import PRESETS
from theme import apply_theme, build_css, get_capex_colors, get_diverging, get_palette, register_themes

st.set_page_config(page_title="Orbital Data Center TEM", layout="wide")
register_themes()

# Dark/light toggle must be read before injecting CSS and theming charts.
dark_mode = st.sidebar.toggle("Dark mode", value=False, key="dark_mode")
display_currency = st.sidebar.selectbox("Currency", ["USD", "EUR"], key="display_currency")
exchange_rate = st.sidebar.number_input(
    "USD per EUR", min_value=0.5, max_value=2.0, value=1.08, step=0.01,
    key="usd_per_eur", disabled=(display_currency == "USD"),
    help="Exchange rate. 1.08 means 1 EUR = 1.08 USD.",
)
st.sidebar.caption("Inputs in USD. Outputs shown in the selected currency.")
MODE = "dark" if dark_mode else "light"
CURRENCY_SYMBOL = "$" if display_currency == "USD" else "€"
CURRENCY_FACTOR = 1.0 if display_currency == "USD" else (1.0 / exchange_rate)
PAL = get_palette(MODE)
CAPEX_COLORS = get_capex_colors(MODE)
DIVERGING = get_diverging(MODE)
st.markdown(build_css(MODE), unsafe_allow_html=True)

PRESET_NAMES = list(PRESETS.keys())
DEFAULT_PRESET = PRESET_NAMES[0]
CONFIG_FIELDS = [f.name for f in fields(SystemConfig)]

# Inputs ranked by the tornado chart. Each gets perturbed +/-20% to show how much the
# post-tax NPV moves -- the longest bars are the highest-leverage cost optimizations.
TORNADO_FIELDS: tuple[tuple[str, str], ...] = (
    ("launch_cost_per_kg", "Launch cost ($/kg)"),
    ("rack_unit_cost", "Rack hardware unit cost"),
    ("revenue_per_gpu_hour", "Revenue per GPU-hour"),
    ("space_qualification_premium", "Space-qualification premium"),
    ("rd_integration_overhead_pct", "R&D + integration overhead"),
    ("insurance_pct_of_hardware", "Annual insurance rate"),
    ("solar_cost_per_m2", "Solar manufacturing $/m²"),
    ("radiator_cost_per_m2", "Radiator manufacturing $/m²"),
    ("panel_efficiency", "Solar panel efficiency"),
    ("radiator_areal_density_kg_m2", "Radiator areal density"),
    ("utilization", "Utilization"),
    ("annual_degradation_pct", "Annual degradation"),
)


# --------------------------------------------------------------------------- helpers
def fmt_money(x: float | None) -> str:
    """Money in the active display currency (the model stores USD)."""
    if x is None or not np.isfinite(x):
        return "N/A"
    v = x * CURRENCY_FACTOR
    sign = "-" if v < 0 else ""
    a = abs(v)
    sym = CURRENCY_SYMBOL
    if a >= 1e9:
        return f"{sign}{sym}{a / 1e9:,.2f}B"
    if a >= 1e6:
        return f"{sign}{sym}{a / 1e6:,.1f}M"
    if a >= 1e3:
        return f"{sign}{sym}{a / 1e3:,.0f}k"
    return f"{sign}{sym}{a:,.0f}"


def fmt_rate(x: float | None) -> str:
    """Per-hour money in the active currency, cents preserved."""
    if x is None or not np.isfinite(x):
        return "N/A"
    return f"{CURRENCY_SYMBOL}{x * CURRENCY_FACTOR:,.2f}"


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
    revenue_mode = st.selectbox("Revenue model", ["per_gpu_hour", "per_flop"], key="revenue_mode")
    st.number_input("Revenue ($/GPU-hour)", min_value=0.0, max_value=50.0, step=0.10,
                    key="revenue_per_gpu_hour", disabled=revenue_mode != "per_gpu_hour")
    st.number_input("Revenue ($/FLOP-year)", min_value=0.0, step=1e-18,
                    key="revenue_per_flop", format="%.2e", disabled=revenue_mode != "per_flop")
    st.number_input("System FLOPS", min_value=0.0, step=1e15,
                    key="flops_per_system", format="%.2e", disabled=revenue_mode != "per_flop")
    st.slider("Utilization", min_value=0.0, max_value=1.0, step=0.01, key="utilization")

with st.sidebar.expander("Spacecraft Engineering"):
    st.slider("Solar cell efficiency", min_value=0.05, max_value=0.45, step=0.01, key="panel_efficiency")
    st.slider("Illumination fraction", min_value=0.5, max_value=1.0, step=0.01, key="illumination_fraction")
    st.slider("Radiator emissivity", min_value=0.1, max_value=1.0, step=0.01, key="radiator_emissivity")
    st.radio("Radiator sides", options=[1, 2], key="radiator_sides", horizontal=True)
    st.number_input("Radiator width (m)", min_value=0.5, max_value=100.0, step=0.5, key="radiator_width_m")
    st.number_input("Deep-space sink temp (K)", min_value=2.7, max_value=300.0, step=1.0, key="T_sink_K")
    st.number_input("Max surface temp (K)", min_value=273.0, max_value=400.0, step=1.0, key="T_surface_max_K")
    st.number_input(
        "Environmental load (W/m²)", min_value=0.0, max_value=500.0, step=10.0,
        key="environmental_thermal_load_W_m2",
        help="Incident Earth IR + albedo absorbed by the radiator. ~250 W/m² in LEO; 0 for deep space.",
    )
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
    st.slider("Insurance (fraction of hardware/yr)", min_value=0.0, max_value=0.5, step=0.01, key="insurance_pct_of_hardware")
    st.number_input("Ops / management ($/yr)", min_value=0.0, step=100_000.0, key="terrestrial_ops_cost_annual", format="%.0f")

with st.sidebar.expander("Financial Framework"):
    st.slider("Discount rate (nominal)", min_value=0.04, max_value=0.20, step=0.005, key="discount_rate")

with st.sidebar.expander("Tax, Depreciation & Inflation"):
    st.slider("Tax rate", min_value=0.0, max_value=0.60, step=0.01, key="tax_rate", help="Fraction, e.g. 0.30 = 30%")
    st.checkbox("Tax loss carry-forward", key="enable_tax_loss_carryforward",
                help="Bank losses to offset future taxable income")
    st.slider("Inflation from base year", min_value=0.0, max_value=0.15, step=0.005, key="inflation_rate",
              help="Fraction per year, e.g. 0.025 = 2.5%")
    overwrite_dep = st.checkbox("Overwrite depreciation", key="overwrite_depreciation",
                                help="Off = straight-line over the project life")
    dep_scheme = st.selectbox("Depreciation scheme", ["SLN", "DB", "MACRS"],
                              key="depreciation_scheme", disabled=not overwrite_dep)
    st.number_input("Expenditure years (SLN)", min_value=1, max_value=40, step=1,
                    key="depreciation_years_sln", disabled=not (overwrite_dep and dep_scheme == "SLN"))
    st.slider("Declining-balance rate (DB)", min_value=0.01, max_value=0.50, step=0.01,
              key="depreciation_rate_db", disabled=not (overwrite_dep and dep_scheme == "DB"),
              help="Fraction, e.g. 0.05 = 5%")
    st.selectbox("MACRS recovery class (years)", [3, 5, 7, 10, 15, 20],
                 key="macrs_years", disabled=not (overwrite_dep and dep_scheme == "MACRS"))

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
k4.metric("NPV", fmt_money(economics.npv_posttax))
k5.metric("IRR", fmt_pct(economics.irr_posttax))
k6.metric("Payback", fmt_payback(economics.payback_months))
st.caption(
    f"NPV and IRR shown post-tax, nominal (tax {cfg.tax_rate:.0%}, inflation "
    f"{cfg.inflation_rate:.1%}/yr). Pre-tax and real figures in Financial detail below."
)

for note in physics.notes:
    st.info(note)


# --------------------------------------------------------------------------- charts
def cashflow_fig(column: str, label: str) -> go.Figure:
    df = economics.cashflow_df
    raw = df[column].to_numpy()
    y = raw * CURRENCY_FACTOR / 1e6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["year"], y=y, mode="lines+markers",
            line=dict(color=PAL["accent"], width=3),
            marker=dict(size=7), name="Cumulative cash flow",
            hovertemplate=f"Year %{{x}}: {CURRENCY_SYMBOL}%{{y:.1f}}M<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dash", line_color=PAL["muted"], line_width=1)
    frac = econ.zero_crossing_period(raw)
    if frac is not None:
        fig.add_vline(x=frac, line_dash="dot", line_color=PAL["positive"])
        fig.add_annotation(
            x=frac, y=0, text=f"break-even ~{frac:.1f} yr",
            showarrow=True, arrowhead=2, ax=40, ay=-40,
            font=dict(color=PAL["positive"], size=12),
        )
    fig.update_layout(
        title=f"Cumulative cash flow ({label})",
        xaxis_title="Year", yaxis_title=f"{CURRENCY_SYMBOL}M",
    )
    return apply_theme(fig, MODE, height=380)


def capex_fig() -> go.Figure:
    fig = go.Figure()
    for (label, value), color in zip(economics.capex_breakdown.items(), CAPEX_COLORS):
        fig.add_trace(
            go.Bar(
                y=["CapEx"], x=[value * CURRENCY_FACTOR / 1e6], name=label, orientation="h",
                marker_color=color,
                hovertemplate=f"{label}: {CURRENCY_SYMBOL}%{{x:.1f}}M<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"CapEx breakdown ({CURRENCY_SYMBOL}M)", barmode="stack", xaxis_title=None,
        yaxis=dict(showticklabels=False),
        legend=dict(orientation="h", yanchor="top", y=-0.25, x=0.0),
    )
    return apply_theme(fig, MODE, height=280)


@st.cache_data(show_spinner=False)
def advantage_grid(cfg_dict: dict, launch_hi: float, grid_hi: float, n: int):
    """Lifetime orbital cost advantage (terrestrial TCO - orbital TCO, $M) over a
    launch-cost x grid-cost grid. Physics is independent of both axes, so it runs once."""
    local = SystemConfig(**cfg_dict)
    local_phys = phys.run_physics(local)
    capex_total, breakdown = econ.compute_capex(local, local_phys)
    opex_annual, _ = econ.compute_opex(local, breakdown["Hardware"])
    capex_ex_launch = capex_total - breakdown["Launch"]
    launches = np.linspace(0.0, launch_hi, n)
    grids = np.linspace(0.0, grid_hi, n)
    z = np.zeros((n, n))
    for i, g in enumerate(grids):
        _, terr = econ.terrestrial_baseline(replace(local, grid_cost_per_kWh=g), local_phys.total_electrical_load_kW)
        orbital = capex_ex_launch + launches * local_phys.total_dry_mass_kg + opex_annual * local.lifespan_years
        z[i, :] = (terr - orbital) / 1e6
    return launches, grids, z


@st.cache_data(show_spinner=False)
def compute_tornado(cfg_dict: dict, field_keys: tuple, delta: float):
    """For each field, return the post-tax NPV when it's set to value*(1+/-delta).

    Physics runs once per perturbation since several fields (e.g. rack_unit_cost,
    panel_efficiency) feed into both mass/area and cost.
    """
    local = SystemConfig(**cfg_dict)
    base = econ.run_economics(local, phys.run_physics(local)).npv_posttax
    rows = []
    for name in field_keys:
        cur = float(getattr(local, name))
        lo_cfg = replace(local, **{name: cur * (1.0 - delta)})
        hi_cfg = replace(local, **{name: cur * (1.0 + delta)})
        lo_npv = econ.run_economics(lo_cfg, phys.run_physics(lo_cfg)).npv_posttax
        hi_npv = econ.run_economics(hi_cfg, phys.run_physics(hi_cfg)).npv_posttax
        rows.append({"field": name, "low": lo_npv, "high": hi_npv, "swing": abs(hi_npv - lo_npv)})
    rows.sort(key=lambda r: r["swing"], reverse=True)
    return base, rows


def tornado_fig() -> go.Figure:
    field_keys = tuple(k for k, _ in TORNADO_FIELDS)
    label_map = dict(TORNADO_FIELDS)
    base_npv, rows = compute_tornado(dict(cfg.__dict__), field_keys, 0.20)
    base_m = base_npv * CURRENCY_FACTOR / 1e6
    fig = go.Figure()
    for r in reversed(rows):  # render highest-swing on top of the y-axis
        lo = min(r["low"], r["high"]) * CURRENCY_FACTOR / 1e6
        hi = max(r["low"], r["high"]) * CURRENCY_FACTOR / 1e6
        fig.add_trace(
            go.Bar(
                y=[label_map[r["field"]]], x=[hi - lo], base=lo,
                orientation="h", marker_color=PAL["accent"],
                customdata=[[r["low"] * CURRENCY_FACTOR / 1e6, r["high"] * CURRENCY_FACTOR / 1e6]],
                hovertemplate=(
                    f"<b>{label_map[r['field']]}</b><br>"
                    f"Input 20% lower: {CURRENCY_SYMBOL}%{{customdata[0]:.1f}}M<br>"
                    f"Input 20% higher: {CURRENCY_SYMBOL}%{{customdata[1]:.1f}}M"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )
    fig.add_vline(
        x=base_m, line_dash="dash", line_color=PAL["muted"],
        annotation_text=f"baseline {CURRENCY_SYMBOL}{base_m:,.1f}M",
        annotation_position="top",
    )
    fig.update_layout(
        title="Cost-leverage tornado · each input ±20% (post-tax NPV)",
        xaxis_title=f"Post-tax NPV ({CURRENCY_SYMBOL}M)",
        barmode="overlay", margin=dict(l=240, r=40, t=80, b=56),
    )
    return apply_theme(fig, MODE, height=480)


def sensitivity_fig() -> go.Figure:
    launch_hi = max(cfg.launch_cost_per_kg * 2, 3000)
    grid_hi = max(cfg.grid_cost_per_kWh * 2, 0.20)
    launches, grids, z = advantage_grid(dict(cfg.__dict__), launch_hi, grid_hi, 40)
    z_display = z * CURRENCY_FACTOR
    fig = go.Figure(
        go.Heatmap(
            x=launches, y=grids, z=z_display, colorscale=DIVERGING, zmid=0,
            colorbar=dict(title=f"{CURRENCY_SYMBOL}M"),
            hovertemplate=(
                "Launch $%{x:.0f}/kg, grid $%{y:.3f}/kWh<br>"
                f"advantage {CURRENCY_SYMBOL}%{{z:.0f}}M<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[cfg.launch_cost_per_kg], y=[cfg.grid_cost_per_kWh], mode="markers+text",
            marker=dict(color=PAL["ink"], size=12, symbol="x"),
            text=["current"], textposition="top center",
            textfont=dict(color=PAL["ink"]), showlegend=False, hoverinfo="skip",
        )
    )
    fig.update_layout(
        title="Orbital cost advantage vs terrestrial (green = orbital cheaper)",
        xaxis_title="Launch cost ($/kg)", yaxis_title="Grid power ($/kWh)",
    )
    return apply_theme(fig, MODE, height=440)


tax_choice = st.radio("Cash-flow basis", ["Post-tax", "Pre-tax"], horizontal=True, key="cf_basis")
dollar_choice = st.radio("Dollars", ["Nominal", "Real"], horizontal=True, key="cf_dollars")
cf_col = f"cum_{'posttax' if tax_choice == 'Post-tax' else 'pretax'}_{'nominal' if dollar_choice == 'Nominal' else 'real'}"
cf_label = f"{tax_choice}, {dollar_choice.lower()}"

left, right = st.columns([3, 2])
with left:
    st.plotly_chart(cashflow_fig(cf_col, cf_label), width="stretch", theme=None)
with right:
    st.plotly_chart(capex_fig(), width="stretch", theme=None)

st.plotly_chart(sensitivity_fig(), width="stretch", theme=None)

st.plotly_chart(tornado_fig(), width="stretch", theme=None)
st.caption(
    "Each input was independently perturbed ±20% from the configured baseline; the bar "
    "shows the resulting post-tax NPV range. Longest bars are the highest-leverage "
    "components to optimize first."
)


# --------------------------------------------------------------------------- financial detail
st.subheader("Financial detail")
detail = pd.DataFrame(
    {
        "Basis": ["Pre-tax", "Post-tax"],
        "NPV": [fmt_money(economics.npv_pretax), fmt_money(economics.npv_posttax)],
        "IRR (nominal)": [fmt_pct(economics.irr_pretax), fmt_pct(economics.irr_posttax)],
        "IRR (real)": [fmt_pct(economics.irr_pretax_real), fmt_pct(economics.irr_posttax_real)],
    }
)
st.dataframe(detail, hide_index=True, width="stretch")
st.caption(
    f"NPV is identical in real and nominal terms under consistent discounting (real "
    f"discount rate {fmt_pct(economics.real_discount_rate)}); inflation's effect appears "
    f"through taxes and the cash-flow trajectory. Total tax paid over the life: "
    f"{fmt_money(economics.total_tax)}."
)

with st.expander("Tax & depreciation schedule"):
    s = economics.cashflow_df
    schedule = pd.DataFrame(
        {
            "Year": s["year"].astype(int),
            f"Revenue ({CURRENCY_SYMBOL}M)": (s["nominal_revenue"] * CURRENCY_FACTOR / 1e6).round(2),
            f"OpEx ({CURRENCY_SYMBOL}M)": (s["nominal_opex"] * CURRENCY_FACTOR / 1e6).round(2),
            f"Depreciation ({CURRENCY_SYMBOL}M)": (s["depreciation"] * CURRENCY_FACTOR / 1e6).round(2),
            f"Taxable ({CURRENCY_SYMBOL}M)": (s["taxable_income"] * CURRENCY_FACTOR / 1e6).round(2),
            f"Tax ({CURRENCY_SYMBOL}M)": (s["tax_paid"] * CURRENCY_FACTOR / 1e6).round(2),
            f"Loss c/f ({CURRENCY_SYMBOL}M)": (s["loss_carryforward"] * CURRENCY_FACTOR / 1e6).round(2),
        }
    )
    st.dataframe(schedule, hide_index=True, width="stretch")
    scheme_label = (
        f"{cfg.depreciation_scheme}"
        + (f", {cfg.depreciation_years_sln} yr" if cfg.depreciation_scheme == "SLN" else "")
        + (f", {cfg.depreciation_rate_db:.0%}/yr" if cfg.depreciation_scheme == "DB" else "")
        + (f", {cfg.macrs_years}-yr class" if cfg.depreciation_scheme == "MACRS" else "")
        if cfg.overwrite_depreciation
        else f"straight-line over {cfg.lifespan_years} yr"
    )
    st.caption(f"Depreciation: {scheme_label} on a {fmt_money(economics.depreciation_basis)} basis.")


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
c3.metric("Orbital $/GPU-hr", fmt_rate(orbital_per_hr))
c4.metric("Terrestrial $/GPU-hr", fmt_rate(terr_per_hr))

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
    val = economics.crossover_launch_cost * CURRENCY_FACTOR
    cross.append(f"orbital wins below **{CURRENCY_SYMBOL}{val:,.0f}/kg** launch cost")
if economics.crossover_power_cost is not None:
    val = economics.crossover_power_cost * CURRENCY_FACTOR
    cross.append(f"orbital wins above **{CURRENCY_SYMBOL}{val:,.3f}/kWh** terrestrial grid power")
if cross:
    st.caption("Crossover points (holding everything else fixed): " + "; ".join(cross) + ".")
