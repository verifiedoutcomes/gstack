"""Shared visual system with light and dark modes.

One restrained accent, generous whitespace, no chartjunk. Centralizes the Plotly
templates, palettes, fonts and the Streamlit CSS so every figure and the page chrome
stay consistent and switch together. Imported only by ``app.py`` -- the pure engines
never touch presentation.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

PALETTE_LIGHT = {
    "ink": "#0A0A0A",
    "muted": "#6B7280",
    "grid": "#EEEEEE",
    "paper": "#FFFFFF",
    "sidebar": "#FAFAFA",
    "card": "#FAFAFA",
    "input": "#FFFFFF",
    "border": "#E5E7EB",
    "accent": "#FF5C35",
    "neutral": "#9CA3AF",
    "positive": "#1F9D72",
    "negative": "#D1453B",
}

PALETTE_DARK = {
    "ink": "#F5F5F7",
    "muted": "#9CA3AF",
    "grid": "#2A2A2E",
    "paper": "#0E0E11",
    "sidebar": "#131317",
    "card": "#17171C",
    "input": "#1C1C22",
    "border": "#2A2A2E",
    "accent": "#FF6F4D",
    "neutral": "#6B7280",
    "positive": "#34D399",
    "negative": "#F87171",
}

_CAPEX_LIGHT = ["#FF8A6B", "#FF5C35", "#E8431F", "#B8330F", "#7A2008"]
_CAPEX_DARK = ["#FFC2AE", "#FF9E82", "#FF6F4D", "#E8552D", "#B8451F"]


def get_palette(mode: str) -> dict[str, str]:
    return PALETTE_DARK if mode == "dark" else PALETTE_LIGHT


def get_capex_colors(mode: str) -> list[str]:
    return _CAPEX_DARK if mode == "dark" else _CAPEX_LIGHT


def get_diverging(mode: str) -> list:
    """Diverging colorscale for the advantage map; pair with zmid=0 (negative -> positive)."""
    pal = get_palette(mode)
    midpoint = "#2A2A2E" if mode == "dark" else "#F4EDE9"
    return [[0.0, pal["negative"]], [0.5, midpoint], [1.0, pal["positive"]]]


def _template_for(mode: str) -> go.layout.Template:
    pal = get_palette(mode)
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=14, color=pal["ink"]),
        paper_bgcolor=pal["paper"],
        plot_bgcolor=pal["paper"],
        colorway=[pal["accent"], pal["neutral"], pal["ink"], pal["positive"]],
        title=dict(font=dict(size=18, color=pal["ink"]), x=0.0, xanchor="left"),
        margin=dict(l=56, r=24, t=56, b=48),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True, linecolor=pal["grid"],
            ticks="outside", tickcolor=pal["grid"], tickfont=dict(color=pal["muted"], size=12),
            title=dict(font=dict(color=pal["muted"], size=13)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=pal["grid"], zeroline=False, showline=False,
            tickfont=dict(color=pal["muted"], size=12),
            title=dict(font=dict(color=pal["muted"], size=13)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=pal["muted"], size=12),
        ),
        hoverlabel=dict(
            bgcolor=pal["ink"], font=dict(family=FONT_FAMILY, color=pal["paper"], size=12)
        ),
    )
    return template


def register_themes() -> None:
    """Register the 'odc_light' and 'odc_dark' Plotly templates."""
    pio.templates["odc_light"] = _template_for("light")
    pio.templates["odc_dark"] = _template_for("dark")


def apply_theme(fig: go.Figure, mode: str = "light", height: int = 360) -> go.Figure:
    """Apply the active template plus explicit colors, sizing and hover behavior.

    Colors are set concretely (not just via the template) because Streamlit + Plotly.js
    do not reliably inherit nested template font colors -- explicit layout properties
    take precedence and serialize correctly.
    """
    pal = get_palette(mode)
    fig.update_layout(
        template=f"odc_{mode}",
        height=height,
        hovermode="x unified",
        paper_bgcolor=pal["paper"],
        plot_bgcolor=pal["paper"],
        font=dict(family=FONT_FAMILY, color=pal["ink"]),
        title_font_color=pal["ink"],
        legend_font_color=pal["muted"],
    )
    fig.update_xaxes(title_font_color=pal["muted"], tickfont_color=pal["muted"])
    fig.update_yaxes(title_font_color=pal["muted"], tickfont_color=pal["muted"])
    return fig


def build_css(mode: str) -> str:
    """Streamlit CSS for the active mode: page chrome, sidebar, metric cards, inputs."""
    p = get_palette(mode)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: {p["paper"]}; color: {p["ink"]}; }}
[data-testid="stHeader"] {{ background: {p["paper"]}; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

h1, h2, h3, h4, h5, h6 {{ color: {p["ink"]}; font-weight: 700; letter-spacing: -0.01em; }}
p, span, label, li, .stMarkdown, [data-testid="stCaptionContainer"] {{ color: {p["ink"]}; }}

/* Sidebar */
[data-testid="stSidebar"] {{ background: {p["sidebar"]}; border-right: 1px solid {p["border"]}; }}
[data-testid="stSidebar"] * {{ color: {p["ink"]}; }}

/* KPI metric cards */
div[data-testid="stMetric"] {{
    background: {p["card"]};
    border: 1px solid {p["border"]};
    border-radius: 12px;
    padding: 16px 18px;
}}
div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
    color: {p["muted"]}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
}}
div[data-testid="stMetricValue"] {{ font-size: 1.5rem; font-weight: 700; color: {p["ink"]}; }}

/* Inputs */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] {{
    background: {p["input"]} !important; color: {p["ink"]} !important;
}}
[data-baseweb="input"] input, textarea {{ color: {p["ink"]} !important; }}
[data-testid="stExpander"] {{ border: 1px solid {p["border"]}; border-radius: 10px; }}
[data-testid="stExpander"] summary {{ color: {p["ink"]}; }}

/* Quiet the default Streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}
</style>
"""
