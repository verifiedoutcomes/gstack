"""Shared visual system: one restrained accent, generous whitespace, no chartjunk.

Centralizes the Plotly template, palette, fonts and the Streamlit CSS so every figure
and the page chrome stay consistent. Imported only by ``app.py`` -- the pure engines
never touch presentation.
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

FONT_FAMILY = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"

PALETTE = {
    "ink": "#0A0A0A",       # primary text / strong lines
    "muted": "#6B7280",     # secondary text / axis labels
    "grid": "#EEEEEE",      # faint gridlines
    "paper": "#FFFFFF",     # page / plot background
    "accent": "#FF5C35",    # the one accent: orbital / primary series
    "accent_soft": "#FFE3DA",  # accent tint for fills / highlights
    "neutral": "#9CA3AF",   # terrestrial / comparison series
    "neutral_soft": "#E5E7EB",
    "positive": "#1F9D72",  # break-even reached / profit
    "negative": "#D1453B",  # underwater / loss
}

# CapEx stack: tints of the accent (light -> dark), never rainbow.
CAPEX_COLORS = ["#FF8A6B", "#FF5C35", "#E8431F", "#B8330F", "#7A2008"]

# Diverging scale for the NPV sensitivity map; pair with zmid=0 so break-even sits at the pivot.
DIVERGING_NPV = [
    [0.0, "#D1453B"],
    [0.5, "#F4EDE9"],
    [1.0, "#1F9D72"],
]


def register_theme() -> None:
    """Register and activate the 'odc' Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, size=14, color=PALETTE["ink"]),
        paper_bgcolor=PALETTE["paper"],
        plot_bgcolor=PALETTE["paper"],
        colorway=[PALETTE["accent"], PALETTE["neutral"], PALETTE["ink"], PALETTE["positive"]],
        title=dict(font=dict(size=18, color=PALETTE["ink"]), x=0.0, xanchor="left"),
        margin=dict(l=56, r=24, t=56, b=48),
        xaxis=dict(
            showgrid=False, zeroline=False, showline=True, linecolor=PALETTE["grid"],
            ticks="outside", tickcolor=PALETTE["grid"], tickfont=dict(color=PALETTE["muted"], size=12),
            title=dict(font=dict(color=PALETTE["muted"], size=13)),
        ),
        yaxis=dict(
            showgrid=True, gridcolor=PALETTE["grid"], zeroline=False, showline=False,
            tickfont=dict(color=PALETTE["muted"], size=12),
            title=dict(font=dict(color=PALETTE["muted"], size=13)),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
            bgcolor="rgba(0,0,0,0)", font=dict(color=PALETTE["muted"], size=12),
        ),
        hoverlabel=dict(bgcolor=PALETTE["ink"], font=dict(family=FONT_FAMILY, color="#FFFFFF", size=12)),
    )
    pio.templates["odc"] = template
    pio.templates.default = "odc"


def apply_theme(fig: go.Figure, height: int = 360) -> go.Figure:
    """Apply consistent sizing/margins/hover to a finished figure."""
    fig.update_layout(template="odc", height=height, hovermode="x unified")
    return fig


CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {{
    font-family: {FONT_FAMILY};
    color: {PALETTE["ink"]};
}}
.stApp {{ background: {PALETTE["paper"]}; }}

/* Tighter, card-like KPI metrics */
div[data-testid="stMetric"] {{
    background: #FAFAFA;
    border: 1px solid {PALETTE["grid"]};
    border-radius: 12px;
    padding: 16px 18px;
}}
div[data-testid="stMetricLabel"] {{
    color: {PALETTE["muted"]};
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
div[data-testid="stMetricValue"] {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {PALETTE["ink"]};
}}

h1, h2, h3 {{ font-weight: 700; letter-spacing: -0.01em; }}
.block-container {{ padding-top: 2.2rem; max-width: 1280px; }}

/* Quiet the default Streamlit chrome */
#MainMenu, footer {{ visibility: hidden; }}
[data-testid="stSidebar"] {{ background: #FAFAFA; border-right: 1px solid {PALETTE["grid"]}; }}

/* Comparison card */
.odc-card {{
    background: #FAFAFA;
    border: 1px solid {PALETTE["grid"]};
    border-radius: 14px;
    padding: 22px 24px;
}}
.odc-card h4 {{ margin: 0 0 4px 0; font-size: 0.8rem; color: {PALETTE["muted"]};
    text-transform: uppercase; letter-spacing: 0.04em; }}
.odc-verdict-pos {{ color: {PALETTE["positive"]}; font-weight: 700; }}
.odc-verdict-neg {{ color: {PALETTE["negative"]}; font-weight: 700; }}
</style>
"""
