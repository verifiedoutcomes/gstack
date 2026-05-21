# Orbital Data Center — Techno-Economic Model

An interactive techno-economic model (TEM) for **data centers in orbit**. It combines
first-principles spacecraft physics (solar generation, radiative cooling, mass) with a
full discounted-cash-flow economic model (CapEx, OpEx, revenue, NPV/IRR/payback) and
benchmarks the result against an **equal-power terrestrial data center**.

Built with Streamlit, NumPy/Pandas, Plotly and numpy-financial.

## Quick start

```bash
cd orbital-dc-tem
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## What it models

**Physics (`physics_engine.py`)**

- **Power balance.** Essentially 100% of the electrical power drawn by the compute
  hardware becomes waste heat, so the radiator must reject the *same* power the solar
  array generates (plus a tunable parasitic overhead for comms/attitude/thermal loops).
  Solar and thermal sizing are coupled, not independent.
- **Solar generation.** Dawn-dusk Sun-synchronous orbit gives near-continuous sunlight at
  the solar constant (1361 W/m², ~36% above Earth's surface). Array area =
  `load / (efficiency × 1361 × illumination)`.
- **Radiative cooling (Stefan-Boltzmann).** `q = ε·σ·(T⁴ − T_sink⁴)`. Radiators are
  modeled as two-sided deployable panels (configurable), sized at a 318 K (45 °C) surface
  cap. The model flags thermal *survivability* failures when no finite radiator could
  reject the heat.
- **Mass.** Compute + solar + radiator mass, scaled up by a structural/bus multiplier;
  total dry mass drives launch cost.

**Economics (`economic_engine.py`)**

- **CapEx:** launch (dry mass × $/kg), space-qualified hardware, solar and radiator
  manufacturing, and R&D + integration overhead.
- **OpEx (annual):** ground-station downlink, station-keeping, insurance (annual % of
  hardware value), and ops/management.
- **Revenue:** `$/GPU-hour × GPUs × utilization × 8760`, or a `$/FLOP` mode, with optional
  annual degradation.
- **Metrics:** NPV and IRR (via numpy-financial, with graceful "N/A" when no real IRR
  root exists) and payback period in months (interpolated on a monthly cash-flow series).
- **Terrestrial comparison:** equal-power ground data center (energy × PUE × grid price,
  plus land/water/ops and amortized build cost), with the **crossover** launch cost and
  grid power price at which orbital total cost of ownership matches terrestrial.

## Case-study presets (`presets.py`)

Each preset is modeled **per satellite** — one tile of a larger constellation. Fleet
economics scale by the number of satellites.

| Preset | Idea | Headline |
|---|---|---|
| **Google Project Suncatcher** | Hyperscale TPU constellation, ~650 km SSO, Starship-class launch ($200/kg) | Marginal — nearly viable on cheap launch |
| **Starcloud** | SmallSat H100/B200 clusters, premium sovereign/edge billing, $1500/kg launch | Profitable on premium pricing |
| **First-Principles Rack-Sat** | 1 satellite = 1–3 racks + massive ~350 ft radiators | Thermal-extreme, not yet economic |

> **Scale note.** A single full-fleet hyperscale node (tens of MW) would require a
> multi-kilometre radiator as one panel, which is unphysical — hence the per-satellite
> framing. The app surfaces an advisory note if a configured radiator gets implausibly long.

## Architecture

`physics_engine.py` and `economic_engine.py` are **pure** — they import only NumPy/Pandas/
numpy-financial and the local `models`/`constants`, never Streamlit — so they're fully
unit-testable. `app.py` is the only Streamlit module; `theme.py` holds the shared visual
system (Plotly template, palette, CSS).

```
app.py            Streamlit UI (sidebar, KPIs, charts, terrestrial card)
physics_engine.py pure physics  -> PhysicsResult
economic_engine.py pure economics -> EconomicsResult
models.py         dataclasses + validate_config
presets.py        three case-study configs
constants.py      physical constants
theme.py          visual system
tests/            pytest suite for the engines
```

## Testing

```bash
./.venv/bin/python -m pytest tests/ -q
```

The suite covers the physics (Stefan-Boltzmann flux, solar/radiator sizing, two-sided
halving, mass monotonicity, survivability), the financials against numpy-financial
ground truth (NPV, IRR, payback interpolation), and that every preset loads, stays
thermally survivable, and runs the full pipeline.
