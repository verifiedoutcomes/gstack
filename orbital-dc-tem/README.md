# Orbital Data Center — Techno-Economic Model

An interactive techno-economic model (TEM) for **data centers in orbit**. It combines
first-principles spacecraft physics (solar generation, radiative cooling, mass) with a
full discounted-cash-flow economic model (CapEx, OpEx, revenue, NPV/IRR/payback) and
benchmarks the result against an **equal-power terrestrial data center**.

Built with Streamlit, NumPy/Pandas, Plotly and numpy-financial. Ships with a
light/dark mode toggle and a clean, minimal visual system.

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
  `load / (efficiency × 1361 × illumination)`. Default solar mass uses **3.5 kg/m²**,
  anchored to NASA's flight-proven Roll-Out Solar Array (ROSA: ~325 kg per 82 m² = 3.95
  kg/m² with the composite booms that double as structure and actuator -- no motors).
- **Radiative cooling (Stefan-Boltzmann + environmental load).**
  `q_net = ε·(σ·(T⁴ − T_sink⁴) − environmental_load)`. Radiators in LEO absorb incident
  Earth IR (~240 W/m² at an effective 255 K) plus albedo; by Kirchhoff's law absorbed and
  emitted IR share the same emissivity, so the absorbed load is `ε × env_load`. At 318 K
  with ε=0.9 and the default 250 W/m² LEO env load, the model recovers **~297 W/m²
  net single-face** — squarely inside the [100–350 W/m² range NASA reports for real
  spacecraft radiators](https://www.nasa.gov/smallsat-institute/sst-soa/thermal-control/).
  Set `environmental_thermal_load_W_m2 = 0` for the deep-space ideal (522 W/m²).
- **Mass.** Compute + solar + radiator mass, scaled up by a structural/bus multiplier;
  total dry mass drives launch cost.

**Economics (`economic_engine.py`)**

- **CapEx:** launch (dry mass × $/kg), space-qualified hardware, solar and radiator
  manufacturing, and R&D + integration overhead.
- **OpEx (annual):** ground-station downlink, station-keeping, insurance (annual % of
  hardware value), and ops/management.
- **Revenue:** `$/GPU-hour × GPUs × utilization × 8760`, or a `$/FLOP` mode, with optional
  annual degradation.
- **Tax, depreciation & inflation:** a custom tax rate with optional loss carry-forward;
  depreciation by straight-line-over-life (default) or an overridden scheme — SLN over a
  chosen recovery period, declining balance, or a MACRS class (3/5/7/10/15/20 yr); and
  inflation from a base year. Results are reported pre- and post-tax in both real and
  nominal dollars. (Depreciation is non-cash and front/back-loads the tax shield;
  inflation's bite shows up through tax, since depreciation is a fixed nominal deduction.)
- **Metrics:** pre- and post-tax NPV (inflation-invariant under consistent discounting),
  nominal and real IRR (via numpy-financial, with graceful "N/A" when no real root
  exists), and an interpolated payback period.
- **Terrestrial comparison:** equal-power ground data center (energy × PUE × grid price,
  plus land/water/ops and amortized build cost), with the **crossover** launch cost and
  grid power price at which orbital total cost of ownership matches terrestrial.
- **Cost-leverage tornado:** every key cost input is independently perturbed ±20% from
  the active baseline; a horizontal bar chart ranks them by resulting post-tax NPV swing,
  so the longest bars are the highest-leverage optimizations.

## Case-study presets (`presets.py`)

Each preset is modeled **per satellite** — one tile of a larger constellation. Fleet
economics scale by the number of satellites.

| Preset | Idea | Headline |
|---|---|---|
| **Google Project Suncatcher** | Hyperscale TPU constellation, ~650 km SSO, Starship-class launch ($200/kg) | Marginal — nearly viable on cheap launch |
| **Starcloud** | SmallSat H100/B200 clusters, premium sovereign/edge billing, $1500/kg launch | Profitable on premium pricing |
| **First-Principles Rack-Sat** | 1 satellite = 1–3 racks + massive ~350 ft radiators | Thermal-extreme, not yet economic |
| **NVIDIA GB200 NVL72** | One satellite = one NVL72 reference rack (72 B200 + 36 Grace, 132 kW, 1.36 t, ~$2.5M, 1.44 ExaFLOPS FP4) | Unit economics for the current AI flagship rack |

Default rack mass and power are anchored to NVIDIA's published GB200 NVL72 reference
(1.36 t, ~132 kW). Sources: [NVIDIA](https://www.nvidia.com/en-us/data-center/gb200-nvl72/),
[SemiAnalysis BOM](https://newsletter.semianalysis.com/p/gb200-hardware-architecture-and-component),
[Introl deployment guide](https://introl.com/blog/gb200-nvl72-deployment-72-gpu-liquid-cooled).

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
presets.py        four case-study configs incl. NVIDIA GB200 NVL72
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
ground truth (NPV, IRR, payback interpolation), depreciation schemes (each totals the
basis; MACRS tables sum to 1; declining balance front-loads), tax loss carry-forward,
inflation, and that every preset loads, stays thermally survivable, and runs the full
pipeline. A headless `AppTest` smoke test renders the app (and switches presets) without
a browser.
