"""Physical constants and unit conversions for the orbital data center model.

Only true physical/universal constants live here. Engineering assumptions that a
user might reasonably want to tune (panel efficiency, areal densities, costs) are
fields on ``SystemConfig`` in ``models.py`` instead.
"""

# --- Universal physical constants ---
SIGMA: float = 5.67e-8
"""Stefan-Boltzmann constant, W / (m^2 . K^4)."""

G_ORBIT: float = 1361.0
"""Solar irradiance above the atmosphere (the solar constant), W/m^2.

About 36% higher than the ~1000 W/m^2 peak available at Earth's surface, with no
atmospheric or cloud attenuation.
"""

TERRESTRIAL_PEAK_IRRADIANCE: float = 1000.0
"""Reference peak irradiance at Earth's surface (1 sun), W/m^2 -- for context only."""

# --- Time / unit conversions ---
HOURS_PER_YEAR: float = 8760.0
MONTHS_PER_YEAR: int = 12
FT_PER_M: float = 3.28084
WATTS_PER_KW: float = 1000.0
KW_PER_MW: float = 1000.0
