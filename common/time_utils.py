"""
Low-level time and solar-geometry helpers shared by config.py (RAAN /
local-solar-time targeting) and sgp4_propagation.py (TEME->ECEF rotation).

Kept dependency-free of config.py on purpose: config.py needs these to
compute GAIA_RAAN_0_DEG at import time, and sgp4_propagation.py imports
config.py for physical constants, so these two couldn't both import a
version of this code that lived in either of those files without a
circular import.
"""

import numpy as np
from sgp4.api import jday


def jd_fr(dt):
    """UTC datetime -> (jd, fr) Julian date split, as sgp4 expects."""
    return jday(
        dt.year, dt.month, dt.day,
        dt.hour, dt.minute, dt.second + dt.microsecond * 1e-6,
    )


def gmst_rad(jd, fr):
    """
    Greenwich Mean Sidereal Time (IAU-82 / Vallado formula), radians.

    Materially more accurate than a linear
    `GMST_AT_EPOCH_DEG + EARTH_ROTATION_RATE * t` approximation, since it's
    referenced to the actual UT1 Julian date rather than an assumed
    constant sidereal rate from an arbitrary GMST=0 convention.
    """
    jd_ut1 = jd + fr
    T = (jd_ut1 - 2451545.0) / 36525.0

    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T ** 2
        - 6.2e-6 * T ** 3
    )

    gmst_deg = (gmst_sec % 86400.0) / 240.0  # 240 s of time == 1 deg
    return np.deg2rad(gmst_deg % 360.0)


def solar_ra_dec_deg(jd, fr):
    """
    Low-precision solar right ascension / declination (deg), good to
    about 0.01 deg — from the Astronomical Almanac's low-precision solar
    coordinates formula (equivalent to Meeus ch. 25's low-accuracy
    method). Plenty for local-solar-time targeting, where 0.25 deg of
    longitude error is only ~1 minute of time.
    """
    jd_tt = jd + fr  # UT1/TT distinction (~tens of ms) is negligible here
    n = jd_tt - 2451545.0

    L = np.deg2rad((280.460 + 0.9856474 * n) % 360.0)   # mean longitude
    g = np.deg2rad((357.528 + 0.9856003 * n) % 360.0)   # mean anomaly

    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)
    eps = np.deg2rad(23.439 - 0.0000004 * n)             # obliquity

    ra = np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))
    dec = np.arcsin(np.sin(eps) * np.sin(lam))

    return np.rad2deg(ra) % 360.0, np.rad2deg(dec)


def subsolar_longitude_deg(jd, fr):
    """
    Earth-fixed longitude directly under the sun at this instant (deg,
    -180..180). This is what makes a "10:30 local solar time" ground-track
    crossing target meaningful for a real calendar date, instead of the
    old fixed convention of assuming the sun sits at 0 deg longitude at
    epoch.
    """
    gmst_deg = np.rad2deg(gmst_rad(jd, fr))
    ra_deg, _dec_deg = solar_ra_dec_deg(jd, fr)

    lon = (gmst_deg - ra_deg + 180.0) % 360.0 - 180.0
    return lon