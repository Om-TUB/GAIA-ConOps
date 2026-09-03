"""
SGP4-based orbit propagation.

Replaces the hand-rolled two-body-plus-J2-secular propagator in
orbit_propagation.py with the standard SGP4 analytic theory (Hoots &
Roehrich, as maintained in Brandon Rhodes' `sgp4` package). SGP4 folds in
zonal harmonics through J4, luni-solar-like secular drag via BSTAR, and
several periodic correction terms that the old J2-only model didn't
capture -- material over a 24h simulation window, especially for the
along-track / ground-track-crossing-time claims this repo checks.

Two different situations are handled here, and they are NOT equally
trustworthy:

1. H2Sat (Heinrich Hertz, NORAD 57213) is a real, currently-operating
   satellite. We propagate its actual tracked TLE (see config.py for
   source/epoch). This is the normal, fully-supported way to use SGP4
   and is about as accurate as this class of model gets.

2. GAIA-A / GAIA-B are notional CubeSats that haven't flown. There is no
   fitted TLE for them, so we seed SGP4 directly from the mission's
   computed osculating Keplerian elements via `Satrec.sgp4init()`
   (bypassing TLE text entirely). This is a reasonable way to get a
   more accurate propagation than plain two-body + J2 -- SGP4 still adds
   drag and higher-order zonal terms -- but it is an approximation: SGP4
   is formally defined over *mean* elements recovered from a real
   fitted TLE, and we're substituting osculating elements at epoch
   instead. Treat GAIA-A/B results as "better than before," not as
   flight-truth. Re-seed from a real TLE once the satellites launch and
   are catalogued.
"""

import numpy as np
from sgp4.api import Satrec, WGS72

import CONOPs.conops_config as cfg
from common.time_utils import jd_fr, gmst_rad

# Re-exported for backwards compatibility with anything importing these
# names from this module.
_jd_fr = jd_fr


# ---------------------------------------------------------------------------
# Building Satrec objects
# ---------------------------------------------------------------------------

def satrec_from_tle(line1, line2):
    """Wrap a real, tracked two-line element set (e.g. H2Sat)."""
    return Satrec.twoline2rv(line1, line2)


def keplerian_to_satrec(
    satnum, epoch_dt, a_km, e, i_deg, raan_deg, argp_deg,
    mean_anomaly_deg, bstar=0.0,
):
    """
    Build a Satrec directly from classical elements via sgp4init(),
    for satellites with no real TLE to propagate (see module docstring
    caveat above).
    """
    jd, fr = _jd_fr(epoch_dt)
    epoch_sgp4 = (jd - 2433281.5) + fr  # days since 1949-12-31 00:00 UT

    n_rad_min = np.sqrt(cfg.MU_EARTH / a_km ** 3) * 60.0  # rad/s -> rad/min

    satrec = Satrec()
    satrec.sgp4init(
        WGS72,
        "i",                        # 'improved' (post-2000 AFSPC) mode
        satnum,
        epoch_sgp4,
        bstar,
        0.0,                        # ndot (unused by SGP4, TLE legacy field)
        0.0,                        # nddot (unused by SGP4, TLE legacy field)
        e,
        np.deg2rad(argp_deg),
        np.deg2rad(i_deg),
        np.deg2rad(mean_anomaly_deg),
        n_rad_min,
        np.deg2rad(raan_deg),
    )
    return satrec


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

def propagate_sgp4(satrec, epoch_dt, t_seconds):
    """
    Propagate `satrec` across t_seconds (relative to epoch_dt).

    Returns the same dict shape as orbit_propagation.propagate_orbit(),
    so this is a drop-in replacement wherever that was called:
        t_s, r_eci_km (TEME, see note below), lat_deg, lon_deg, alt_km,
        radius_km

    Note on r_eci_km: SGP4 outputs position in TEME (True Equator, Mean
    Equinox of date) -- a "true of date" frame, not a fixed-epoch
    inertial frame like J2000/GCRF. It's what every TLE-based tool
    uses, but if you need to compare against J2000 vectors elsewhere,
    a frame rotation (TEME->GCRF) is needed on top of this.
    """
    t = np.atleast_1d(np.asarray(t_seconds, dtype=float))
    jd0, fr0 = _jd_fr(epoch_dt)

    jd_arr = np.full(t.shape, jd0)
    fr_arr = fr0 + t / 86400.0

    err, r_teme, v_teme = satrec.sgp4_array(jd_arr, fr_arr)

    if np.any(err):
        bad_codes = sorted(set(err[err != 0]))
        raise RuntimeError(
            f"SGP4 propagation failed (error code(s) {bad_codes}) for "
            f"satnum={satrec.satnum} at {int(np.count_nonzero(err))} of "
            f"{len(err)} timesteps"
        )

    n = len(t)
    lat_deg = np.empty(n)
    lon_deg = np.empty(n)
    alt_km = np.empty(n)
    radius_km = np.empty(n)

    for k in range(n):
        gmst = gmst_rad(jd_arr[k], fr_arr[k])
        cos_g, sin_g = np.cos(gmst), np.sin(gmst)

        x, y, z = r_teme[k]
        x_ecef = cos_g * x + sin_g * y
        y_ecef = -sin_g * x + cos_g * y
        z_ecef = z

        r = np.sqrt(x_ecef ** 2 + y_ecef ** 2 + z_ecef ** 2)

        radius_km[k] = r
        lat_deg[k] = np.rad2deg(np.arcsin(np.clip(z_ecef / r, -1.0, 1.0)))
        lon_deg[k] = np.rad2deg(np.arctan2(y_ecef, x_ecef))
        alt_km[k] = r - cfg.R_EARTH

    return dict(
        t_s=t,
        r_eci_km=r_teme,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_km=alt_km,
        radius_km=radius_km,
    )