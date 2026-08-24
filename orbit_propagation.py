"""
GAIA CONOPS Simulator — Orbit propagation & visibility geometry
==================================================================
Minimal-dependency (numpy only): two-body Keplerian propagation with J2
secular drift on RAAN and argument of perigee. No skyfield/sgp4/astropy.

This is NOT high-fidelity (no J2 short-period terms, no drag, no lunar/solar
perturbation) — appropriate for a 24h CONOPS visualization, not for
operational orbit determination.
"""

import numpy as np
import config as cfg


# ---------------------------------------------------------------------------
# Low-level orbital mechanics
# ---------------------------------------------------------------------------

def solve_kepler(mean_anomaly_rad, eccentricity, tol=1e-10, max_iter=50):
    """Solve Kepler's equation M = E - e*sin(E) for eccentric anomaly E."""
    M = np.mod(np.asarray(mean_anomaly_rad, dtype=float), 2 * np.pi)
    E = M.copy()

    for _ in range(max_iter):
        dE = (
            E - eccentricity * np.sin(E) - M
        ) / (
            1 - eccentricity * np.cos(E)
        )

        E = E - dE

        if np.all(np.abs(dE) < tol):
            break

    return E


def true_anomaly_from_eccentric(E, e):
    return 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2),
    )


def j2_secular_rates(a_km, e, i_deg):
    """
    Return (raan_dot_rad_s, argp_dot_rad_s) secular drift rates due to J2.
    Standard first-order J2 perturbation formulas.
    """
    i = np.deg2rad(i_deg)

    n = np.sqrt(
        cfg.MU_EARTH / a_km ** 3
    )

    p = a_km * (1 - e ** 2)

    factor = (
        1.5
        * n
        * cfg.J2
        * (cfg.R_EARTH / p) ** 2
    )

    raan_dot = -factor * np.cos(i)

    argp_dot = factor * (
        2 - 2.5 * np.sin(i) ** 2
    )

    return raan_dot, argp_dot


# ---------------------------------------------------------------------------
# Orbit propagation
# ---------------------------------------------------------------------------

def propagate_orbit(elements, t_seconds):
    """
    Propagate a single Keplerian element set to time t_seconds.

    elements:
        a_km
        e
        i_deg
        raan_deg
        argp_deg
        true_anomaly_0_deg

    Returns:
        r_eci_km
        lat_deg
        lon_deg
        alt_km
        radius_km
    """

    a = elements["a_km"]
    e = elements["e"]

    i = np.deg2rad(elements["i_deg"])
    raan0 = np.deg2rad(elements["raan_deg"])
    argp0 = np.deg2rad(elements["argp_deg"])
    nu0 = np.deg2rad(elements["true_anomaly_0_deg"])

    n = np.sqrt(
        cfg.MU_EARTH / a ** 3
    )

    raan_dot, argp_dot = j2_secular_rates(
        a,
        e,
        elements["i_deg"],
    )

    t = np.atleast_1d(
        np.asarray(t_seconds, dtype=float)
    )

    # Initial true anomaly -> eccentric anomaly -> mean anomaly
    E0 = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(nu0 / 2),
        np.sqrt(1 + e) * np.cos(nu0 / 2),
    )

    M0 = E0 - e * np.sin(E0)

    # Propagate mean anomaly
    M = M0 + n * t

    E = solve_kepler(
        M,
        e,
    )

    nu = true_anomaly_from_eccentric(
        E,
        e,
    )

    # J2 secular drift
    raan = raan0 + raan_dot * t
    argp = argp0 + argp_dot * t

    # Orbital radius
    r = a * (
        1 - e * np.cos(E)
    )

    # Argument of latitude
    u = argp + nu

    # ECI coordinates
    cos_O = np.cos(raan)
    sin_O = np.sin(raan)

    cos_u = np.cos(u)
    sin_u = np.sin(u)

    cos_i = np.cos(i)
    sin_i = np.sin(i)

    x_eci = r * (
        cos_O * cos_u
        - sin_O * sin_u * cos_i
    )

    y_eci = r * (
        sin_O * cos_u
        + cos_O * sin_u * cos_i
    )

    z_eci = r * (
        sin_u * sin_i
    )

    # Earth rotation
    gmst_deg = (
        cfg.GMST_AT_EPOCH_DEG
        + cfg.EARTH_ROTATION_RATE * t
    )

    gmst_rad = np.deg2rad(
        gmst_deg
    )

    lon_inertial = np.arctan2(
        y_eci,
        x_eci,
    )

    lon_ecef = (
        lon_inertial
        - gmst_rad
    )

    lon_deg = np.rad2deg(
        np.mod(
            lon_ecef + np.pi,
            2 * np.pi,
        )
        - np.pi
    )

    lat_deg = np.rad2deg(
        np.arcsin(
            np.clip(
                z_eci / r,
                -1.0,
                1.0,
            )
        )
    )

    alt_km = r - cfg.R_EARTH

    return dict(
        t_s=t,
        r_eci_km=np.stack(
            [x_eci, y_eci, z_eci],
            axis=-1,
        ),
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        alt_km=alt_km,
        radius_km=r,
    )


# ---------------------------------------------------------------------------
# GEO satellite
# ---------------------------------------------------------------------------

def geo_fixed_position(lon_deg, t_seconds):
    """
    H2Sat: GEO satellite, station-kept at fixed longitude.
    """

    t = np.atleast_1d(
        np.asarray(t_seconds, dtype=float)
    )

    r = cfg.H2SAT_SEMI_MAJOR_AXIS_KM

    lat_deg = np.zeros_like(t)

    lon_arr = np.full_like(
        t,
        lon_deg,
        dtype=float,
    )

    lat_rad = np.deg2rad(
        lat_deg
    )

    lon_rad = np.deg2rad(
        lon_arr
    )

    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)

    return dict(
        t_s=t,
        r_eci_km=np.stack(
            [x, y, z],
            axis=-1,
        ),
        lat_deg=lat_deg,
        lon_deg=lon_arr,
        alt_km=np.full_like(
            t,
            cfg.H2SAT_ALTITUDE_KM,
        ),
        radius_km=np.full_like(
            t,
            r,
        ),
    )


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def geodetic_to_ecef(lat_deg, lon_deg, alt_km=0.0):
    """
    Convert spherical-Earth geodetic coordinates to ECEF.

    Supports both scalars and arrays.

    Scalar input:
        returns shape (3,)

    Array input:
        returns shape (N, 3)
    """

    lat = np.deg2rad(
        np.asarray(lat_deg, dtype=float)
    )

    lon = np.deg2rad(
        np.asarray(lon_deg, dtype=float)
    )

    alt = np.asarray(
        alt_km,
        dtype=float,
    )

    r = cfg.R_EARTH + alt

    x = (
        r
        * np.cos(lat)
        * np.cos(lon)
    )

    y = (
        r
        * np.cos(lat)
        * np.sin(lon)
    )

    z = (
        r
        * np.sin(lat)
    )

    return np.stack(
        [x, y, z],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Elevation geometry
# ---------------------------------------------------------------------------

def elevation_angle_deg(
    observer_lat_deg,
    observer_lon_deg,
    observer_alt_km,
    sat_lat_deg,
    sat_lon_deg,
    sat_alt_km,
):
    """
    Elevation angle of a satellite as seen from an observer.

    Parameters
    ----------
    observer_lat_deg:
        Observer latitude.

    observer_lon_deg:
        Observer longitude.

    observer_alt_km:
        Observer altitude.

    sat_lat_deg:
        Target satellite latitude.

    sat_lon_deg:
        Target satellite longitude.

    sat_alt_km:
        Target satellite altitude.

    Both scalar and array inputs are supported.

    For array inputs the resulting ECEF coordinates have shape:

        (N, 3)

    where N is the number of time steps.
    """

    obs_ecef = geodetic_to_ecef(
        observer_lat_deg,
        observer_lon_deg,
        observer_alt_km,
    )

    sat_ecef = geodetic_to_ecef(
        sat_lat_deg,
        sat_lon_deg,
        sat_alt_km,
    )

    obs_ecef = np.asarray(
        obs_ecef,
        dtype=float,
    )

    sat_ecef = np.asarray(
        sat_ecef,
        dtype=float,
    )

    # Scalar -> (1, 3)
    if obs_ecef.ndim == 1:
        obs_ecef = obs_ecef[np.newaxis, :]

    if sat_ecef.ndim == 1:
        sat_ecef = sat_ecef[np.newaxis, :]

    # Broadcast observer and target across time.
    obs_ecef, sat_ecef = np.broadcast_arrays(
        obs_ecef,
        sat_ecef,
    )

    # Line of sight
    los = sat_ecef - obs_ecef

    los_norm = los / np.linalg.norm(
        los,
        axis=-1,
        keepdims=True,
    )

    # Local zenith
    zenith = obs_ecef / np.linalg.norm(
        obs_ecef,
        axis=-1,
        keepdims=True,
    )

    # Angle between LOS and local zenith
    sin_el = np.sum(
        los_norm * zenith,
        axis=-1,
    )

    el_deg = np.rad2deg(
        np.arcsin(
            np.clip(
                sin_el,
                -1.0,
                1.0,
            )
        )
    )

    return el_deg


# ---------------------------------------------------------------------------
# LEO -> H2Sat visibility
# ---------------------------------------------------------------------------

def leo_sees_geo(
    leo_lat_deg,
    leo_lon_deg,
    leo_alt_km,
    geo_lon_deg,
):
    """
    Visibility of H2Sat from a LEO satellite.

    The LEO satellite is the observer.

    H2Sat is the target.
    """

    el = elevation_angle_deg(
        leo_lat_deg,
        leo_lon_deg,
        leo_alt_km,
        0.0,
        geo_lon_deg,
        cfg.H2SAT_ALTITUDE_KM,
    )

    return el > 0.0


# ---------------------------------------------------------------------------
# Full mission timeline generation
# ---------------------------------------------------------------------------

def generate_timeline(
    duration_s=cfg.SIM_DURATION_S,
    step_s=cfg.SIM_TIMESTEP_S,
):
    """
    Propagate all satellites over the full simulation window and compute
    visibility + mode at every timestep.

    Returns:

        {
            "t_s": array,
            "sats": {
                satellite_name: {
                    "lat_deg": ...,
                    "lon_deg": ...,
                    "alt_km": ...,
                    "visibility": ...,
                    "sees_h2sat": ...,
                    "mode": ...
                }
            },
            "h2sat": ...
        }
    """

    t = np.arange(
        0,
        duration_s,
        step_s,
    )

    h2sat_track = geo_fixed_position(
        cfg.H2SAT_LON_DEG,
        t,
    )

    results = {
        "t_s": t,
        "sats": {},
        "h2sat": h2sat_track,
    }

    for sat_name, elements in cfg.SATELLITES.items():

        track = propagate_orbit(
            elements,
            t,
        )

        # ---------------------------------------------------------------
        # Ground station + IoT visibility
        # ---------------------------------------------------------------

        visibility = {}

        sites = {
            **cfg.GROUND_STATIONS,
            "IoT": cfg.IOT_PAYLOAD_SITE,
        }

        for site_name, site in sites.items():

            el = elevation_angle_deg(
                site["lat_deg"],
                site["lon_deg"],
                0.0,
                track["lat_deg"],
                track["lon_deg"],
                track["alt_km"],
            )

            visibility[site_name] = (
                el >= site["min_elevation_deg"]
            )

        # ---------------------------------------------------------------
        # H2Sat visibility
        # ---------------------------------------------------------------

        sees_h2sat = leo_sees_geo(
            track["lat_deg"],
            track["lon_deg"],
            track["alt_km"],
            cfg.H2SAT_LON_DEG,
        )

        # ---------------------------------------------------------------
        # Mission mode
        # ---------------------------------------------------------------

        modes = assign_modes(
            visibility,
            sees_h2sat,
        )

        results["sats"][sat_name] = dict(
            lat_deg=track["lat_deg"],
            lon_deg=track["lon_deg"],
            alt_km=track["alt_km"],
            visibility=visibility,
            sees_h2sat=sees_h2sat,
            mode=modes,
        )

    return results


# ---------------------------------------------------------------------------
# Mode assignment
# ---------------------------------------------------------------------------

def assign_modes(
    visibility,
    sees_h2sat,
):
    """
    Apply the mode-priority logic defined in config.py.

    Priority:

        1. IoT/Payload
        2. ISL
        3. DTE Optical
        4. Idle/Safe
    """

    n = len(sees_h2sat)

    modes = np.full(
        n,
        "Idle/Safe",
        dtype=object,
    )

    iot_visible = visibility["IoT"]

    tubogs_visible = visibility[
        "TUBOGS (Optical)"
    ]

    # ---------------------------------------------------------------
    # Priority 3: DTE Optical
    # ---------------------------------------------------------------

    modes[tubogs_visible] = "DTE Optical"

    # ---------------------------------------------------------------
    # Priority 2: ISL
    # ---------------------------------------------------------------

    if cfg.ISL_PREFER_REALTIME:
        modes[sees_h2sat] = "ISL (Real-Time)"
    else:
        modes[sees_h2sat] = "ISL (Store&Fwd)"

    # ---------------------------------------------------------------
    # Priority 1: IoT/Payload
    # ---------------------------------------------------------------

    modes[iot_visible] = "IoT/Payload"

    return modes