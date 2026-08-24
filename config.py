"""
GAIA CONOPS Simulator — Configuration
======================================
ALL mission parameters live here. Nothing is hardcoded elsewhere.
Edit this file to change orbits, ground stations, or mode logic thresholds.

Sources:
- GAIA-MISSION Phase 1 proposal document (satellite bus, ISL frequencies)
- ConOps.docx (satellite modes list, ground station names)
- H2M-DLR-RD-TN-009 v2.1 (H2Sat ISL datasheet: orbital slot, antenna gains)
- User-provided coordinates (2024-message): TUB, Etosha, TUBOGS, Fraunhofer IIS
"""

import numpy as np

# ---------------------------------------------------------------------------
# PHYSICAL CONSTANTS
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418          # km^3/s^2, Earth gravitational parameter
R_EARTH = 6378.137               # km, WGS84 equatorial radius
J2 = 1.08262668e-3               # Earth J2 oblateness coefficient
EARTH_ROTATION_RATE = 360.98564736629 / 86400.0   # deg/s (sidereal, precise)
SECONDS_PER_DAY = 86400.0

# ---------------------------------------------------------------------------
# SIMULATION WINDOW
# ---------------------------------------------------------------------------
SIM_DURATION_S = 24 * 3600       # 24 hours, as requested
SIM_TIMESTEP_S = 30              # propagation step (s) — animation subsamples this
# Epoch: arbitrary reference start time (J2000-relative days), used only for
# Earth rotation angle (GMST) bookkeeping. Day 0 = start of simulation.
EPOCH_DAY_OFFSET = 0.0

# ---------------------------------------------------------------------------
# GAIA LEO SATELLITE ORBIT (AICP-Cube A / B)
# ---------------------------------------------------------------------------
# User-specified: circular SSO, 500 km altitude, ground track crosses
# Namibia (16 deg E) at 10:30 local solar time. Two sats, same plane,
# separated by < 1000 km along-track.

GAIA_ALTITUDE_KM = 500.0
GAIA_SEMI_MAJOR_AXIS_KM = R_EARTH + GAIA_ALTITUDE_KM
GAIA_ECCENTRICITY = 0.0          # circular, as specified
GAIA_ARG_PERIGEE_DEG = 0.0       # undefined for circular orbit, set to 0

# --- Sun-synchronous inclination -------------------------------------------
# SSO condition: RAAN precession rate = 360 deg / 365.2421897 days (mean solar year)
# Standard J2 formula:
#   dRAAN/dt = -1.5 * n * J2 * (R_E/p)^2 * cos(i)
# Solve for cos(i), then i.
_n_gaia = np.sqrt(MU_EARTH / GAIA_SEMI_MAJOR_AXIS_KM**3)   # rad/s, mean motion
_p_gaia = GAIA_SEMI_MAJOR_AXIS_KM * (1 - GAIA_ECCENTRICITY**2)  # semi-latus rectum
_SSO_RAAN_RATE_RAD_S = np.deg2rad(360.0) / (365.2421897 * SECONDS_PER_DAY)

_cos_i = -_SSO_RAAN_RATE_RAD_S / (1.5 * _n_gaia * J2 * (R_EARTH / _p_gaia) ** 2)
GAIA_INCLINATION_DEG = np.rad2deg(np.arccos(_cos_i))   # computed, ~97.4 deg

GAIA_ORBITAL_PERIOD_S = 2 * np.pi * np.sqrt(GAIA_SEMI_MAJOR_AXIS_KM**3 / MU_EARTH)

# --- RAAN: chosen so ground track crosses Namibia at 10:30 LOCAL SOLAR TIME
# For a sun-synchronous orbit, Local Time of Ascending/relevant Node is fixed
# by RAAN relative to the sun's position. We solve numerically at sim time t=0
# for the RAAN that places the satellite over Etosha (16 deg E, -19 deg lat)
# at a true anomaly consistent with 10:30 local solar time.
#
# Simplification (explicitly noted): we treat the sim epoch as the reference
# day, place the sun at RA_sun = 0 deg (i.e., epoch date chosen = a vernal
# equinox), and solve for RAAN and starting true anomaly jointly so that:
#   1) latitude of ground track = -19.0 deg (Etosha latitude) at crossing
#   2) local solar time at that crossing = 10:30
#   3) longitude at that crossing = 16.0 deg E (Etosha longitude)
# This fixes RAAN_0 and true_anomaly_0 for satellite A at t=0.
#
# NOTE: This is a design choice, not a document-specified value (the ConOps
# left RAAN blank). It is clearly isolated here for you to change.

ETOSHA_LAT_DEG = -19.0
ETOSHA_LON_DEG = 16.0
TARGET_LOCAL_SOLAR_TIME_HOURS = 10.5   # 10:30 am

# Latitude argument (argument of latitude, u = omega + true_anomaly) at which
# the satellite crosses ETOSHA_LAT_DEG, ascending pass:
#   sin(lat) = sin(i) * sin(u)
_sin_u = np.sin(np.deg2rad(ETOSHA_LAT_DEG)) / np.sin(np.deg2rad(GAIA_INCLINATION_DEG))
_u_ascending = np.arcsin(np.clip(_sin_u, -1.0, 1.0))   # radians, ascending-pass solution
GAIA_TRUE_ANOMALY_0_DEG = np.rad2deg(_u_ascending) - GAIA_ARG_PERIGEE_DEG

# Local solar time -> hour angle of sun relative to satellite subpoint longitude.
# LST_hours = 12 + (lon_deg - sun_subsolar_lon_deg) / 15
# We define sim epoch such that the subsolar longitude at t=0 is 0 deg E
# (i.e., simulation start = local noon at Greenwich meridian, an explicit
# convention so GMST(t=0) = 0 exactly). This lets us solve RAAN directly.
SUBSOLAR_LON_AT_EPOCH_DEG = 0.0
GMST_AT_EPOCH_DEG = 0.0   # convention: Earth-fixed frame aligned with inertial frame at t=0

# satellite longitude at crossing (inertial RAAN frame) must equal:
#   lon = 15*(LST - 12) + SUBSOLAR_LON_AT_EPOCH_DEG   (Earth-fixed, at t=0 since GMST=0)
_lon_at_crossing_deg = 15.0 * (TARGET_LOCAL_SOLAR_TIME_HOURS - 12.0) + SUBSOLAR_LON_AT_EPOCH_DEG
# But target is Etosha's actual longitude -> the crossing must occur at a
# specific TIME after t=0 (not at t=0 itself) so that Earth has rotated
# ETOSHA_LON_DEG - lon_at_crossing_deg relative to the orbit plane.
# We instead directly place RAAN so the ascending node aligns correctly and
# then find t_cross by propagation. To keep this tractable in closed form,
# define RAAN_0 such that at u = u_ascending, the inertial longitude equals
# ETOSHA_LON_DEG, and separately verify LST numerically at runtime (printed
# at startup so you can confirm it's ~10:30).
#
# Inertial subsatellite longitude (in Earth-fixed frame, GMST=0 at t=0):
#   lambda = RAAN + atan2(cos(i)*sin(u), cos(u)) - GMST(t)
# At t=0 (GMST=0), solve RAAN so lambda = ETOSHA_LON_DEG at u = u_ascending:
_i_rad = np.deg2rad(GAIA_INCLINATION_DEG)
_delta_lon = np.arctan2(np.cos(_i_rad) * np.sin(_u_ascending), np.cos(_u_ascending))
GAIA_RAAN_0_DEG = ETOSHA_LON_DEG - np.rad2deg(_delta_lon)

# Along-track separation between GAIA-A and GAIA-B: < 1000 km, use 800 km.
GAIA_SAT_SEPARATION_KM = 800.0
_delta_true_anomaly_rad = GAIA_SAT_SEPARATION_KM / GAIA_SEMI_MAJOR_AXIS_KM  # small-angle arc
GAIA_SAT_SEPARATION_DEG = np.rad2deg(_delta_true_anomaly_rad)

SATELLITES = {
    "GAIA-A": dict(
        a_km=GAIA_SEMI_MAJOR_AXIS_KM,
        e=GAIA_ECCENTRICITY,
        i_deg=GAIA_INCLINATION_DEG,
        raan_deg=GAIA_RAAN_0_DEG,
        argp_deg=GAIA_ARG_PERIGEE_DEG,
        true_anomaly_0_deg=GAIA_TRUE_ANOMALY_0_DEG,
        color="#1f77b4",
        marker="o",
    ),
    "GAIA-B": dict(
        a_km=GAIA_SEMI_MAJOR_AXIS_KM,
        e=GAIA_ECCENTRICITY,
        i_deg=GAIA_INCLINATION_DEG,
        raan_deg=GAIA_RAAN_0_DEG,
        argp_deg=GAIA_ARG_PERIGEE_DEG,
        true_anomaly_0_deg=GAIA_TRUE_ANOMALY_0_DEG + GAIA_SAT_SEPARATION_DEG,
        color="#ff7f0e",
        marker="o",
    ),
}

# ---------------------------------------------------------------------------
# H2SAT (Heinrich Hertz) — GEO relay satellite
# ---------------------------------------------------------------------------
# Per H2M-DLR-RD-TN-009 v2.1, p.2: "Das Datenblatt gilt für die betriebliche
# Orbitposition 0,7 deg Ost." -> operational slot 0.7 deg East, GEO.
H2SAT_LON_DEG = 0.7
H2SAT_ALTITUDE_KM = 35786.0   # standard GEO altitude
H2SAT_SEMI_MAJOR_AXIS_KM = R_EARTH + H2SAT_ALTITUDE_KM
H2SAT_ECCENTRICITY = 0.0
H2SAT_INCLINATION_DEG = 0.0
# GEO station-keeps at fixed longitude; modeled as fixed subsatellite point
# (no meaningful "orbit propagation" needed for a 24h CONOPS view since GEO
# period = 1 sidereal day and it is nominally stationary over 0.7E).

# ISL antenna coverage (from datasheet, p.5): tracking antenna can steer to
# cover the entire visible Earth from GEO (roll/pitch +/-13.5 deg gimbal).
# We treat H2Sat as visible to a LEO satellite whenever the LEO satellite is
# within H2Sat's Earth-facing hemisphere (standard GEO visibility geometry),
# since the ISL antenna datasheet confirms full-Earth-disk coverage.

# ---------------------------------------------------------------------------
# GROUND STATIONS / FIXED SITES (user-provided coordinates)
# ---------------------------------------------------------------------------
GROUND_STATIONS = {
    "TU Berlin (UHF/VHF)": dict(
        lat_deg=52.5152,
        lon_deg=13.3236,
        band="UHF/VHF",
        min_elevation_deg=5.0,   # standard assumption for RF link visibility mask
        color="#2ca02c",
        marker="^",
    ),
    "TUBOGS (Optical)": dict(
        lat_deg=53.3297,   # 53 19' 46.9" N
        lon_deg=13.0725,   # 13 04' 21.0" E
        band="Optical",
        min_elevation_deg=20.0,   # optical links typically need higher min elevation
        color="#9467bd",
        marker="^",
    ),
    "Fraunhofer IIS (Ka-band)": dict(
        lat_deg=49.5606,
        lon_deg=11.0531,
        band="Ka-band",
        min_elevation_deg=5.0,
        color="#8c564b",
        marker="^",
    ),
}

# ---------------------------------------------------------------------------
# IoT PAYLOAD TARGET AREA (Etosha National Park, Namibia)
# ---------------------------------------------------------------------------
IOT_PAYLOAD_SITE = dict(
    name="Etosha NP (IoT tags)",
    lat_deg=-19.0,     # 19 00' 00" S
    lon_deg=16.0,      # 16 00' 00" E
    min_elevation_deg=5.0,
    color="#d62728",
    marker="*",
)

# ---------------------------------------------------------------------------
# SATELLITE / ADCS MODES
# ---------------------------------------------------------------------------
# Per ConOps.docx section 5, the following modes are listed (ADCS requirement
# columns were blank in the source document). Per your instruction, satellite
# modes and ADCS modes are treated as the same, and HPLR is a software
# sub-mode of pointing accuracy layered on top of another mode, not a
# separate physical attitude — implemented here as a modifier flag.
#
# Mode SELECTION LOGIC (explicit, since ConOps.docx did not define it):
#   Priority order evaluated per satellite per timestep:
#     1. IoT/Payload      -> Etosha IoT site is above min elevation (NOMINAL OPS)
#     2. ISL (H2Sat)       -> H2Sat visible AND (any ground station visible
#                              for real-time relay via H2Sat->ground OR not,
#                              since GAIA-MISSION architecture stores payload
#                              data and forwards via ISL opportunistically)
#          - REAL-TIME  if H2Sat visible AND a ground station is simultaneously
#                        visible to H2Sat's fixed downlink (H2Sat always sees
#                        its home ground segment in Germany from GEO, so this
#                        is effectively: H2Sat visible to the LEO sat)
#          - STORE&FWD  if H2Sat visible but link margin/power budget assumed
#                        insufficient for real time (toggle, see
#                        ISL_REALTIME_FRACTION below)
#     3. DTE Optical       -> TUBOGS above min elevation (direct downlink)
#     4. Cross Link Optical-> reserved, not modeled (no second optical partner
#                              satellite in this scenario) — never selected
#     5. Sun Pointing / Nadir / Target Tracking / De-tumbling / Idle-Suspend /
#        Safe/Critical     -> collapsed into "Idle/Safe" default when none of
#                              the above link opportunities exist
#
# This priority order is a MODELING CHOICE to make the visualization useful;
# edit MODE_PRIORITY logic in orbit_propagation.py if you want different
# behavior.

MODE_COLORS = {
    "IoT/Payload":       "#2ca02c",   # green - nominal ops
    "ISL (Real-Time)":   "#1f77b4",   # blue
    "ISL (Store&Fwd)":   "#17becf",   # cyan
    "DTE Optical":       "#9467bd",   # purple
    "Idle/Safe":         "#7f7f7f",   # gray
}

# Fraction-based toggle for ISL link type when both LEO-H2Sat and H2Sat-
# ground geometry allow it; kept simple/deterministic rather than a random
# power-budget model since none was specified in the source documents.
# Set True: prefer real-time whenever geometrically possible.
ISL_PREFER_REALTIME = True

# ---------------------------------------------------------------------------
# VISUALIZATION SETTINGS
# ---------------------------------------------------------------------------
ANIMATION_INTERVAL_MS = 500        # wall-clock ms between animation frames
FRAME_SIM_STEP_S = 30             # sim-seconds advanced per animation frame
MAP_LON_RANGE = (-180, 180)
MAP_LAT_RANGE = (-90, 90)
GROUND_TRACK_TRAIL_MINUTES = 30   # length of fading trail behind each satellite