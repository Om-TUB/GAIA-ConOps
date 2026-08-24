import numpy as np
from datetime import datetime, timezone, timedelta

from time_utils import jd_fr, gmst_rad, subsolar_longitude_deg

# PHYSICAL CONSTANTS
MU_EARTH = 398600.4418          # km^3/s^2, Earth gravitational parameter
R_EARTH = 6378.137               # km, WGS84 equatorial radius
J2 = 1.08262668e-3               # Earth J2 oblateness coefficient
EARTH_ROTATION_RATE = 360.98564736629 / 86400.0   # deg/s (sidereal, precise)
SECONDS_PER_DAY = 86400.0

# SGP4 PROPAGATION EPOCH
#
# Real-world UTC epoch that simulation time t=0 corresponds to. Needed up
# here (before the RAAN/LST solve below) because — unlike the old J2-only
# propagator, which used an arbitrary "GMST=0, sun at 0 deg longitude at
# t=0" convention — SGP4 is tied to real dates, so the sun's actual
# position and Earth's actual rotation angle at this specific moment now
# feed directly into that solve. The calendar date is an arbitrary "today"
# for the demo; the time-of-day is solved for below so the mission's
# 10:30 LST design target actually comes out right for that date.
_SIM_EPOCH_DATE = datetime(2026, 8, 24, 0, 0, 0, tzinfo=timezone.utc)

# SIMULATION WINDOW
SIM_DURATION_S = 3600       # 3 hours: enough for two useful GAIA passes
SIM_TIMESTEP_S = 30              # propagation step (s) — animation subsamples this
# Epoch: arbitrary reference start time (J2000-relative days), used only for
# Earth rotation angle (GMST) bookkeeping. Day 0 = start of simulation.
EPOCH_DAY_OFFSET = 0.0

# GAIA LEO SATELLITE ORBIT (AICP-Cube A / B)-
# circular SSO, 500 km altitude, ground track crosses
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

# Start slightly east/right of the first Etosha pass.
# Negative = start before the designed 10:30 crossing.
GAIA_START_OFFSET_S = -60.0      # 1 minute before the target crossing


# --- RAAN: chosen so ground track crosses Namibia at 10:30 LOCAL SOLAR TIME
# For a sun-synchronous orbit, Local Time of Ascending/relevant Node is fixed
# by RAAN relative to the sun's position. We solve for the RAAN that places
# the satellite over Etosha (16 deg E, -19 deg lat) at simulation t=0, at a
# true anomaly consistent with 10:30 local solar time.
#
# This fixes RAAN_0 and true_anomaly_0 for satellite A at t=0:
#   1) latitude of ground track = -19.0 deg (Etosha latitude) at crossing
#   2) local solar time at that crossing = 10:30
#   3) longitude at that crossing = 16.0 deg E (Etosha longitude)
#
# NOTE: RAAN itself is a design choice, not a document-specified value (the
# ConOps left it blank) — the target LST is the actual requirement.
#
# Getting (2) and (3) to both hold at the same instant t=0 needs the real
# sun position and real Earth-rotation angle at whatever calendar date the
# sim starts on (SIM_EPOCH_UTC). Earlier revisions of this file sidestepped
# that by inventing a fictitious epoch where the sun sits at 0 deg
# Earth-fixed longitude and GMST=0 — convenient algebra, but it silently
# breaks once the propagator (SGP4, see sgp4_propagation.py) is tied to a
# real UTC date, since the real sun and real Earth rotation angle at that
# date generally aren't 0. Fixed here by solving for the time-of-day
# component of SIM_EPOCH_UTC that actually makes the subsolar longitude
# equal what's needed for a 10:30 crossing over Etosha, then using the
# real GMST at that solved epoch for the RAAN solve below.

ETOSHA_LAT_DEG = -19.0
ETOSHA_LON_DEG = 16.0
TARGET_LOCAL_SOLAR_TIME_HOURS = 10.5   # 10:30 am

# Latitude argument (argument of latitude, u = omega + true_anomaly) at which
# the satellite crosses ETOSHA_LAT_DEG, ascending pass:
#   sin(lat) = sin(i) * sin(u)
_sin_u = np.sin(np.deg2rad(ETOSHA_LAT_DEG)) / np.sin(np.deg2rad(GAIA_INCLINATION_DEG))
_u_ascending = np.arcsin(np.clip(_sin_u, -1.0, 1.0))   # radians, ascending-pass solution
# Keep RAAN targeted to the Etosha crossing, but start the simulation
# one minute earlier so the first pass enters the animation from the right.
GAIA_TRUE_ANOMALY_0_DEG = (
    np.rad2deg(_u_ascending)
    - GAIA_ARG_PERIGEE_DEG
    + np.rad2deg(
        (2.0 * np.pi / GAIA_ORBITAL_PERIOD_S) * GAIA_START_OFFSET_S
    )
)

# Local solar time -> hour angle of sun relative to satellite subpoint longitude.
# LST_hours = 12 + (lon_deg - subsolar_lon_deg) / 15
# Solve for the subsolar longitude a 10:30 crossing at Etosha's longitude
# requires, then solve for the UTC time-of-day (on _SIM_EPOCH_DATE's
# calendar date) at which the real sun is actually at that longitude.
# Subsolar longitude sweeps a very close to exactly linear -360 deg per
# solar day (equation-of-time curvature is under a minute over the few
# hours searched here), so a single linear solve from a t=0 sample is
# accurate to a small fraction of a second of local time.
_target_subsolar_lon_deg = (
    ETOSHA_LON_DEG - 15.0 * (TARGET_LOCAL_SOLAR_TIME_HOURS - 12.0)
)

_jd0, _ = jd_fr(_SIM_EPOCH_DATE)
_subsolar_lon_at_midnight_deg = subsolar_longitude_deg(_jd0, 0.0)
_delta_lon_deg = (
    (_target_subsolar_lon_deg - _subsolar_lon_at_midnight_deg + 180.0) % 360.0
    - 180.0
)
# Subsolar longitude increases ~+360 deg per UTC day (as Earth's rotation
# carries the Greenwich meridian eastward under the sun), not decreases —
# verified empirically against subsolar_longitude_deg() rather than assumed.
_epoch_day_fraction = (_delta_lon_deg / 360.0) % 1.0

SIM_EPOCH_UTC = _SIM_EPOCH_DATE + timedelta(days=_epoch_day_fraction)

_jd_epoch, _fr_epoch = jd_fr(SIM_EPOCH_UTC)
GMST_AT_EPOCH_DEG = np.rad2deg(gmst_rad(_jd_epoch, _fr_epoch))
SUBSOLAR_LON_AT_EPOCH_DEG = subsolar_longitude_deg(_jd_epoch, _fr_epoch)

# Inertial subsatellite longitude (Earth-fixed frame):
#   lambda = RAAN + atan2(cos(i)*sin(u), cos(u)) - GMST(t)
# At t=0, solve RAAN so lambda = ETOSHA_LON_DEG at u = u_ascending, given
# the real GMST at the (now solved-for) epoch:
_i_rad = np.deg2rad(GAIA_INCLINATION_DEG)
_delta_lon = np.arctan2(np.cos(_i_rad) * np.sin(_u_ascending), np.cos(_u_ascending))
GAIA_RAAN_0_DEG = ETOSHA_LON_DEG - np.rad2deg(_delta_lon) + GMST_AT_EPOCH_DEG

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

# SGP4 PROPAGATION
#
# (SIM_EPOCH_UTC itself is solved for above, before the RAAN/LST targeting
# block, since that solve needs it.)

# GAIA-A/B haven't flown, so there's no fitted TLE to seed SGP4 from — see
# sgp4_propagation.py docstring for how they're initialized instead.
# BSTAR estimated for a 500 km SSO 3U-class CubeSat (Cd ~ 2.2,
# area/mass ~ 0.01 m^2/kg); replace with the real fitted value once the
# satellites are catalogued.
GAIA_BSTAR = 1.2e-4  # 1/earth radii

# H2SAT (Heinrich Hertz) — GEO relay satellite, real satellite launched 2023
# (NORAD 57213 / COSPAR 2023-093A). Ground-track longitude/altitude below
# are only used as fallbacks/labels; actual position now comes from SGP4
# propagation of the tracked TLE below.
H2SAT_LON_DEG = 0.7  # nominal station-kept longitude (datasheet)
H2SAT_ALTITUDE_KM = 35786.0   # standard GEO altitude
H2SAT_SEMI_MAJOR_AXIS_KM = R_EARTH + H2SAT_ALTITUDE_KM
H2SAT_ECCENTRICITY = 0.0
H2SAT_INCLINATION_DEG = 0.0

# Real tracked TLE for H2Sat, NORAD 57213. TLEs age — accuracy degrades
# from days to weeks after epoch, so refresh this from a public source
# (e.g. celestrak.org, CATNR=57213) for anything beyond a quick demo run.
H2SAT_TLE_LINE1 = "1 57213U 23093A   26235.07752475 -.00000001  00000-0  00000-0 0  9992"
H2SAT_TLE_LINE2 = "2 57213   0.0206  51.1055 0001204 310.3713 358.5178  1.00270896 11596"
H2SAT_TLE_EPOCH_NOTE = "epoch 2026 day 235.0775 (~Aug 23, 2026 01:51 UTC)"

# GROUND STATIONS / FIXED SITES (user-provided coordinates)
GROUND_STATIONS = {
    "TU Berlin (UHF/VHF)": dict(
        lat_deg=52.5152,
        lon_deg=13.3236,
        band="UHF/VHF",
        min_elevation_deg=5.0,   
        color="#2ca02c",
        marker="^",
    ),
    "TUBOGS (Optical)": dict(
        lat_deg=53.3297,   # 53 19' 46.9" N
        lon_deg=13.0725,   # 13 04' 21.0" E
        band="Optical",
        min_elevation_deg=20.0,   
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

# IoT PAYLOAD TARGET AREA (Etosha National Park, Namibia)
IOT_PAYLOAD_SITE = dict(
    name="Etosha NP (IoT tags)",
    lat_deg=-19.0,     # 19 00' 00" S
    lon_deg=16.0,      # 16 00' 00" E
    min_elevation_deg=5.0,
    color="#d62728",
    marker="*",
)

# SATELLITE MODES
#
# Mode SELECTION LOGIC
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
# edit MODE_PRIORITY logic in orbit_propagation.py for different behavior.

MODE_COLORS = {
    "IoT/Payload":       "#2ca02c",   # green - nominal ops
    "ISL (Real-Time)":   "#1f77b4",   # blue
    "ISL (Store&Fwd)":   "#17becf",   # cyan
    "DTE Optical":       "#9467bd",   # purple
    "Idle/Safe":         "#7f7f7f",   # gray
}

# Fraction-based toggle for ISL link type when both LEO-H2Sat and H2Sat-
# ground geometry allow it; kept simple/deterministic rather than a random
# Set True: prefer real-time whenever geometrically possible.
ISL_PREFER_REALTIME = True

# VISUALIZATION SETTINGS
ANIMATION_INTERVAL_MS = 75        # wall-clock ms between animation frames
FRAME_SIM_STEP_S = 45             # sim-seconds advanced per animation frame
MAP_LON_RANGE = (-180, 180)
MAP_LAT_RANGE = (-90, 90)
GROUND_TRACK_TRAIL_MINUTES = 30   # length of fading trail behind each satellite

# GIF EXPORT
SAVE_GIF = True                  # set True to export the animation
GIF_FILENAME = "gaia_conops.gif"
GIF_FPS = 20
GIF_DPI = 120