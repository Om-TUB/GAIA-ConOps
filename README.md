# GAIA-ConOps

A concept-of-operations (ConOps) simulator and visualizer for the **GAIA-MISSION** satellite system. It propagates the orbits of two LEO CubeSats (GAIA-A / GAIA-B), a GEO relay satellite (H2Sat / Heinrich Hertz), and a set of ground/payload sites, then determines and animates which **operations mode** each GAIA satellite is in at every timestep based on line-of-sight visibility.

## Repository layout

| File | Purpose |
|---|---|
| `config.py` | All physical constants, orbital design parameters, ground station/site definitions, mode-priority documentation, and visualization/export settings. |
| `main.py` | Entry point. Prints derived orbital parameters, verifies the 10:30 LST crossing claim numerically, generates the timeline, and launches the animation. |
| `orbit_propagation.py` | Orbital mechanics (Kepler solver, J2 secular rates, ECI→geodetic conversion), elevation-angle geometry, visibility checks, and mode-selection logic. |
| `visualization.py` | Animation / rendering of the ground tracks and mode state (not inspected in detail for this doc). |
| `earth_texture.jpg` | Basemap texture asset for the animation. |
| `sgp4_propagation.py` | SGP4 wrapper: build a satellite record from a TLE, build one from Keplerian elements, and propagate either forward in time. |
| `time_utils.py` | Julian date / GMST / subsolar-longitude helpers used by `config.py` and `main.py`. |

## Running it

```bash
python main.py
```

This will:
1. Print derived orbital parameters (inclination, period, RAAN, etc.) to stdout.
2. Numerically verify that GAIA-A's ground track actually crosses Etosha at ~10:30 LST.
3. Generate the mission timeline (3 hours by default, at 30-second resolution).
4. Open a live animation window (close it to end the program). If GIF export is enabled, also writes out an animated GIF.

## `config.py` — constants, orbit design, and site definitions

### Physical constants
Standard values: Earth's gravitational parameter, WGS84 equatorial radius, the J2 oblateness coefficient, Earth's precise sidereal rotation rate, and seconds-per-day.

### Simulation epoch
The real-world UTC date/time that simulation time zero corresponds to. This matters because the propagator (SGP4) is tied to real calendar dates — the sun's actual position and Earth's actual rotation angle at this specific moment feed directly into the orbit-design solve described below. The calendar date itself is arbitrary (a "today" for the demo); the time-of-day is solved for, not chosen, so that the mission's 10:30 local-solar-time design target comes out correct for that date.

### Simulation window
A 3-hour simulated duration at 30-second propagation steps — long enough to capture two useful passes over the ground station cluster.

### GAIA orbit design
The orbit is circular, sun-synchronous, at 500 km altitude. From that altitude the code derives:
- **Semi-major axis** (Earth radius + altitude).
- **Inclination**, computed (not hardcoded) from the standard J2 sun-synchronous condition: the orbit's nodal precession rate must match the rate the sun appears to move around the sky over a year (360°/365.2422 days). Solving the standard J2 RAAN-precession formula for inclination gives ~97.4°.
- **Orbital period**, from the standard two-body relation.

The satellite's starting point is offset one minute before its first designed crossing of the target site, so the first pass enters the animation from off-screen rather than starting mid-crossing.

### Solving for the ground-track crossing (RAAN and start position)
The ConOps document specifies *when* the ground track must cross the target (10:30 local solar time over Etosha National Park, Namibia) but leaves the orbit's right ascension of ascending node (RAAN) — effectively, *where in its orbital plane relative to the stars* — unspecified. RAAN is therefore treated as a free design variable and solved for, in three steps:

1. **Find the argument of latitude** (the angle around the orbit, measured from the ascending node) at which the orbit's ground track reaches Etosha's latitude, using the standard spherical-trigonometry relation between orbital inclination, latitude, and argument of latitude.
2. **Find the time-of-day** within the simulation epoch's calendar date at which the real sun sits at the exact longitude needed for a 10:30 local-solar-time crossing at Etosha's longitude. This uses the fact that the subsolar point's longitude sweeps very close to linearly through 360° per day, so a single linear solve from a sample at midnight is accurate to a fraction of a second. (An earlier version of this code sidestepped this by inventing a fictitious epoch where the sun sits at a fixed reference longitude — that shortcut silently broke once the propagator was tied to real calendar dates, because the real sun generally isn't there.)
3. **Solve for RAAN itself**, using the now-known real Earth-rotation angle (GMST) at the solved epoch, so that the satellite's ground-track longitude equals Etosha's longitude at the argument of latitude found in step 1.

### Satellite pair separation
The two GAIA satellites are placed 800 km apart along-track (within the mission's "under 1000 km" requirement), expressed as a small angular offset in their starting position around the orbit.

### Orbit propagation method
Because neither GAIA satellite has flown yet, there's no real fitted tracking data to seed the propagator from. Instead, the computed circular-orbit elements above are converted directly into propagator input. A placeholder atmospheric-drag parameter is estimated for a small SSO cube-shaped satellite and marked in-code to be replaced with a real fitted value once the satellites exist and are tracked.

H2Sat, by contrast, is a real satellite already in orbit (Heinrich Hertz, launched 2023) — it's propagated from an actual tracked two-line element set (TLE), with a note that TLE accuracy degrades over days to weeks and should be refreshed from a public tracking source for anything beyond a quick demo.

### Ground stations and sites
Three ground stations are defined by coordinates, communication band, and minimum required elevation angle above the horizon for a usable link:
- **TU Berlin** — UHF/VHF band, 5° minimum elevation.
- **TUBOGS** — Optical band, 20° minimum elevation (optical links need a much clearer, higher-elevation view).
- **Fraunhofer IIS** — Ka-band, 5° minimum elevation.

A fourth site, Etosha National Park in Namibia, is defined the same way as the target for the IoT payload (5° minimum elevation).

### Mode-priority design intent (as written in this file's comments)
This file's comments describe an intended priority order for evaluating which operations mode a satellite is in, per satellite per timestep: IoT payload visibility first, then inter-satellite link opportunities with H2Sat (split conceptually into "real-time" vs. "store-and-forward" depending on link budget), then direct optical downlink, then a reserved-but-unused cross-link optical mode, with everything else (sun-pointing, nadir-tracking, de-tumbling, idle, safe/critical) collapsed into a single idle/safe default. A five-entry color table is defined here to match that intended set of modes.

**This intended design does not fully match what the mode-selection code in `orbit_propagation.py` actually does — see the internal documentation for the discrepancies.** This file's mode comments should be treated as a design note, not as accurate current behavior.

### Visualization / export settings
Animation frame timing, how many simulated seconds advance per rendered frame, the map's longitude/latitude bounds, how long a satellite's ground-track trail persists, and GIF export settings (on/off, filename, frame rate, resolution).

## `main.py` — entry point

On startup, the program:
1. Prints a block of derived orbital parameters to the console (altitude, semi-major axis, computed inclination, orbital period, computed RAAN, starting position, satellite separation, H2Sat's nominal longitude).
2. Runs a **numerical self-check**: it generates a short timeline covering slightly more than one orbit, finds the single closest point GAIA-A's ground track comes to Etosha's coordinates, and at that exact moment (not at the simulation's start time — the sun moves ~15°/hour across the Earth-fixed frame, which matters over this time window) computes what local solar time the crossing actually occurred at. It prints both the target and the computed value so a mismatch would be immediately visible.
3. Generates the full mission timeline using the default duration and timestep from `config.py`.
4. Launches the live animation window, which runs until the user closes it.

## `orbit_propagation.py` — mechanics, visibility, and mode assignment

### Low-level orbital mechanics
- A **Kepler equation solver** — an iterative Newton's-method solve that converts mean anomaly (roughly, "time-averaged position around the orbit") into eccentric anomaly, the intermediate quantity needed to get true position.
- A conversion from eccentric anomaly to **true anomaly** (actual angular position around the orbit, accounting for the orbit's shape).
- A calculation of **J2 secular drift rates** — the standard first-order correction for how Earth's slightly non-spherical shape causes an orbit's orientation (RAAN) and the location of its low point (argument of perigee) to slowly rotate over time.

### Propagating a satellite forward in time
Given a set of six orbital elements (size, shape, tilt, orientation, and starting position) and a list of times, this produces the satellite's 3D position at each time, then converts that into latitude, longitude, and altitude by: advancing the orbit's mean position forward using the two-body motion rate, re-solving Kepler's equation at each new time, applying the J2 drift to the orientation angles, converting to Earth-centered inertial coordinates, then rotating into an Earth-fixed frame using Earth's rotation angle at each moment (derived from the rotation angle at the solved simulation epoch plus Earth's precise rotation rate times elapsed time).

### GEO satellite position
A simplified fixed-longitude model is available for a geostationary satellite (station-kept at a constant longitude, zero latitude, standard GEO altitude) — used as a fallback/reference; the actual H2Sat position used in the timeline comes from real TLE propagation instead (see below).

### Coordinate conversion and visibility geometry
- A **geodetic-to-Earth-fixed** coordinate conversion (latitude/longitude/altitude → 3D Cartesian), used both for ground sites and for satellites.
- An **elevation-angle calculation**: given an observer's position and a target's position, this computes the angle of the target above the observer's local horizon, by taking the line-of-sight vector between them and measuring its angle against the observer's local "straight up" direction. This is the core geometry used everywhere visibility is checked — ground station to satellite, satellite to IoT site, and satellite to H2Sat.
- A specific check for whether a LEO satellite can see H2Sat, built on the elevation-angle calculation, treating the LEO satellite as the observer and H2Sat as the target, with "visible" meaning any elevation above the local horizon (0°).

### Building the full mission timeline
For the requested duration and timestep, the code:
1. Adds two extra orbital periods of "pre-roll" propagation time before the displayed simulation clock starts at zero, so that satellites are already in a natural position (not artificially starting exactly at their designed crossing point) when the visible animation begins.
2. Propagates H2Sat using its real tracked TLE.
3. For each GAIA satellite, builds propagator input directly from its computed circular-orbit elements (no TLE exists yet, so this seeds the propagator from the design values in `config.py` instead), then propagates it forward.
4. For each GAIA satellite at every timestep, checks elevation-angle visibility against each ground station and the IoT site, and separately checks visibility to H2Sat (using H2Sat's actual propagated position, not the simplified fixed-longitude fallback).
5. Feeds all of that visibility information into the mode-assignment logic (below) to get each satellite's operations mode at every timestep.
6. Returns everything — positions, visibility flags per site, H2Sat visibility, and assigned modes — for each satellite, keyed by satellite name, plus H2Sat's own track.

### Mode assignment logic
This is the actual operations-mode decision logic (distinct from — and, in a few respects, inconsistent with — the design intent described in `config.py`'s comments; see the internal documentation for the discrepancies). Every satellite starts each timestep in a default "idle/safe" state. Then, in order, visibility to H2Sat, to the TU Berlin ground station, to the TUBOGS optical ground station, and finally to the IoT site are each checked, and where visible, overwrite the satellite's mode for that timestep with the corresponding mode label. Because each check overwrites the previous one, the **last** check applied wins — which means the real priority (highest to lowest) is: IoT site visibility, then TUBOGS optical visibility, then TU Berlin visibility, then H2Sat visibility, with idle/safe remaining only where none of the four are true. The H2Sat-based mode is only assigned at all if the "prefer real-time" link setting in `config.py` is enabled — there is no separate logic for a "store-and-forward" alternative; H2Sat visibility always produces the same single mode label regardless of that setting's intended meaning.

## `visualization.py`

Not inspected in detail for this document. It's invoked from `main.py` with the generated timeline and is responsible for rendering the animated ground-track map (and, per the settings in `config.py`, optionally exporting it as a GIF).

## Related documentation

See `ConOps.doc` for the operations-modes list written for internal/non-technical reference, and for the assumptions made in producing this documentation.
