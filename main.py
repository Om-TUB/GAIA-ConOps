from datetime import timedelta

import numpy as np
import config as cfg
import orbit_propagation as prop
from visualization import run_animation
from time_utils import jd_fr, subsolar_longitude_deg


def print_derived_parameters():
    print("=" * 70)
    print("GAIA CONOPS SIMULATOR — Derived Orbital Parameters")
    print("=" * 70)
    print(f"GAIA orbit altitude:            {cfg.GAIA_ALTITUDE_KM:.1f} km")
    print(f"GAIA semi-major axis:           {cfg.GAIA_SEMI_MAJOR_AXIS_KM:.3f} km")
    print(f"GAIA inclination (SSO, computed): {cfg.GAIA_INCLINATION_DEG:.4f} deg")
    print(f"GAIA orbital period:            {cfg.GAIA_ORBITAL_PERIOD_S/60:.2f} min "
          f"({cfg.GAIA_ORBITAL_PERIOD_S:.1f} s)")
    print(f"GAIA RAAN_0 (computed):         {cfg.GAIA_RAAN_0_DEG:.4f} deg")
    print(f"GAIA true anomaly_0 (sat A):    {cfg.GAIA_TRUE_ANOMALY_0_DEG:.4f} deg")
    print(f"GAIA-A / GAIA-B separation:     {cfg.GAIA_SAT_SEPARATION_KM:.1f} km "
          f"({cfg.GAIA_SAT_SEPARATION_DEG:.4f} deg true anomaly)")
    print(f"H2Sat GEO longitude (from datasheet): {cfg.H2SAT_LON_DEG} deg E")
    print("-" * 70)

    # Verify the Namibia 10:30 LST crossing claim numerically
    timeline = prop.generate_timeline(duration_s=cfg.GAIA_ORBITAL_PERIOD_S * 1.2,
                                       step_s=1.0)
    sat = timeline["sats"]["GAIA-A"]
    lat = sat["lat_deg"]
    lon = sat["lon_deg"]
    t = timeline["t_s"]

    target_lat = cfg.ETOSHA_LAT_DEG
    target_lon = cfg.ETOSHA_LON_DEG
    dist = np.sqrt((lat - target_lat) ** 2 +
                    (np.mod(lon - target_lon + 180, 360) - 180) ** 2)
    closest_idx = np.argmin(dist)

    crossing_lon = lon[closest_idx]

    # Real subsolar longitude AT THE CROSSING TIME (not at epoch): the sun
    # moves ~15 deg/hour in the Earth-fixed frame, which is not negligible
    # over the up-to ~114 min window searched above, so this is sampled at
    # t[closest_idx] rather than reusing a value fixed at t=0.
    crossing_dt = cfg.SIM_EPOCH_UTC + timedelta(seconds=float(t[closest_idx]))
    jd_cross, fr_cross = jd_fr(crossing_dt)
    subsolar_lon_at_cross = subsolar_longitude_deg(jd_cross, fr_cross)

    local_solar_time = 12.0 + (crossing_lon - subsolar_lon_at_cross) / 15.0
    local_solar_time = local_solar_time % 24.0

    print(f"Verification: closest approach to Etosha at t={t[closest_idx]:.1f}s")
    print(f"  lat={lat[closest_idx]:.3f} deg, lon={lon[closest_idx]:.3f} deg "
          f"(target: {target_lat}, {target_lon})")
    print(f"  computed local solar time at crossing: {local_solar_time:.2f} h "
          f"(target: {cfg.TARGET_LOCAL_SOLAR_TIME_HOURS:.2f} h)")
    print("=" * 70)
    print()


def main():
    print_derived_parameters()

    print(f"Generating {cfg.SIM_DURATION_S/3600:.1f} h timeline "
          f"at {cfg.SIM_TIMESTEP_S} s resolution...")
    timeline = prop.generate_timeline()
    print("Timeline generated. Launching live animation window...")
    print("(Close the plot window to end the program.)")

    run_animation(timeline)


if __name__ == "__main__":
    main()