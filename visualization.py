import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import matplotlib.image as mpimg
import config as cfg

# Utility functions
def format_hms(seconds):
    """Convert simulation seconds to HH:MM:SS."""
    seconds = max(0, float(seconds))

    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    return f"{h:02d}:{m:02d}:{s:02d}"


def yes_no(value):
    return "YES" if bool(value) else "NO"
# Animator
class GaiaConopsAnimator:

    def __init__(self, timeline):

        self.timeline = timeline

        self.t = np.asarray(timeline["t_s"])
        self.dt = self.t[1] - self.t[0]
        self.n_steps = len(self.t)

        # Animation stepping


        self.step_stride = max(
            1,
            int(round(cfg.FRAME_SIM_STEP_S / self.dt))
        )

        self.frame_indices = np.arange(
            0,
            self.n_steps,
            self.step_stride
        )

        # Ground-track trail length
        self.trail_len_steps = max(
            1,
            int(round(
                cfg.GROUND_TRACK_TRAIL_MINUTES * 60 / self.dt
            ))
        )

        # Earth image
        self.earth_image = None

        # Dynamic artists
        self.sat_points = {}
        self.sat_trails = {}
        self.sat_labels = {}

        self.gs_points = {}
        self.gs_labels = {}

        self.contact_lines = {}
        self.h2sat_contact_lines = {}

        self.mode_texts = {}
        self.status_texts = {}

        self._build_figure()

    # Earth texture
    def _load_earth_texture(self):

        texture_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "earth_texture.jpg"
        )

        if not os.path.exists(texture_path):
            raise FileNotFoundError(
                "\n"
                "Earth texture not found.\n\n"
                f"Expected:\n  {texture_path}\n\n"
                "Download a NASA Blue Marble equirectangular image "
                "and save it as:\n"
                "  earth_texture.jpg\n"
            )

        image = mpimg.imread(texture_path)

        return image

    # Figure construction
    def _build_figure(self):

        self.fig = plt.figure(figsize=(17, 9))

        # Main map
        self.ax = self.fig.add_axes(
            [0.04, 0.15, 0.68, 0.76]
        )

        # Mission status panel
        self.status_ax = self.fig.add_axes(
            [0.74, 0.08, 0.24, 0.84]
        )

        # Map
        self.ax.set_xlim(*cfg.MAP_LON_RANGE)
        self.ax.set_ylim(*cfg.MAP_LAT_RANGE)

        self.ax.set_aspect("equal")

        self.ax.set_xlabel(
            "Longitude (deg)",
            color="white",
            fontsize=10
        )

        self.ax.set_ylabel(
            "Latitude (deg)",
            color="white",
            fontsize=10
        )

        self.ax.tick_params(
            colors="white",
            labelsize=8
        )

        for spine in self.ax.spines.values():
            spine.set_color("white")

        # Earth image
        self.earth_image = self._load_earth_texture()

        self.ax.imshow(
            self.earth_image,
            extent=[
                cfg.MAP_LON_RANGE[0],
                cfg.MAP_LON_RANGE[1],
                cfg.MAP_LAT_RANGE[0],
                cfg.MAP_LAT_RANGE[1],
            ],
            aspect="auto",
            zorder=0,
            alpha=0.88,
            interpolation="bilinear",
        )

        # Grid / equator
        self.ax.axhline(
            0,
            color="white",
            linewidth=0.5,
            alpha=0.35,
            zorder=1
        )

        self.ax.grid(
            True,
            color="white",
            linewidth=0.3,
            alpha=0.22,
            zorder=1
        )

        # Ground stations
        for name, site in cfg.GROUND_STATIONS.items():

            point = self.ax.scatter(
                site["lon_deg"],
                site["lat_deg"],
                marker=site["marker"],
                s=150,
                color=site["color"],
                edgecolors="white",
                linewidths=1.2,
                zorder=10,
            )

            self.gs_points[name] = point

            label = self.ax.annotate(
                name.split(" (")[0],
                (
                    site["lon_deg"],
                    site["lat_deg"]
                ),
                textcoords="offset points",
                xytext=(7, 7),
                fontsize=8,
                color="white",
                fontweight="bold",
                zorder=10,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.55,
                    edgecolor="none",
                ),
            )

            self.gs_labels[name] = label

        # IoT target
        site = cfg.IOT_PAYLOAD_SITE

        self.iot_point = self.ax.scatter(
            site["lon_deg"],
            site["lat_deg"],
            marker=site["marker"],
            s=230,
            color=site["color"],
            edgecolors="white",
            linewidths=1.2,
            zorder=11,
        )

        self.ax.annotate(
            "Etosha NP",
            (
                site["lon_deg"],
                site["lat_deg"]
            ),
            textcoords="offset points",
            xytext=(8, -14),
            fontsize=8,
            color="white",
            fontweight="bold",
            zorder=11,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.55,
                edgecolor="none",
            ),
        )

        # H2Sat
        self.h2sat_point = self.ax.scatter(
            [cfg.H2SAT_LON_DEG],
            [0],
            marker="D",
            s=170,
            color="gold",
            edgecolors="white",
            linewidths=1.2,
            zorder=12,
        )

        self.ax.annotate(
            "H2Sat",
            (
                cfg.H2SAT_LON_DEG,
                0
            ),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="gold",
            fontweight="bold",
            zorder=12,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.55,
                edgecolor="none",
            ),
        )

        # Satellites
        for sat_name, elements in cfg.SATELLITES.items():

            point, = self.ax.plot(
                [],
                [],
                marker=elements["marker"],
                markersize=12,
                color=elements["color"],
                markeredgecolor="white",
                markeredgewidth=1.5,
                linestyle="None",
                zorder=20,
            )

            trail, = self.ax.plot(
                [],
                [],
                color=elements["color"],
                linewidth=1.5,
                alpha=0.65,
                zorder=15,
            )

            label = self.ax.annotate(
                "",
                (0, 0),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=9,
                color="white",
                fontweight="bold",
                zorder=21,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="black",
                    alpha=0.70,
                    edgecolor="none",
                ),
            )

            self.sat_points[sat_name] = point
            self.sat_trails[sat_name] = trail
            self.sat_labels[sat_name] = label

        # Title
        self.title_text = self.ax.text(
            0.5,
            1.025,
            "",
            transform=self.ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=15,
            color="white",
            fontweight="bold",
        )

        # Status panel
        self.status_ax.set_facecolor("#07111f")

        self.status_ax.set_xlim(0, 1)
        self.status_ax.set_ylim(0, 1)

        self.status_ax.set_xticks([])
        self.status_ax.set_yticks([])

        for spine in self.status_ax.spines.values():
            spine.set_color("white")
            spine.set_alpha(0.4)

        self.panel_title = self.status_ax.text(
            0.05,
            0.97,
            "MISSION STATUS",
            color="white",
            fontsize=13,
            fontweight="bold",
            va="top",
        )

        self.panel_time = self.status_ax.text(
            0.05,
            0.925,
            "",
            color="white",
            fontsize=10,
            family="monospace",
            va="top",
        )

        # Static legend
        legend_handles = []

        for mode, color in cfg.MODE_COLORS.items():

            legend_handles.append(
                mpatches.Patch(
                    color=color,
                    label=mode
                )
            )

        self.status_ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(0.04, 0.01),
            fontsize=8,
            framealpha=0.35,
            facecolor="#07111f",
            labelcolor="white",
        )

        self.fig.patch.set_facecolor("#07111f")

    # Longitude wrapping
    def _unwrap_lon_trail(self, lon_deg_array):

        lon = np.asarray(
            lon_deg_array,
            dtype=float
        )

        if len(lon) < 2:
            return lon

        diffs = np.diff(lon)

        breaks = np.where(
            np.abs(diffs) > 180
        )[0]

        if len(breaks) == 0:
            return lon

        result = lon.copy()

        for b in breaks:
            result[b + 1] = np.nan

        return result

    # Contact line helper
    def _set_contact_line(
        self,
        key,
        x1,
        y1,
        x2,
        y2,
        color,
        visible,
        linewidth=2.0,
        linestyle="-",
    ):

        if key not in self.contact_lines:

            line, = self.ax.plot(
                [],
                [],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=0.9,
                zorder=18,
            )

            self.contact_lines[key] = line

        line = self.contact_lines[key]

        if visible:

            line.set_data(
                [x1, x2],
                [y1, y2]
            )

            line.set_alpha(0.95)

        else:

            line.set_data(
                [],
                []
            )

            line.set_alpha(0.0)

    # Update
    def _update(self, frame_num):

        idx = self.frame_indices[frame_num]

        t_now = self.t[idx]

        # Title
        self.title_text.set_text(
            "GAIA-MISSION CONOPS  |  "
            f"T+{format_hms(t_now)}  /  "
            f"{format_hms(self.t[-1])}"
        )

        self.panel_time.set_text(
            f"SIMULATION TIME\n"
            f"T+ {format_hms(t_now)}"
        )

        # Satellites
        panel_y = 0.86

        for sat_name in cfg.SATELLITES:

            sat = self.timeline["sats"][sat_name]

            lat = float(
                sat["lat_deg"][idx]
            )

            lon = float(
                sat["lon_deg"][idx]
            )

            alt = float(
                sat["alt_km"][idx]
            )

            mode = sat["mode"][idx]

            sees_h2sat = bool(
                sat["sees_h2sat"][idx]
            )

            # Satellite marker
            mode_color = cfg.MODE_COLORS.get(
                mode,
                "white"
            )

            point = self.sat_points[sat_name]

            point.set_data(
                [lon],
                [lat]
            )

            point.set_markerfacecolor(
                mode_color
            )

            point.set_markeredgecolor(
                "white"
            )

            # Ground track
            trail_start = max(
                0,
                idx - self.trail_len_steps
            )

            trail_lon = self._unwrap_lon_trail(
                sat["lon_deg"][
                    trail_start:idx + 1
                ]
            )

            trail_lat = sat["lat_deg"][
                trail_start:idx + 1
            ]

            self.sat_trails[sat_name].set_data(
                trail_lon,
                trail_lat
            )

            # Satellite map label
            label_text = (
                f"{sat_name}\n"
                f"{mode}"
            )

            label = self.sat_labels[sat_name]

            label.set_text(label_text)

            label.set_position(
                (lon, lat)
            )

            label.set_color(
                mode_color
            )

            # Ground station contacts
            for gs_name, site in cfg.GROUND_STATIONS.items():

                visible = bool(
                    sat["visibility"][gs_name][idx]
                )

                key = (
                    f"{sat_name}_GS_{gs_name}"
                )

                self._set_contact_line(
                    key=key,
                    x1=lon,
                    y1=lat,
                    x2=site["lon_deg"],
                    y2=site["lat_deg"],
                    color=site["color"],
                    visible=visible,
                    linewidth=2.2,
                )

            # H2Sat contact
            h2key = (
                f"{sat_name}_H2SAT"
            )

            self._set_contact_line(
                key=h2key,
                x1=lon,
                y1=lat,
                x2=cfg.H2SAT_LON_DEG,
                y2=0,
                color="gold",
                visible=sees_h2sat,
                linewidth=2.5,
                linestyle="--",
            )

            # Status panel
            gs_lines = []

            for gs_name in cfg.GROUND_STATIONS:

                connected = bool(
                    sat["visibility"][gs_name][idx]
                )

                short_name = (
                    gs_name
                    .replace(" (UHF/VHF)", "")
                    .replace(" (Optical)", "")
                    .replace(" (Ka-band)", "")
                )

                if connected:
                    symbol = "●"
                else:
                    symbol = "○"

                gs_lines.append(
                    f"  {symbol} "
                    f"{short_name:<20} "
                    f"{'CONTACT' if connected else 'NO CONTACT'}"
                )

            h2_status = (
                "● VISIBLE / ISL AVAILABLE"
                if sees_h2sat
                else
                "○ NOT VISIBLE"
            )

            block = (
                f"{sat_name}\n"
                f"  MODE: {mode}\n"
                f"  LAT:  {lat:8.3f} deg\n"
                f"  LON:  {lon:8.3f} deg\n"
                f"  ALT:  {alt:8.1f} km\n"
                f"\n"
                f"  GROUND CONTACT\n"
                + "\n".join(gs_lines)
                + "\n\n"
                f"  H2SAT VISIBILITY\n"
                f"  {h2_status}"
            )

            # Create / update panel text
            if sat_name not in self.status_texts:

                text = self.status_ax.text(
                    0.05,
                    panel_y,
                    "",
                    transform=self.status_ax.transAxes,
                    color="white",
                    fontsize=8.5,
                    family="monospace",
                    va="top",
                    linespacing=1.35,
                )

                self.status_texts[sat_name] = text

            text = self.status_texts[sat_name]

            text.set_text(block)

            text.set_color("white")

            # Mode color gets a small colored line by changing title-ish
            # text color isn't possible per line, so use a colored box.
            text.set_bbox(
                dict(
                    boxstyle="round,pad=0.45",
                    facecolor="#0b1728",
                    edgecolor=mode_color,
                    linewidth=1.5,
                    alpha=0.92,
                )
            )

            panel_y -= 0.42

        # Collect artists for animation
        artists = []

        artists.extend(
            self.sat_points.values()
        )

        artists.extend(
            self.sat_trails.values()
        )

        artists.extend(
            self.sat_labels.values()
        )

        artists.extend(
            self.contact_lines.values()
        )

        artists.append(
            self.title_text
        )

        artists.append(
            self.panel_time
        )

        artists.extend(
            self.status_texts.values()
        )

        return artists

    # Run
    def run(self):

        self.anim = FuncAnimation(
            self.fig,
            self._update,
            frames=len(self.frame_indices),
            interval=cfg.ANIMATION_INTERVAL_MS,
            blit=False,
            repeat=True,
        )

        plt.show()

# Public entry point
def run_animation(timeline):

    animator = GaiaConopsAnimator(
        timeline
    )

    animator.run()