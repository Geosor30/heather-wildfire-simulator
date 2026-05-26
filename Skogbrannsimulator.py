"""

Suppression sizing:
- (A) Full area
- (B1) Head strip: width * head_depth
- (B2) Head line/band: (fraction of perimeter) * head_depth
- (C) Discrete circular drops:

Plot:
- Fire gradient + optional overlays for B1, B2, and C
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches


# USER INPUTS

FUEL_TYPE = "heather"
TIME_MIN = 10.0

#Wind. Direction is where the fire head moves TOWARD.
OPEN_WIND_MPS = 4.0
WIND_DIR_DEG = 90.0        # degrees: 0=+x, 90=+y

# Terrain
SLOPE_DEG = 15.0

# Option for overwriting ROS
ROS_OVERRIDE = None

# Heather ROS model (Davies 1 / Davies 2)
USE_DAVIES_FOR_HEATHER = True
DAVIES_MODEL = "davies2"      # "davies1" or "davies2"
HEATHER_HEIGHT_M = 0.35       # heather height (m)
LIVE_MOISTURE_PERCENT = 55.0  

# Optional startup delay 
USE_RAMP = True
RAMP_MIN = 1.0                

# Wind adjustment factor
WAF = 0.35                     

# Slope scaling
USE_SLOPE = True
SLOPE_MULT_CAP = 5.0         

# Grid / plot settings
GRID_RES_M = 1.0
SAVE_PNG = True
OUT_PNG = "fire_footprint.png"


PLOT_MARGIN_M = 30.0

# Agent requirement settings 
COVERAGE_LEVEL = 4
AGENT_DENSITY_KG_PER_L = 1.00   # water-like

# Head knockdown: treated depth behind the head (meters)
HEAD_DEPTH_M = 4.0

# Head-line method: fraction of total perimeter considered "head+flanks band"
HEAD_PERIMETER_FRACTION = 0.30

# Optional: estimate sorties
DRONE_PAYLOAD_L = 20.0    

# Visual cue: make head edge “hotter” than backing edge (for direction)
USE_FORWARD_HOTNESS = True
BACK_EDGE_FACTOR = 0.5     # 0.0=back edge very light, 1.0=no difference

# Suppression methods (compute + print + plot)
USE_METHOD_B1_STRIP = True
USE_METHOD_B2_LINE = True
USE_METHOD_C_CIRCLES = False

SHOW_B1_OVERLAY = USE_METHOD_B1_STRIP
SHOW_B2_OVERLAY = USE_METHOD_B2_LINE
SHOW_CIRCLE_OVERLAY = USE_METHOD_C_CIRCLES

# C: Discrete circular drops settings 
CIRCLE_RADIUS_MODE = "payload"     # "fixed" or "payload"

# If "fixed"
CIRCLE_DROP_RADIUS_M = 2.0

# If "payload": liters represented by ONE circle/drop (usually same as DRONE_PAYLOAD_L)
CIRCLE_DROP_PAYLOAD_L = DRONE_PAYLOAD_L       # set to DRONE_PAYLOAD_L if you want them identical

# Choose how N and placement are defined:
CIRCLE_PATTERN_BASIS = "B2_LINE"  # "B1_WIDTH" or "B2_LINE"

# Optional inward offset for arc placement
CIRCLE_INWARD_OFFSET_M = None    

# Circle styling
CIRCLE_OVERLAY_ALPHA = 0.70
CIRCLE_FACE_COLOR = "none"        
CIRCLE_EDGE_COLOR = "black"
CIRCLE_EDGE_LW = 1.5

# B1 overlay styling
B1_OVERLAY_ALPHA = 0.50
B1_FACE_COLOR = "deepskyblue"
B1_EDGE_COLOR = "teal"
B1_EDGE_LW = 1.2

# B2 overlay styling
B2_OVERLAY_ALPHA = 0.50
B2_FACE_COLOR = "none"
B2_EDGE_COLOR = "black"
B2_EDGE_LW = 1.7

# Plot font sizing (A4 side-by-side readability)
AXIS_LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 18
TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 17
IGNITION_FONTSIZE = 18



# Fuel presets 

FUEL_PRESETS: Dict[str, Dict[str, float]] = {
    "heather": {"ros_m_per_min_default": 4.0},
    "grass": {"ros_m_per_min_default": 6.0},
    "pine_litter": {"ros_m_per_min_default": 2.0},
}



# Model functions

def wind_adjusted_midflame_wind(open_wind_mps: float, waf: float) -> float:
    if not (0.0 < waf <= 1.0):
        raise ValueError("WAF must be in (0, 1].")
    return max(0.0, open_wind_mps) * waf


def anderson_lw_ratio(midflame_wind_mph: float) -> float:
    u = max(0.0, midflame_wind_mph)
    lw = 0.936 * math.exp(0.2566 * u) + 0.461 * math.exp(-0.1548 * u) - 0.397
    return min(max(lw, 1.0), 8.0)


def slope_multiplier_vanwagner(slope_deg: float, cap: Optional[float] = 5.0) -> float:
    s = max(0.0, min(slope_deg, 31.0))
    t = math.tan(math.radians(s))
    mult = math.exp(3.533 * (t ** 1.2))
    if cap is not None:
        mult = min(mult, cap)
    return mult


def davies_ros_heather_m_per_min(
    model: str,
    wind_10_20m_mps: float,
    heather_height_m: float,
    live_moisture_percent: float
) -> float:
    U = max(0.0, wind_10_20m_mps)
    h = max(0.0, heather_height_m)
    M1 = max(0.0, live_moisture_percent)

    model = model.strip().lower()
    if model == "davies1":
        ros = 0.791 + 7.917 * (h ** 2) * U
    elif model == "davies2":
        ros = 8.304 + 7.286 * (h ** 2) * U - 0.097 * M1
    else:
        raise ValueError("DAVIES_MODEL must be 'davies1' or 'davies2'.")
    return max(0.0, ros)


def effective_head_distance(ros_steady_m_per_min: float, time_min: float,
                            use_ramp: bool, ramp_min: float) -> float:
    ros = max(0.0, ros_steady_m_per_min)
    T = max(0.0, time_min)

    if (not use_ramp) or ramp_min <= 0:
        return ros * T

    R = max(1e-9, ramp_min)
    if T <= R:
        return ros * (T * T) / (2.0 * R)
    else:
        return ros * (T - R / 2.0)


def ellipse_metrics_rear_focus(head_ros_m_per_min: float,
                               time_min: float,
                               lw_ratio: float,
                               use_ramp: bool,
                               ramp_min: float) -> Tuple[float, float, float, float, float, float, float]:
    """
    Returns:
      a, b, area, perimeter, center_shift_xprime, head_dist, back_dist
    """
    H = effective_head_distance(head_ros_m_per_min, time_min, use_ramp, ramp_min)

    lw = max(lw_ratio, 1.0)
    e = math.sqrt(max(0.0, 1.0 - 1.0 / (lw * lw)))

    # head distance H = a + c = a(1+e) -> a = H/(1+e)
    a = H / (1.0 + e + 1e-12)
    b = a / lw

    c = a * e
    center_shift = c

    area = math.pi * a * b

    # Ramanujan perimeter approximation
    hh = ((a - b) ** 2) / ((a + b) ** 2 + 1e-12)
    perimeter = math.pi * (a + b) * (1 + (3 * hh) / (10 + math.sqrt(4 - 3 * hh)))

    back_dist = a - c
    return a, b, area, perimeter, center_shift, H, back_dist


def rotate_to_wind_frame(X: np.ndarray, Y: np.ndarray, wind_dir_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    theta = math.radians(wind_dir_deg)
    xprime = X * math.cos(theta) + Y * math.sin(theta)
    yprime = -X * math.sin(theta) + Y * math.cos(theta)
    return xprime, yprime


def wind_to_world(xp: float, yp: float, wind_dir_deg: float) -> Tuple[float, float]:
    theta = math.radians(wind_dir_deg)
    x = xp * math.cos(theta) - yp * math.sin(theta)
    y = xp * math.sin(theta) + yp * math.cos(theta)
    return x, y


def ellipse_display_limits(a: float, center_shift: float, wind_dir_deg: float, margin_m: float
                          ) -> Tuple[float, float, float, float]:
    cx, cy = wind_to_world(center_shift, 0.0, wind_dir_deg)
    r = max(1.0, a)
    xmin = cx - r - margin_m
    xmax = cx + r + margin_m
    ymin = cy - r - margin_m
    ymax = cy + r + margin_m
    return xmin, xmax, ymin, ymax


def cl_to_liters_per_m2(cl: float) -> float:
    # 1 CL = 1 US gal / 100 ft^2 = 3.785 L / 9.2903 m^2 ≈ 0.407 L/m^2
    return 0.407 * cl


def agent_needed(area_m2: float, cl: float, density_kg_per_l: float) -> Tuple[float, float]:
    liters = area_m2 * cl_to_liters_per_m2(cl)
    kg = liters * density_kg_per_l
    return liters, kg


def radius_from_payload(payload_l: float, cl: float) -> float:
    """
    Compute effective circular footprint radius r such that one drop of payload_l liters
    applied at coverage level CL covers area A = payload / (0.407*CL).
    """
    if payload_l <= 0:
        raise ValueError("CIRCLE_DROP_PAYLOAD_L must be > 0 in payload mode.")
    L_per_m2 = cl_to_liters_per_m2(cl)
    area_m2 = payload_l / max(1e-12, L_per_m2)
    return math.sqrt(area_m2 / math.pi)


# Suppression geometry + circle placement


def b1_strip_polygon_world(head_dist: float, b: float, head_depth: float, wind_dir_deg: float) -> np.ndarray:
    d = max(0.0, head_depth)
    x1 = head_dist - d
    x2 = head_dist
    y1 = -b
    y2 = +b
    corners_wind = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float)
    corners_world = np.array([wind_to_world(xp, yp, wind_dir_deg) for xp, yp in corners_wind], dtype=float)
    return corners_world


def b2_head_band_polygon_world(a: float, b: float, center_shift: float,
                               head_fraction: float, head_depth: float,
                               wind_dir_deg: float, n: int = 220) -> Optional[np.ndarray]:
    frac = float(np.clip(head_fraction, 0.0, 1.0))
    if frac <= 0.0 or head_depth <= 0:
        return None

    ai = max(1e-6, a - head_depth)
    bi = max(1e-6, b - head_depth)

    span = frac * math.pi
    t = np.linspace(-span, +span, n)

    xo = center_shift + a * np.cos(t)
    yo = b * np.sin(t)

    xi = center_shift + ai * np.cos(t[::-1])
    yi = bi * np.sin(t[::-1])

    poly_wind = np.vstack([np.column_stack([xo, yo]), np.column_stack([xi, yi])])
    poly_world = np.array([wind_to_world(xp, yp, wind_dir_deg) for xp, yp in poly_wind], dtype=float)
    return poly_world


def circle_centers_across_width_wind_frame(head_dist: float,
                                           head_depth: float,
                                           r_m: float,
                                           n_needed: int) -> List[Tuple[float, float]]:
    """
    Place n circles across width (y-direction) in wind frame.
    Circles touch (center spacing = 2r) and row is centered about y'=0.
    x' is at mid-depth of B1: x' = H - d/2.
    """
    n = max(0, int(n_needed))
    if n == 0:
        return []
    r = max(1e-6, float(r_m))
    step = 2.0 * r
    x_center = head_dist - max(0.0, head_depth) / 2.0
    y0 = -0.5 * (n - 1) * step
    return [(x_center, float(y0 + i * step)) for i in range(n)]


def circle_centers_along_forward_arc_wind_frame(a: float, b: float, center_shift: float,
                                               head_fraction: float,
                                               inward_offset: float,
                                               n_needed: int) -> List[Tuple[float, float]]:
    """
    Place n circles along a forward ellipse arc (wind frame), approximating the B2 midline.
    Uses an inward-shrunk ellipse (a_i, b_i) so centers sit inside the band.
    """
    n = max(0, int(n_needed))
    if n == 0:
        return []

    frac = float(np.clip(head_fraction, 0.0, 1.0))
    span = frac * math.pi
    ai = max(1e-6, a - max(0.0, inward_offset))
    bi = max(1e-6, b - max(0.0, inward_offset))

   
    t = np.linspace(-span, +span, 2000)
    xs = center_shift + ai * np.cos(t)
    ys = bi * np.sin(t)

    ds = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    total = s[-1] if s[-1] > 0 else 1.0

    targets = np.linspace(0.0, total, n)
    centers: List[Tuple[float, float]] = []
    j = 0
    for tt in targets:
        while j < len(s)-1 and s[j] < tt:
            j += 1
        centers.append((float(xs[j]), float(ys[j])))
    return centers


def circle_centers_world_from_wind(centers_wind: List[Tuple[float, float]],
                                  wind_dir_deg: float) -> List[Tuple[float, float]]:
    return [wind_to_world(xp, yp, wind_dir_deg) for (xp, yp) in centers_wind]


# Plotting

def plot_fire_ellipse(intensity_masked: np.ma.MaskedArray,
                      extent: Tuple[float, float, float, float],
                      title: str,
                      out_png: Optional[str],
                      b1_poly: Optional[np.ndarray],
                      b2_poly: Optional[np.ndarray],
                      circle_centers: Optional[List[Tuple[float, float]]],
                      circle_r: Optional[float],
                      circle_label: Optional[str]) -> None:
    fire_cmap = LinearSegmentedColormap.from_list(
        "fire_grad",
        [(0.00, "#fffde7"), (0.55, "#fff176"), (1.00, "#ff0000")],
    )
    fire_cmap.set_bad((0, 0, 0, 0))

    xmin, xmax, ymin, ymax = extent

    fig = plt.figure()
    ax = plt.gca()
    ax.set_facecolor("#7a9c7c")

    ax.imshow(
        intensity_masked,
        origin="lower",
        extent=[xmin, xmax, ymin, ymax],
        interpolation="nearest",
        cmap=fire_cmap,
        vmin=0.0,
        vmax=1.0,
        zorder=1,
    )

    overlay_handles: List[mpatches.Patch] = []

    # ---- B1 strip ----
    if b1_poly is not None and SHOW_B1_OVERLAY:
        ax.add_patch(
            mpatches.Polygon(
                b1_poly,
                closed=True,
                fill=True,
                facecolor=B1_FACE_COLOR,
                edgecolor=B1_EDGE_COLOR,
                linewidth=B1_EDGE_LW,
                alpha=B1_OVERLAY_ALPHA,
                zorder=6
            )
        )
        overlay_handles.append(
            mpatches.Patch(
                facecolor=B1_FACE_COLOR,
                edgecolor=B1_EDGE_COLOR,
                linewidth=B1_EDGE_LW,
                alpha=B1_OVERLAY_ALPHA,
                label="B1 head strip"
            )
        )

    # ---- B2 head band ----
    if b2_poly is not None and SHOW_B2_OVERLAY:
        ax.add_patch(
            mpatches.Polygon(
                b2_poly,
                closed=True,
                fill=(B2_FACE_COLOR != "none"),
                facecolor=B2_FACE_COLOR,
                edgecolor=B2_EDGE_COLOR,
                linewidth=B2_EDGE_LW,
                alpha=B2_OVERLAY_ALPHA,
                zorder=5
            )
        )
        overlay_handles.append(
            mpatches.Patch(
                facecolor=B2_FACE_COLOR if (B2_FACE_COLOR != "none") else "none",
                edgecolor=B2_EDGE_COLOR,
                linewidth=B2_EDGE_LW,
                alpha=B2_OVERLAY_ALPHA,
                label="B2 head line/band"
            )
        )

    # ---- C circles ----
    if circle_centers and circle_r and SHOW_CIRCLE_OVERLAY and circle_r > 0:
        fill_circles = (CIRCLE_FACE_COLOR != "none")
        for (cx, cy) in circle_centers:
            ax.add_patch(
                mpatches.Circle(
                    (cx, cy),
                    radius=circle_r,
                    fill=fill_circles,
                    facecolor=CIRCLE_FACE_COLOR,
                    edgecolor=CIRCLE_EDGE_COLOR,
                    linewidth=CIRCLE_EDGE_LW,
                    alpha=CIRCLE_OVERLAY_ALPHA,
                    zorder=7
                )
            )
        if circle_label:
            overlay_handles.append(
                mpatches.Circle(
                    (0, 0),
                    radius=max(0.5, circle_r * 0.4),
                    fill=fill_circles,
                    facecolor=CIRCLE_FACE_COLOR,
                    edgecolor=CIRCLE_EDGE_COLOR,
                    linewidth=CIRCLE_EDGE_LW,
                    alpha=CIRCLE_OVERLAY_ALPHA,
                    label=circle_label
                )
            )

    active_patch = mpatches.Patch(color="#ff3300", label="Active edge")
    cool_patch = mpatches.Patch(color="#fffde7", label="Older interior")
    handles = [active_patch, cool_patch] + overlay_handles
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=LEGEND_FONTSIZE)

    ax.scatter([0], [0], s=30, marker="x", zorder=10, color="black")
    ax.text(0, 0, " Ignition", ha="left", va="bottom", zorder=10, fontsize=IGNITION_FONTSIZE)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("y (m)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=TITLE_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=TICK_LABEL_FONTSIZE)

    plt.tight_layout()
    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"\nSaved plot: {out_png}")
    plt.show()


# Run

@dataclass
class Result:
    ros_steady: float
    ros_used_for_growth: float
    lw_ratio: float
    a_m: float
    b_m: float
    center_shift_m: float
    head_dist_m: float
    back_dist_m: float
    area_m2: float
    perimeter_m: float
    width_m: float
    head_area_strip_m2: float
    head_line_length_m: float
    head_area_line_m2: float


def main() -> None:
    if FUEL_TYPE not in FUEL_PRESETS and ROS_OVERRIDE is None:
        raise ValueError(f"Unknown FUEL_TYPE '{FUEL_TYPE}'. Add it to FUEL_PRESETS or set ROS_OVERRIDE.")

    # Wind for shape (L/W)
    midflame_mps = wind_adjusted_midflame_wind(OPEN_WIND_MPS, WAF)
    midflame_mph = midflame_mps * 2.236936
    lw = anderson_lw_ratio(midflame_mph)

    # Steady ROS
    if ROS_OVERRIDE is not None:
        ros_steady = max(0.0, float(ROS_OVERRIDE))
        ros_source = "ROS_OVERRIDE"
    else:
        if FUEL_TYPE.lower() == "heather" and USE_DAVIES_FOR_HEATHER:
            ros_steady = davies_ros_heather_m_per_min(
                model=DAVIES_MODEL,
                wind_10_20m_mps=OPEN_WIND_MPS,
                heather_height_m=HEATHER_HEIGHT_M,
                live_moisture_percent=LIVE_MOISTURE_PERCENT,
            )
            ros_source = f"Davies ({DAVIES_MODEL})"
        else:
            ros_steady = float(FUEL_PRESETS[FUEL_TYPE]["ros_m_per_min_default"])
            ros_source = "Preset constant ROS"

    ros_for_growth = ros_steady
    if USE_SLOPE:
        ros_for_growth *= slope_multiplier_vanwagner(SLOPE_DEG, cap=SLOPE_MULT_CAP)

    a, b, area, perim, center_shift, head_dist, back_dist = ellipse_metrics_rear_focus(
        ros_for_growth, TIME_MIN, lw, USE_RAMP, RAMP_MIN
    )

    width = 2 * b
    length_total = head_dist + back_dist

    # B1 / B2 treated areas
    head_area_strip = width * max(0.0, HEAD_DEPTH_M)
    f = float(np.clip(HEAD_PERIMETER_FRACTION, 0.0, 1.0))
    head_line_length = f * perim
    head_area_line = head_line_length * max(0.0, HEAD_DEPTH_M)

    res = Result(
        ros_steady=ros_steady,
        ros_used_for_growth=ros_for_growth,
        lw_ratio=lw,
        a_m=a,
        b_m=b,
        center_shift_m=center_shift,
        head_dist_m=head_dist,
        back_dist_m=back_dist,
        area_m2=area,
        perimeter_m=perim,
        width_m=width,
        head_area_strip_m2=head_area_strip,
        head_line_length_m=head_line_length,
        head_area_line_m2=head_area_line,
    )

    strip_L, strip_kg = agent_needed(res.head_area_strip_m2, COVERAGE_LEVEL, AGENT_DENSITY_KG_PER_L)
    line_L, line_kg = agent_needed(res.head_area_line_m2, COVERAGE_LEVEL, AGENT_DENSITY_KG_PER_L)

    # --- C: radius from payload or fixed ---
    if CIRCLE_RADIUS_MODE.lower().strip() == "payload":
        r_drop = radius_from_payload(CIRCLE_DROP_PAYLOAD_L, COVERAGE_LEVEL)
        payload_info = f"{CIRCLE_DROP_PAYLOAD_L:.1f} L/drop"
    elif CIRCLE_RADIUS_MODE.lower().strip() == "fixed":
        r_drop = float(CIRCLE_DROP_RADIUS_M)
        payload_info = "fixed r"
    else:
        raise ValueError("CIRCLE_RADIUS_MODE must be 'fixed' or 'payload'.")

    circle_centers_world: Optional[List[Tuple[float, float]]] = None
    circle_label: Optional[str] = None
    n_circles = 0

    if USE_METHOD_C_CIRCLES and SHOW_CIRCLE_OVERLAY:
        if CIRCLE_PATTERN_BASIS.upper().strip() == "B1_WIDTH":
            n_circles = int(math.ceil(res.width_m / max(1e-9, (2.0 * r_drop))))
            centers_wind = circle_centers_across_width_wind_frame(
                head_dist=res.head_dist_m,
                head_depth=HEAD_DEPTH_M,
                r_m=r_drop,
                n_needed=n_circles
            )
            circle_centers_world = circle_centers_world_from_wind(centers_wind, WIND_DIR_DEG)
            circle_label = f"C: {payload_info}, r={r_drop:.2f} m, N={n_circles} (basis=B1 width)"

        elif CIRCLE_PATTERN_BASIS.upper().strip() == "B2_LINE":
            n_circles = int(math.ceil(res.head_line_length_m / max(1e-9, (2.0 * r_drop))))
            inward = (HEAD_DEPTH_M / 2.0) if (CIRCLE_INWARD_OFFSET_M is None) else float(CIRCLE_INWARD_OFFSET_M)
            centers_wind = circle_centers_along_forward_arc_wind_frame(
                a=res.a_m,
                b=res.b_m,
                center_shift=res.center_shift_m,
                head_fraction=HEAD_PERIMETER_FRACTION,
                inward_offset=inward,
                n_needed=n_circles
            )
            circle_centers_world = circle_centers_world_from_wind(centers_wind, WIND_DIR_DEG)
            circle_label = f"C: {payload_info}, r={r_drop:.2f} m, N={n_circles} (basis=B2 line)"
        else:
            raise ValueError("CIRCLE_PATTERN_BASIS must be 'B1_WIDTH' or 'B2_LINE'.")

   
    def sorties_required(volume_l: float) -> Optional[int]:
        if DRONE_PAYLOAD_L is None or DRONE_PAYLOAD_L <= 0:
            return None
        return int(math.ceil(volume_l / DRONE_PAYLOAD_L))

  
    print("\n=== Wildfire footprint result (rear-focus ellipse) ===")
    print(f"Fuel: {FUEL_TYPE} | T={TIME_MIN:.2f} min | Wind(open)={OPEN_WIND_MPS:.2f} m/s | Slope={SLOPE_DEG:.1f} deg")
    print(f"L/W={res.lw_ratio:.2f} | ROS model: {ros_source} | ROS used: {res.ros_used_for_growth:.2f} m/min")
    print(f"H={res.head_dist_m:.1f} m | W={res.width_m:.1f} m | Area={res.area_m2/10_000:.3f} ha | P≈{res.perimeter_m:.1f} m")

    print("\n=== Suppression sizing (ideal on-target) ===")
    print(f"CL={COVERAGE_LEVEL} -> {cl_to_liters_per_m2(COVERAGE_LEVEL):.3f} L/m² | density={AGENT_DENSITY_KG_PER_L:.2f} kg/L")

    if USE_METHOD_B1_STRIP:
        print("\n(B1) Head strip:")
        print(f"  Area: {res.head_area_strip_m2:.1f} m² | Volume: {strip_L:.1f} L | Mass: {strip_kg:.1f} kg")
        s = sorties_required(strip_L)
        if s is not None:
            print(f"  Sorties @ {DRONE_PAYLOAD_L:.1f} L: {s}")

    if USE_METHOD_B2_LINE:
        print("\n(B2) Head line/band:")
        print(f"  f={HEAD_PERIMETER_FRACTION:.2f} | Length: {res.head_line_length_m:.1f} m | Area: {res.head_area_line_m2:.1f} m²")
        print(f"  Volume: {line_L:.1f} L | Mass: {line_kg:.1f} kg")
        s = sorties_required(line_L)
        if s is not None:
            print(f"  Sorties @ {DRONE_PAYLOAD_L:.1f} L: {s}")

    if USE_METHOD_C_CIRCLES and SHOW_CIRCLE_OVERLAY:
        print("\n(C) Discrete drops (pattern control):")
        print(f"  Radius mode: {CIRCLE_RADIUS_MODE} | r_drop={r_drop:.2f} m | {payload_info}")
        print(f"  Pattern basis: {CIRCLE_PATTERN_BASIS} | N={n_circles}")
        if CIRCLE_RADIUS_MODE.lower().strip() == "payload":
            print(f"  Total dropped volume (N * payload): {n_circles*CIRCLE_DROP_PAYLOAD_L:.1f} L")

    xmin, xmax, ymin, ymax = ellipse_display_limits(
        a=res.a_m,
        center_shift=res.center_shift_m,
        wind_dir_deg=WIND_DIR_DEG,
        margin_m=PLOT_MARGIN_M,
    )

    xs = np.arange(xmin, xmax + GRID_RES_M, GRID_RES_M)
    ys = np.arange(ymin, ymax + GRID_RES_M, GRID_RES_M)
    X, Y = np.meshgrid(xs, ys)

    
    xprime, yprime = rotate_to_wind_frame(X, Y, WIND_DIR_DEG)
    x_centered = xprime - res.center_shift_m
    y_centered = yprime

    r2 = (x_centered / res.a_m) ** 2 + (y_centered / res.b_m) ** 2
    inside = r2 <= 1.0
    base_intensity = np.sqrt(np.clip(r2, 0.0, 1.0))

    if USE_FORWARD_HOTNESS:
        x_head = res.center_shift_m + res.a_m
        x_back = res.center_shift_m - res.a_m
        w = (xprime - x_back) / max(1e-9, (x_head - x_back))
        w = np.clip(w, 0.0, 1.0)
        forward_weight = BACK_EDGE_FACTOR + (1.0 - BACK_EDGE_FACTOR) * w
        intensity = base_intensity * forward_weight
    else:
        intensity = base_intensity

    intensity_masked = np.ma.array(intensity, mask=~inside)

    b1_poly = b1_strip_polygon_world(res.head_dist_m, res.b_m, HEAD_DEPTH_M, WIND_DIR_DEG) if (SHOW_B1_OVERLAY and USE_METHOD_B1_STRIP) else None
    b2_poly = b2_head_band_polygon_world(res.a_m, res.b_m, res.center_shift_m,
                                         HEAD_PERIMETER_FRACTION, HEAD_DEPTH_M, WIND_DIR_DEG) if (SHOW_B2_OVERLAY and USE_METHOD_B2_LINE) else None

    title = (f"Ellipse | t={TIME_MIN:.1f} min | ROS={res.ros_used_for_growth:.2f} m/min | "
             f"wind={OPEN_WIND_MPS:.1f} m/s | C basis={CIRCLE_PATTERN_BASIS}")

    plot_fire_ellipse(
        intensity_masked=intensity_masked,
        extent=(xmin, xmax, ymin, ymax),
        title=title,
        out_png=OUT_PNG if SAVE_PNG else None,
        b1_poly=b1_poly,
        b2_poly=b2_poly,
        circle_centers=circle_centers_world,
        circle_r=r_drop if (USE_METHOD_C_CIRCLES and SHOW_CIRCLE_OVERLAY) else None,
        circle_label=circle_label
    )


if __name__ == "__main__":
    main()
