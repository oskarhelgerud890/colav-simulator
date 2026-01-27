from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json


DEFAULT_X_CANDIDATES = [
    # angle sweeps
    "ship1_angle_deg",
    "angle_deg",
    "encounter_angle_deg",
    # distance / start-position sweeps
    "ship1_start_east_m",
    "ship1_start_north_m",
    "ship0_start_east_m",
    "ship0_start_north_m",
    "start_range_m",
    "initial_range_m",
    # speed sweeps
    "ship0_speed_knots",
    "ship1_speed_knots",
    "speed_knots",
]


DEFAULT_PATTERNS = [
    # Common evaluation outputs (adjust as you add metrics)
    r"Ship0_S_safety",
    r"Ship0_S_14",
    r"Ship0_S_15",
    r"Ship0_S_16",
    r"Ship0_S_17",
    r"Ship0_P_delay",
    r"Ship0_P_14_nsb",
    r"Ship0_P_14_sts",
    r"Ship0_P_ahead15",
    r"Ship0_P_ahead16",
    r"Ship0_r_cpa",
    r"Ship1_S_safety",
    r"Ship1_S_14",
    r"Ship1_S_15",
    r"Ship1_S_16",
    r"Ship1_S_17",
    r"Ship1_P_delay",
    r"Ship1_P_ahead15",
    r"Ship1_P_ahead16",
    r"Ship1_r_cpa",
    r"true_r_cpa",
]

def _lonlat_to_utm_xy_arrays(lon: pd.Series, lat: pd.Series, utm_zone: int):
    from pyproj import Transformer
    epsg = 32600 + int(utm_zone)
    tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    x, y = tr.transform(lon.to_numpy(), lat.to_numpy())
    return x, y

def compute_true_cpa_from_sim_csv(sim_csv: Path, *, utm_zone: int, tol_s: float) -> float:
    df = read_csv_auto_sep(sim_csv)

    needed = {"mmsi", "lon", "lat", "date_time_utc"}
    missing = needed - set(df.columns)
    if missing:
        return float("nan")

    mmsis = sorted(df["mmsi"].unique().tolist())
    if len(mmsis) < 2:
        return float("nan")

    d0 = df[df["mmsi"] == mmsis[0]].copy()
    d1 = df[df["mmsi"] == mmsis[1]].copy()

    d0["t"] = pd.to_datetime(d0["date_time_utc"], errors="coerce")
    d1["t"] = pd.to_datetime(d1["date_time_utc"], errors="coerce")
    d0 = d0.dropna(subset=["t", "lon", "lat"]).sort_values("t")
    d1 = d1.dropna(subset=["t", "lon", "lat"]).sort_values("t")
    if d0.empty or d1.empty:
        return float("nan")

    try:
        x0, y0 = _lonlat_to_utm_xy_arrays(d0["lon"], d0["lat"], utm_zone)
        x1, y1 = _lonlat_to_utm_xy_arrays(d1["lon"], d1["lat"], utm_zone)
    except Exception:
        return float("nan")

    a0 = pd.DataFrame({"t": d0["t"].to_numpy(), "x0": x0, "y0": y0})
    a1 = pd.DataFrame({"t": d1["t"].to_numpy(), "x1": x1, "y1": y1})

    merged = pd.merge(a0, a1, on="t", how="inner")
    if merged.empty:
        merged = pd.merge_asof(
            a0.sort_values("t"),
            a1.sort_values("t"),
            on="t",
            direction="nearest",
            tolerance=pd.Timedelta(f"{tol_s}s"),
        ).dropna()

    if merged.empty:
        return float("nan")

    dx = merged["x0"] - merged["x1"]
    dy = merged["y0"] - merged["y1"]
    dist = np.sqrt(dx * dx + dy * dy)
    return float(dist.min())


def latest_run_dir(runs_root: Path, prefix: str) -> Path:
    runs = sorted(runs_root.glob(f"{prefix}_*"), key=lambda p: p.name)
    if not runs:
        raise SystemExit(f"No runs found under {runs_root} with prefix '{prefix}_'")
    return runs[-1]


def pick_columns(df: pd.DataFrame, patterns: list[str]) -> list[str]:
    cols: list[str] = []
    for pat in patterns:
        rx = re.compile(pat)
        cols.extend([c for c in df.columns if rx.fullmatch(c)])

    # keep order, unique
    seen = set()
    out: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def choose_xcol(df: pd.DataFrame, xcol: Optional[str], candidates: list[str]) -> str:
    if xcol:
        if xcol not in df.columns:
            raise SystemExit(f"summary.csv missing requested x column: '{xcol}'")
        return xcol

    for c in candidates:
        if c in df.columns:
            return c

    # last resort: first numeric-ish column
    for c in df.columns:
        if c in ("eval_ok",):
            continue
        try:
            pd.to_numeric(df[c].dropna().iloc[:10])
            return c
        except Exception:
            pass

    raise SystemExit(
        "Could not auto-select x column. Provide --x <colname>. "
        f"Available columns: {list(df.columns)}"
    )


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def plot_one(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path) -> None:
    x = to_numeric_safe(df[xcol])
    y = to_numeric_safe(df[ycol])
    mask = x.notna() & y.notna()

    plt.figure()
    plt.plot(x[mask], y[mask], marker="o")
    plt.xlabel(xcol)
    ytitle = "CPA" if ycol == "true_r_cpa" else ycol
    plt.ylabel(ytitle)
    plt.title(f"{ytitle} vs {xcol}")
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_combined(df: pd.DataFrame, xcol: str, ycols: Iterable[str], out_path: Path, title: str) -> None:
    x = to_numeric_safe(df[xcol])

    plt.figure()
    for ycol in ycols:
        if ycol not in df.columns:
            continue
        y = to_numeric_safe(df[ycol])
        mask = x.notna() & y.notna()
        plt.plot(x[mask], y[mask], marker="o", label=ycol)

    plt.xlabel(xcol)
    plt.ylabel("score")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def _find_case_id_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["case_id", "case", "case_idx", "case_index"]:
        if c in df.columns:
            return c
    return None


def _infer_case_dir(run_dir: Path, row: pd.Series) -> Path:
    # 1) If summary already has a path
    for c in ["case_dir", "run_case_dir", "case_path", "path"]:
        if c in row.index and isinstance(row[c], str) and row[c]:
            p = Path(row[c])
            return p if p.is_absolute() else (run_dir / p)

    # 2) Build runs/case_XXXX
    cid_col = _find_case_id_col(pd.DataFrame([row]))
    if cid_col is not None:
        cid_raw = str(row[cid_col]).strip()
        cid_raw = cid_raw.replace("case_", "")
        # tolerate floats like "1.0"
        try:
            cid_int = int(float(cid_raw))
            cid = f"{cid_int:04d}"
        except Exception:
            cid = cid_raw
            if cid.isdigit():
                cid = f"{int(cid):04d}"
        return run_dir / "runs" / f"case_{cid}"

    # 3) fallback: use row position
    return run_dir / "runs" / f"case_{int(row.name)+1:04d}"


def _lonlat_to_utm_xy(lon: pd.Series, lat: pd.Series, utm_zone: int):
    """
    Convert lon/lat (EPSG:4326) to UTM (EPSG:326xx). Falls back to lon/lat if pyproj missing.
    Returns x, y, unit_label.
    """
    try:
        from pyproj import Transformer
        epsg = 32600 + int(utm_zone)
        tr = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
        x, y = tr.transform(lon.to_numpy(), lat.to_numpy())
        return pd.Series(x), pd.Series(y), "m"
    except Exception:
        return lon, lat, "deg"

def read_csv_auto_sep(path: Path) -> pd.DataFrame:
    # autodetect ; vs , ved å sjekke header-linja
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)

def plot_trajectory_case(
    sim_csv: Path,
    out_path: Path,
    *,
    utm_zone: int,
    title: str,
    encounter: str | None = None,
    traj_window_m: float | None = None,
    traj_center: str = "ship0_start",  # beholdes for kompat, men brukes ikke når vi hardkoder senter
    traj_pad_m: float = 0.0,
) -> None:
    if not sim_csv.exists():
        raise FileNotFoundError(f"Missing sim csv: {sim_csv}")

    df = read_csv_auto_sep(sim_csv)

    needed = {"mmsi", "lon", "lat", "date_time_utc"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"{sim_csv} missing columns: {sorted(missing)}")

    mmsis = sorted(df["mmsi"].unique().tolist())
    if len(mmsis) < 2:
        raise SystemExit(f"Expected >=2 MMSI in {sim_csv}, got {mmsis}")

    d0 = df[df["mmsi"] == mmsis[0]].copy()
    d1 = df[df["mmsi"] == mmsis[1]].copy()

    # --- parse time for CPA computation ---
    d0["t"] = pd.to_datetime(d0["date_time_utc"], errors="coerce")
    d1["t"] = pd.to_datetime(d1["date_time_utc"], errors="coerce")
    d0 = d0.dropna(subset=["t", "lon", "lat"]).sort_values("t")
    d1 = d1.dropna(subset=["t", "lon", "lat"]).sort_values("t")
    if d0.empty or d1.empty:
        raise SystemExit(f"Empty track after parsing time/lon/lat in {sim_csv}")

    # lon/lat -> UTM (meters) (or deg fallback)
    x0, y0, unit = _lonlat_to_utm_xy(d0["lon"], d0["lat"], utm_zone)
    x1, y1, unit2 = _lonlat_to_utm_xy(d1["lon"], d1["lat"], utm_zone)
    if unit2 != unit:
        unit = unit  # should not happen, but keep whatever

    # --- align on time and compute CPA (min distance) ---
    a0 = pd.DataFrame({"t": d0["t"].to_numpy(), "x0": x0.to_numpy(), "y0": y0.to_numpy()})
    a1 = pd.DataFrame({"t": d1["t"].to_numpy(), "x1": x1.to_numpy(), "y1": y1.to_numpy()})

    merged = pd.merge(a0, a1, on="t", how="inner")
    if merged.empty:
        merged = pd.merge_asof(
            a0.sort_values("t"),
            a1.sort_values("t"),
            on="t",
            direction="nearest",
            tolerance=pd.Timedelta("1s"),
        ).dropna()

    if merged.empty:
        raise SystemExit(f"Could not align Ship0 and Ship1 time series to compute CPA for {sim_csv}")

    dx = merged["x0"] - merged["x1"]
    dy = merged["y0"] - merged["y1"]
    dist = (dx * dx + dy * dy) ** 0.5

    i_cpa = int(dist.idxmin())
    d_cpa = float(dist.loc[i_cpa])

    x0_cpa = float(merged.loc[i_cpa, "x0"])
    y0_cpa = float(merged.loc[i_cpa, "y0"])
    x1_cpa = float(merged.loc[i_cpa, "x1"])
    y1_cpa = float(merged.loc[i_cpa, "y1"])

    # ---------------------------
    # Plot in RELATIVE coords
    # ---------------------------
    if unit == "m":
        x_ref = float(x0.iloc[0])
        y_ref = float(y0.iloc[0])

        x0p = x0 - x_ref
        y0p = y0 - y_ref
        x1p = x1 - x_ref
        y1p = y1 - y_ref

        x0_cpa_p = x0_cpa - x_ref
        y0_cpa_p = y0_cpa - y_ref
        x1_cpa_p = x1_cpa - x_ref
        y1_cpa_p = y1_cpa - y_ref
    else:
        # deg fallback: keep as-is
        x0p, y0p, x1p, y1p = x0, y0, x1, y1
        x0_cpa_p, y0_cpa_p, x1_cpa_p, y1_cpa_p = x0_cpa, y0_cpa, x1_cpa, y1_cpa

    fig, ax = plt.subplots()

    def _plot_intended_line(ax, x: pd.Series, y: pd.Series, *, color: str):
        k = min(10, len(x))
        if k < 2:
            return

        x_start, y_start = float(x.iloc[0]), float(y.iloc[0])
        dx0 = float(x.iloc[k - 1] - x.iloc[0])
        dy0 = float(y.iloc[k - 1] - y.iloc[0])
        norm = (dx0 * dx0 + dy0 * dy0) ** 0.5
        if norm < 1e-9:
            return
        ux, uy = dx0 / norm, dy0 / norm

        L = float(((x.iloc[-1] - x.iloc[0]) ** 2 + (y.iloc[-1] - y.iloc[0]) ** 2) ** 0.5)
        ax.plot(
            [x_start, x_start + L * ux],
            [y_start, y_start + L * uy],
            "--",
            linewidth=1.5,
            color=color,
            alpha=0.5,
            label="_nolegend_",
        )

    # Intended / nominal straight paths (dashed)
    _plot_intended_line(ax, x0p, y0p, color="purple")
    _plot_intended_line(ax, x1p, y1p, color="red")

    # Trajectories (no legend handles)
    ax.plot(x0p, y0p, linewidth=2.0, color="purple", label="_nolegend_")
    ax.plot(x1p, y1p, linewidth=2.0, color="red", label="_nolegend_")

    # Start markers (legend)
    ax.scatter([x0p.iloc[0]], [y0p.iloc[0]], s=70, marker="o", color="purple", label="Ship0 start")
    ax.scatter([x1p.iloc[0]], [y1p.iloc[0]], s=70, marker="o", color="red", label="Ship1 start")

    # End markers (no legend)
    ax.scatter([x0p.iloc[-1]], [y0p.iloc[-1]], s=70, marker="s", color="purple", label="_nolegend_")
    ax.scatter([x1p.iloc[-1]], [y1p.iloc[-1]], s=70, marker="s", color="red", label="_nolegend_")

    # Small arrows at start
    if len(x0p) >= 2 and len(x1p) >= 2:
        dx0a = float(x0p.iloc[1] - x0p.iloc[0])
        dy0a = float(y0p.iloc[1] - y0p.iloc[0])
        dx1a = float(x1p.iloc[1] - x1p.iloc[0])
        dy1a = float(y1p.iloc[1] - y1p.iloc[0])

        ax.quiver(
            [x0p.iloc[0]], [y0p.iloc[0]], [dx0a], [dy0a],
            angles="xy", scale_units="xy", scale=1, width=0.004, color="purple"
        )
        ax.quiver(
            [x1p.iloc[0]], [y1p.iloc[0]], [dx1a], [dy1a],
            angles="xy", scale_units="xy", scale=1, width=0.004, color="red"
        )

    # CPA markers (one legend entry)
    ax.scatter([x0_cpa_p], [y0_cpa_p], s=140, marker="x", color="purple", linewidths=3, label="CPA")
    ax.scatter([x1_cpa_p], [y1_cpa_p], s=140, marker="x", color="red", linewidths=3, label="_nolegend_")

    # CPA connecting line
    ax.plot(
        [x0_cpa_p, x1_cpa_p],
        [y0_cpa_p, y1_cpa_p],
        linestyle=":",
        linewidth=2.0,
        color="black",
        label="_nolegend_",
    )

    # Annotation
    text = f"d_CPA={d_cpa:.1f} m"
    if encounter:
        text += f"\nencounter: {encounter}"
    ax.text(
        0.02, 0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.6, pad=0.3),
    )

    # ---------------------------
    # Axis limits / zoom (SINGLE SOURCE OF TRUTH)
    # ---------------------------
    if traj_window_m is not None and unit == "m":
        # ---------------------------
        # FIXED AXIS WINDOW (EDIT HERE)
        # ---------------------------
        FIXED_CENTER_X = 0.0
        FIXED_CENTER_Y = 1200.0

        FIXED_X_SPAN = 3000.0   # total width in meters (x-axis)
        FIXED_Y_SPAN = 3000.0   # total height in meters (y-axis)

        if traj_window_m is not None and unit == "m":
            hx = 0.5 * FIXED_X_SPAN
            hy = 0.5 * FIXED_Y_SPAN

            ax.set_autoscale_on(False)
            ax.set_xlim(FIXED_CENTER_X - hx, FIXED_CENTER_X + hx)
            ax.set_ylim(FIXED_CENTER_Y - hy, FIXED_CENTER_Y + hy)

            # Keep metric correctness
            ax.set_aspect("equal", adjustable="box")
    else:
        # auto limits + optional padding
        if traj_pad_m and unit == "m":
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(xmin - traj_pad_m, xmax + traj_pad_m)
            ax.set_ylim(ymin - traj_pad_m, ymax + traj_pad_m)

        # minimum x-span
        xmin, xmax = ax.get_xlim()
        min_span = 200.0
        if (xmax - xmin) < min_span:
            cx = 0.5 * (xmin + xmax)
            ax.set_xlim(cx - min_span / 2, cx + min_span / 2)

        # aspect
        if encounter == "HO":
            ax.set_aspect("auto")
        else:
            ax.set_aspect("equal", adjustable="box")

    ax.grid(True)

    if unit == "m":
        ax.set_xlabel("ΔEasting [m] (relative to Ship0 start)")
        ax.set_ylabel("ΔNorthing [m] (relative to Ship0 start)")
    else:
        ax.set_xlabel("Longitude [deg]")
        ax.set_ylabel("Latitude [deg]")

    ax.set_title(title)

    # Deduplicate legend entries
    handles, labels = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels):
        if l == "_nolegend_" or l.strip() == "":
            continue
        if l not in uniq:
            uniq[l] = h
    ax.legend(uniq.values(), uniq.keys(), loc="upper right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)





def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot evaluation sweep summary.csv outputs (generic across sweep types)."
    )

    group = p.add_mutually_exclusive_group(required=False)
    group.add_argument(
        "--run-dir",
        type=Path,
        help="Explicit run directory (containing summary.csv). Overrides --runs-root/--prefix.",
    )
    group.add_argument(
        "--prefix",
        type=str,
        help="Run prefix under --runs-root, e.g. 'sweep_angle_transition' or 'sweep_crossing_distance'.",
    )

    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("experiments/runs"),
        help="Root directory containing runs (default: experiments/runs)",
    )

    p.add_argument(
        "--summary",
        type=str,
        default="summary.csv",
        help="Summary filename inside run dir (default: summary.csv)",
    )

    p.add_argument(
        "--x",
        type=str,
        default=None,
        help="Column name for x-axis. If omitted, script auto-detects a sensible one.",
    )

    p.add_argument(
        "--keep-failed",
        action="store_true",
        help="Include rows where eval_ok != 1 (default filters them out if eval_ok exists).",
    )

    p.add_argument(
        "--max-cols",
        type=int,
        default=30,
        help="Max number of y-columns to plot if falling back to generic Ship* columns (default: 30)",
    )

    p.add_argument(
        "--pattern",
        action="append",
        default=None,
        help=(
            "Regex (fullmatch) for y-columns to plot. Repeatable. "
            "If omitted, uses a default metric list."
        ),
    )

    p.add_argument(
        "--plots-subdir",
        type=str,
        default="plots",
        help="Subdirectory under run dir to write plots (default: plots)",
    )

        # ---- NEW: trajectory plot from per-case sim_for_eval.csv ----
    p.add_argument(
        "--traj",
        action="store_true",
        help="Also create a trajectory plot from runs/case_XXXX/sim_for_eval.csv for one selected case.",
    )
    
        # ---- NEW: trajectory zoom/window ----
    p.add_argument(
        "--traj-window-m",
        type=float,
        default=None,
        help=(
            "If set, force a fixed square window size (meters) for trajectory plots. "
            "E.g. 800 => show 800m x 800m around chosen center."
        ),
    )
    p.add_argument(
        "--traj-center",
        type=str,
        default="ship0_start",
        choices=["ship0_start", "CPA", "midpoint"],
        help="Center point for --traj-window-m (default: ship0_start).",
    )
    p.add_argument(
        "--traj-pad-m",
        type=float,
        default=0.0,
        help="Extra padding (meters) added on top of auto limits (default: 0).",
    )


    p.add_argument(
        "--true-cpa",
        action="store_true",
        help="Compute geometric/true CPA from each case's sim_for_eval.csv and add column true_r_cpa.",
    )
    p.add_argument(
        "--true-cpa-tol-s",
        type=float,
        default=1.0,
        help="Time alignment tolerance in seconds for true CPA (default: 1.0).",
    )


    p.add_argument(
    "--traj-all",
    action="store_true",
    help="Create trajectory plots for ALL cases in the summary (after filtering).",
)

    p.add_argument(
        "--traj-pick",
        type=str,
        default="CPA",
        choices=["CPA", "best_safety", "first", "last", "index"],
        help="Which case to use for --traj (default: CPA if Ship0_S_safety exists).",
    )
    p.add_argument(
        "--traj-index",
        type=int,
        default=0,
        help="Row index to use if --traj-pick=index (0-based after filtering/sorting).",
    )
    p.add_argument(
        "--sim-csv-name",
        type=str,
        default="sim_for_eval.csv",
        help="Filename inside each case directory (default: sim_for_eval.csv).",
    )
    p.add_argument(
        "--utm-zone",
        type=int,
        default=33,
        help="UTM zone used if converting lon/lat to meters (default: 33).",
    )


    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve run directory:
    # 1) --run-dir (explicit)
    # 2) --prefix (latest matching under runs_root)
    # 3) default: latest run directory under runs_root
    if args.run_dir is not None:
        run_dir = args.run_dir
    elif args.prefix:
        run_dir = latest_run_dir(args.runs_root, args.prefix)
    else:
        runs_root = args.runs_root
        candidates = [p for p in runs_root.iterdir() if p.is_dir()]
        if not candidates:
            raise SystemExit(f"No run directories found under: {runs_root}")
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)

    summary_csv = run_dir / args.summary
    if not summary_csv.exists():
        raise SystemExit(f"Missing: {summary_csv}")

    df = pd.read_csv(summary_csv)
    print("SUMMARY COLUMNS:", list(df.columns))


    print("Computing true geometric CPA from sim_for_eval.csv ...")

    true_vals = []
    for _, row in df.iterrows():
        case_dir = _infer_case_dir(run_dir, row)
        sim_csv = case_dir / args.sim_csv_name
        if sim_csv.exists():
            v = compute_true_cpa_from_sim_csv(
                sim_csv,
                utm_zone=args.utm_zone,
                tol_s=1.0,   # eller args.true_cpa_tol_s hvis du vil beholde fleksibilitet
            )
        else:
            v = float("nan")
        true_vals.append(v)

    df["true_r_cpa"] = true_vals

    print(
        "true_r_cpa: computed for",
        df["true_r_cpa"].notna().sum(),
        "of",
        len(df),
        "cases",
    )



    if (not args.keep_failed) and ("eval_ok" in df.columns):
        df = df[df["eval_ok"] == 1].copy()

    xcol = choose_xcol(df, args.x, DEFAULT_X_CANDIDATES)

    # Sort by x (numeric when possible) for nicer plots
    df = df.copy()
    df["__xnum"] = to_numeric_safe(df[xcol])
    if df["__xnum"].notna().any():
        df = df.sort_values("__xnum")
    else:
        df = df.sort_values(xcol)

    plots_dir = run_dir / args.plots_subdir
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Run dir:", run_dir)
    print("Summary:", summary_csv)
    print("x-axis:", xcol)
    print("Writing plots to:", plots_dir)

    patterns = args.pattern if args.pattern else DEFAULT_PATTERNS
    ycols = pick_columns(df, patterns)

    if not ycols:
        # fallback: plot all Ship0_/Ship1_ columns (cap)
        score_cols = [c for c in df.columns if c.startswith("Ship0_") or c.startswith("Ship1_")]
        ycols = score_cols[: args.max_cols]
        print("No pattern matches. Falling back to first columns:", ycols)

    for ycol in ycols:
        if ycol not in df.columns:
            continue
        out_path = plots_dir / f"{ycol}.png"
        plot_one(df, xcol, ycol, out_path)

    # Combined plot for a quick glance
    combined_candidates = [
        "Ship0_S_safety",
        "Ship0_S_14",
        "Ship0_S_15",
        "Ship0_S_16",
        "Ship0_P_delay",
        "Ship0_r_cpa",
    ]
    combined = [c for c in combined_candidates if c in df.columns]
    if combined:
        plot_combined(df, xcol, combined, plots_dir / "key_metrics.png", "Key metrics")

        # ---- trajectory plot ----
    if args.traj_all:
        print(f"Creating trajectory plots for {len(df)} cases...")
        for i, row in enumerate(df.itertuples(index=True), start=1):
            row_s = df.loc[row.Index]  # Series

            case_dir = _infer_case_dir(run_dir, row_s)
            sim_csv = case_dir / args.sim_csv_name

            # encounter label
            encounter = None
            for c in ["Ship0_situation", "Ship1_situation", "situation"]:
                if c in row_s.index and pd.notna(row_s[c]) and str(row_s[c]).strip():
                    encounter = str(row_s[c]).strip()
                    break

            # filename includes case_id if available
            cid = row_s.get("case_id", None)
            if pd.notna(cid):
                cid_str = f"{int(float(cid)):04d}" if str(cid).replace(".","",1).isdigit() else str(cid)
            else:
                cid_str = f"{i:04d}"

            out_traj = plots_dir / f"trajectory_case_{cid_str}.png"

            xval = row_s.get(xcol, None)
            title = f"Trajectories (case {cid_str})"
            if xval is not None and str(xval) != "nan":
                title += f", {xcol}={xval}"
            if encounter:
                title += f" — {encounter}"

            try:
                plot_trajectory_case(
                    sim_csv,
                    out_traj,
                    utm_zone=args.utm_zone,
                    title=title,
                    encounter=encounter,
                    traj_window_m=args.traj_window_m,
                    traj_center=args.traj_center,
                    traj_pad_m=args.traj_pad_m,
                )

                print(f"[{i:03d}/{len(df):03d}] Wrote: {out_traj}")
            except Exception as e:
                print(f"[{i:03d}/{len(df):03d}] Trajectory plot failed: {e}")
                print("  Tried:", sim_csv)

    elif args.traj:
        # (din eksisterende single-case kode)

        pick_df = df  # already filtered + sorted
        pick_row = None
        tag = args.traj_pick

        if args.traj_pick in ("CPA", "best_safety") and ("Ship0_S_safety" in pick_df.columns):
            s = to_numeric_safe(pick_df["Ship0_S_safety"])
            if s.notna().any():
                idx = s.idxmin() if args.traj_pick == "CPA" else s.idxmax()
                pick_row = pick_df.loc[idx]
            else:
                tag = "first"
        if pick_row is None:
            if args.traj_pick == "last":
                pick_row = pick_df.iloc[-1]
            elif args.traj_pick == "index":
                i = int(args.traj_index)
                i = max(0, min(i, len(pick_df) - 1))
                pick_row = pick_df.iloc[i]
                tag = f"index_{i}"
            else:
                pick_row = pick_df.iloc[0]
                tag = "first"

        case_dir = _infer_case_dir(run_dir, pick_row)
        sim_csv = case_dir / args.sim_csv_name

        # Make a useful filename
        out_traj = plots_dir / f"trajectory_{tag}.png"

        # Add angle/distance value in title if available
        xval = pick_row.get(xcol, None)
        title = f"Trajectories ({tag})"
        # Try to extract encounter / situation type

        encounter = None
        for c in ["Ship0_situation", "Ship1_situation", "situation"]:
            if c in pick_row.index and pd.notna(pick_row[c]) and str(pick_row[c]).strip() != "":
                encounter = str(pick_row[c]).strip()
                break


        print("Picked encounter:", encounter)
        try:
            plot_trajectory_case(
                sim_csv,
                out_traj,
                utm_zone=args.utm_zone,
                title=title,
                encounter=encounter,
                traj_window_m=args.traj_window_m,
                traj_center=args.traj_center,
                traj_pad_m=args.traj_pad_m,
            )

            print("Wrote:", out_traj)
        except Exception as e:
            print("Trajectory plot failed:", e)
            print("Tried:", sim_csv)





    print("Done.")


if __name__ == "__main__":
    main()
