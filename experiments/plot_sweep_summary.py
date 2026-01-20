from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import matplotlib.pyplot as plt


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
]


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
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}")
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

    print("Done.")


if __name__ == "__main__":
    main()
