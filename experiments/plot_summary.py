from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


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
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def plot_one(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path) -> None:
    x = df[xcol].astype(float)
    y = df[ycol].astype(float)

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(f"{ycol} vs {xcol}")
    plt.grid(True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    runs_root = Path("experiments/runs")
    prefix = "sweep_angle_transition"

    run_dir = latest_run_dir(runs_root, prefix)
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Missing: {summary_csv}")

    df = pd.read_csv(summary_csv)

    # Basic sanity checks
    if "ship1_angle_deg" not in df.columns:
        raise SystemExit("summary.csv missing 'ship1_angle_deg' column")

    # Only keep rows that evaluated OK (optional but usually what you want)
    if "eval_ok" in df.columns:
        df = df[df["eval_ok"] == 1].copy()

    # Sort by heading for nice plots
    df = df.sort_values("ship1_angle_deg")

    plots_dir = run_dir / "plots"
    print("Run dir:", run_dir)
    print("Writing plots to:", plots_dir)

    # Choose which scores to plot.
    # You can edit these patterns whenever you add new metrics.
    patterns = [
        r"Ship0_S_safety",
        r"Ship0_S_14",
        r"Ship0_S_16",
        r"Ship0_P_delay",
        r"Ship0_P_14_nsb",
        r"Ship0_P_14_sts",
        r"Ship0_r_cpa",
        r"Ship1_S_safety",
        r"Ship1_S_14",
        r"Ship1_S_16",
        r"Ship1_P_delay",
        r"Ship1_r_cpa",
    ]
    ycols = pick_columns(df, patterns)

    if not ycols:
        # fallback: plot all Ship0_ and Ship1_ score columns, but cap
        score_cols = [c for c in df.columns if c.startswith("Ship0_") or c.startswith("Ship1_")]
        ycols = score_cols[:20]
        print("No pattern matches. Falling back to first 20 score cols:", ycols)

    for ycol in ycols:
        out_path = plots_dir / f"{ycol}.png"
        plot_one(df, "ship1_angle_deg", ycol, out_path)

    # Optional: combined plot for key metrics (handy for quick glance)
    key = [c for c in ["Ship0_S_safety", "Ship0_S_14", "Ship0_P_delay"] if c in df.columns]
    if key:
        plt.figure()
        x = df["ship1_angle_deg"].astype(float)
        for ycol in key:
            plt.plot(x, df[ycol].astype(float), marker="o", label=ycol)
        plt.xlabel("ship1_angle_deg")
        plt.ylabel("score")
        plt.title("Key metrics")
        plt.grid(True)
        plt.legend()
        plt.savefig(plots_dir / "key_metrics.png", dpi=200, bbox_inches="tight")
        plt.close()

    print("Done.")


if __name__ == "__main__":
    main()
