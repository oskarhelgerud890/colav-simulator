from __future__ import annotations

import csv
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml  # PyYAML

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim

import io
import re
import contextlib
import math
import copy
import matplotlib.pyplot as plt
import pandas as pd




from export_for_evaluator import export_episode_to_evaluator_csv

def extract_scores_from_result(result: dict) -> pd.DataFrame:
    """
    Extract per-vessel scores from the sweep_runner result dict.

    Tries common structures:
    - result["scores"] as dict/list
    - result["ship_scores"] etc.

    Returns: DataFrame with one row per ship and columns like Ship0_S_safety, etc.
    """
    # Common case: you already have a "summary row" dict in your code.
    # If not, we fall back to scanning keys that look like score names.
    if isinstance(result, dict):
        # If result already stores a per-ship dict:
        for k in ["scores", "ship_scores", "eval_scores", "metrics"]:
            if k in result and isinstance(result[k], (dict, list)):
                return pd.DataFrame(result[k]) if isinstance(result[k], list) else pd.DataFrame([result[k]])

        # Otherwise: flatten any keys that look like Ship{n}_S_*, Ship{n}_P_*, Ship{n}_C_*
        row = {k: v for k, v in result.items() if isinstance(k, str) and (k.startswith("Ship0_") or k.startswith("Ship1_"))}
        if row:
            return pd.DataFrame([row])

    # If nothing found:
    return pd.DataFrame()


def write_score_report_from_df(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if df.empty:
            f.write("No scores found.\n")
            return
        for _, row in df.iterrows():
            # print all columns in a readable way
            f.write("Scores\n")
            for col in df.columns:
                f.write(f"  {col:30s}: {row[col]}\n")
            f.write("\n")


def frange(start: float, stop: float, step: float) -> list[float]:
    if step == 0:
        raise ValueError("step cannot be 0")
    if (stop - start) * step < 0:
        raise ValueError("step has wrong sign for start/stop")
    vals = []
    x = start
    while (x <= stop + 1e-9) if step > 0 else (x >= stop - 1e-9):
        vals.append(x)
        x += step
    return vals


def safe_dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def safe_json(obj: Any) -> Any:
    """
    Make obj JSON-serializable without triggering dataclass __repr__ on partially-filled objects.
    """
    # 1) Primitive JSON types are fine
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}

    # 2) Numpy scalars/arrays (optional, safe)
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"_ndarray": True, "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, np.generic):
            return obj.item()
    except Exception:
        pass

    # 3) Dataclasses: use asdict (doesn't rely on __repr__)
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj):
            return {"_dataclass": obj.__class__.__name__, **safe_json(dataclasses.asdict(obj))}
    except Exception:
        # even asdict may fail if the object is "broken"; fall through
        pass

    # 4) Final fallback: don't call repr(obj)! (can trigger failing dataclass repr)
    return {"_type": obj.__class__.__name__}



@dataclass(frozen=True)
class Case:
    case_id: str
    ship1_start_east_m: float

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def set_waypoints_straight_from_heading(ship: dict, *, L: float = 4000.0) -> None:
    """
    Set ship['waypoints'] so LOS guidance will follow a straight line in the ship's heading direction.

    Assumes:
      ship['csog_state'] = [N, E, speed, heading_deg]
      ship['waypoints'] format is [[N0, N1], [E0, E1]]  (Northing list, Easting list)
      heading_deg is NAV convention: 0°=North, 90°=East (clockwise)
    """
    csog = ship.get("csog_state", None)
    if not csog or len(csog) < 4:
        raise ValueError("Ship missing csog_state with [N,E,speed,heading_deg]")

    N0 = float(csog[0])
    E0 = float(csog[1])
    h_deg = float(csog[3])
    h = math.radians(h_deg)

    N1 = N0 + L * math.cos(h)
    E1 = E0 + L * math.sin(h)

    ship["waypoints"] = [[N0, N1], [E0, E1]]

def place_ship1_on_circle(ship1: dict, *, center_N: float, center_E: float, R: float, angle_deg: float, wp_len: float) -> None:
    """
    Plasser ship1 på en sirkel rundt (center_N, center_E) og sett heading slik at den peker mot sentrum.
    Deretter settes waypoints rett fram i heading-retningen (så LOS ikke lager rare svinger).
    """
    a = math.radians(float(angle_deg))

    # Posisjon på sirkel
    N = center_N + R * math.cos(a)
    E = center_E + R * math.sin(a)

    # Heading inn mot sentrum (NAV: 0=N, 90=E, bruker atan2(dE, dN))
    dN = center_N - N
    dE = center_E - E
    heading_deg = (math.degrees(math.atan2(dE, dN)) % 360.0)

    cs = list(ship1.get("csog_state", []))
    if len(cs) < 4:
        raise ValueError("ship1 csog_state must be [N,E,speed,heading_deg]")

    cs[0] = float(N)
    cs[1] = float(E)
    cs[3] = float(heading_deg)
    ship1["csog_state"] = cs

    set_waypoints_straight_from_heading(ship1, L=float(wp_len))

def nav_heading_deg_from_dNE(dN: float, dE: float) -> float:
    """NAV heading: 0°=North, 90°=East."""
    return (math.degrees(math.atan2(dE, dN)) + 360.0) % 360.0


def patch_template_for_case(template: dict, case: Case, cfg: dict) -> dict:
    patched = copy.deepcopy(template)

    patched["type"] = "CR_GW"

    center_N, center_E = map(float, cfg["crossing_point_enu"])
    ship0_south = float(cfg.get("ship0_start_south_m", 1200.0))
    L0 = float(cfg.get("ship0_wp_length_m", 6000.0))
    L1 = float(cfg.get("ship1_wp_length_m", 6000.0))

    ship1_heading = float(cfg.get("ship1_heading_deg", 270.0))
    ship1_start_east_m = float(case.ship1_start_east_m)

    ship_list = patched["ship_list"]
    s0, s1 = ship_list[0], ship_list[1]

    # --- Ship0 ---
    cs0 = list(s0.get("csog_state", []))
    if len(cs0) < 4:
        raise ValueError("ship0 csog_state must be [N,E,speed,heading_deg]")
    cs0[0] = center_N - ship0_south
    cs0[1] = center_E
    cs0[3] = 0.0
    s0["csog_state"] = cs0
    set_waypoints_straight_from_heading(s0, L=L0)

    # --- Ship1 ---
    cs1 = list(s1.get("csog_state", []))
    if len(cs1) < 4:
        raise ValueError("ship1 csog_state must be [N,E,speed,heading_deg]")
    cs1[0] = center_N
    cs1[1] = center_E + ship1_start_east_m
    cs1[3] = ship1_heading
    s1["csog_state"] = cs1
    set_waypoints_straight_from_heading(s1, L=L1)

    return patched




def run_sim_and_export_csv(
    scenario_yaml: Path,
    out_csv: Path,
    *,
    show_plots: bool,
    liveplot: bool,
    new_load_of_map_data: bool,
) -> dict:
    scenario_generator = sg.ScenarioGenerator()
    scenario_data = scenario_generator.generate(
        config_file=scenario_yaml,
        new_load_of_map_data=new_load_of_map_data,
        save_scenario=False,
        show_plots=show_plots,
        n_episodes=1,
    )

    use_colav = True  # sett True når du vil ha COLAV igjen
    



    simulator = sim.Simulator()
    print("VIS CFG:",
      "show_results=", simulator.config.visualizer.show_results,
      "show_target_tracking_results=", simulator.config.visualizer.show_target_tracking_results,
      "show_trajectory_tracking_results=", simulator.config.visualizer.show_trajectory_tracking_results,
      "show_liveplot_colav_results=", simulator.config.visualizer.show_liveplot_colav_results,
      "save_result_figures=", simulator.config.visualizer.save_result_figures,
      "matplotlib_backend=", simulator.config.visualizer.matplotlib_backend)

    simulator.toggle_liveplot_visibility(liveplot)

    if use_colav:
        sbmpc_obj = ci.SBMPCWrapper()
        output = simulator.run([scenario_data], colav_systems=[(0, sbmpc_obj)])
    else:
        output = simulator.run([scenario_data])

    ep0 = output[0]["episode_simdata_list"][0]


    export_episode_to_evaluator_csv(ep0, out_csv)
    return ep0


def run_evaluator_on_csv(
    csv_path: Path,
    *,
    map_data_files: list[str],
    utm_zone: int,
    new_map_data_load: bool,
    out_json: Path,
) -> dict[str, object]:
    from colav_evaluation_tool.evaluator import Evaluator

    e = Evaluator()
    e.load_data_from_file(
        csv_path,
        map_data_files=map_data_files,
        utm_zone=utm_zone,
        new_map_data_load=new_map_data_load,
    )

    results = e.evaluate()

    scores_df = extract_scores_from_evaluator(e)

    out_dir = Path("outputs")  # eller case_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df.to_csv(out_dir / "scores.csv", index=False)
    scores_long = scores_to_long(scores_df)
    scores_long.to_csv(out_dir / "scores_long.csv", index=False)
    write_score_report(scores_df, out_dir / "scores.txt")



    # Extract scores robustly via print output
    scores = extract_scores_via_print(e, vessel_ids=[0, 1])

    payload = {
        "csv_path": str(csv_path),
        "results_type": results.__class__.__name__,
        "scores": scores,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2))
    return scores



def append_summary_row(summary_csv: Path, row: dict[str, Any]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = summary_csv.exists()
    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def capture_print_vessel_scores(e, vessel_id: int) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        e.print_vessel_scores(vessel_id=vessel_id)
    return buf.getvalue()

def parse_prettytable_scores(text: str) -> dict[str, object]:
    """
    Parse the score table printed by e.print_vessel_scores().

    - Parses numeric rows as floats
    - Also captures the string row: situation | CRGW / HO / CRSO / ...
    """
    out: dict[str, object] = {}

    # 1) Capture situation (string)
    sit_re = re.compile(r"\|\s*situation\s*\|\s*([A-Za-z0-9_]+)\s*\|", re.IGNORECASE)
    m = sit_re.search(text)
    if m:
        out["situation"] = m.group(1).strip()

    # 2) Capture numeric rows
    num_re = re.compile(
        r"\|\s*([A-Za-z0-9_ ]+?)\s*\|\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\|"
    )
    for m in num_re.finditer(text):
        key = m.group(1).strip()
        if key.lower() in {"ship 0", "ship 1"}:
            continue
        out[key] = float(m.group(2))

    return out


def extract_scores_via_print(e, vessel_ids: list[int]) -> dict[str, object]:
    """
    Returns flat dict with keys like Ship0_S_14, Ship1_P_delay, ...
    """
    flat: dict[str, float] = {}
    for vid in vessel_ids:
        txt = capture_print_vessel_scores(e, vid)
        d = parse_prettytable_scores(txt)
        for k, v in d.items():
            flat[f"Ship{vid}_{k}"] = v
    return flat


def extract_scores_from_evaluator(evaluator):
    """
    Returnerer en pandas DataFrame med én rad per skip
    og kolonner for alle S_, P_, C_-scorer.
    """
    rows = []

    for v in evaluator.vessels:
        row = {
            "vessel_id": v.id,
            "mmsi": getattr(v, "mmsi", None),
            "name": getattr(v, "name", f"Ship{v.id}"),
        }

        for attr in dir(v):
            if attr.startswith(("S_", "P_", "C_")):
                val = getattr(v, attr)
                # filtrer bort metoder og kallbare
                if callable(val):
                    continue
                # numpy -> float
                try:
                    import numpy as np
                    if isinstance(val, np.ndarray):
                        val = float(val)
                except Exception:
                    pass
                row[attr] = val

        rows.append(row)

    import pandas as pd
    return pd.DataFrame(rows)

def scores_to_long(df):
    records = []
    for _, row in df.iterrows():
        for col in df.columns:
            if col.startswith(("S_", "P_", "C_")):
                records.append({
                    "vessel_id": row["vessel_id"],
                    "mmsi": row["mmsi"],
                    "metric": col,
                    "value": row[col],
                })
    import pandas as pd
    return pd.DataFrame(records)


def write_score_report(df, path):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(f"Vessel {row['vessel_id']} (MMSI {row['mmsi']})\n")
            for col in df.columns:
                if col.startswith(("S_", "P_", "C_")):
                    f.write(f"  {col:25s}: {row[col]}\n")
            f.write("\n")

def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "experiments" / "configs" / "sweep_crossing_distance.yaml"
    cfg = yaml.safe_load(config_path.read_text())

    print("CONFIG crossing_point_enu:", cfg.get("crossing_point_enu"))
    print("CONFIG ship0_start_south_m:", cfg.get("ship0_start_south_m"))


    exp_name = cfg["experiment_name"]
    template_path = repo_root / cfg["template"]
    out_root = repo_root / cfg["out_dir"] / f"{exp_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_root.mkdir(parents=True, exist_ok=True)

    show_plots = bool(cfg.get("show_plots", False))
    liveplot = bool(cfg.get("liveplot", False))
    new_load_first = bool(cfg.get("new_load_of_map_data_first_case", True))

    run_eval = bool(cfg.get("run_evaluator", False))
    eval_cfg = cfg.get("evaluator", {}) if run_eval else {}
    map_data_files = eval_cfg.get("map_data_files", [])
    utm_zone = int(eval_cfg.get("utm_zone", 33))
    new_map_data_load = bool(eval_cfg.get("new_map_data_load", False))

    template = yaml.safe_load(template_path.read_text())

    sweep = cfg["sweep"]["ship1_start_east_m"]
    dists = frange(float(sweep["start"]), float(sweep["stop"]), float(sweep["step"]))

    cases: list[Case] = [
        Case(case_id=f"{i:04d}", ship1_start_east_m=d)
        for i, d in enumerate(dists, start=1)
    ]


    vis = cfg.get("visual_debug", {}) or {}
    vis_enabled = bool(vis.get("enabled", False))
    vis_case_id = str(vis.get("case_id", "0001"))

    if vis_enabled:
        cases = [c for c in cases if c.case_id == vis_case_id]
        liveplot = True
        show_plots = False

    (out_root / "cases.csv").write_text(
        "case_id,ship1_start_east_m\n" + "\n".join(f"{c.case_id},{c.ship1_start_east_m}" for c in cases)
    )



    summary_csv = out_root / "summary.csv"

    print(f"\n=== Running experiment: {exp_name} ===")
    print(f"Template: {template_path}")
    print(f"Output:   {out_root}")
    print(f"Cases:    {len(cases)}\n")

    for idx, case in enumerate(cases):
        case_dir = out_root / "runs" / f"case_{case.case_id}"
        scenario_out = case_dir / "scenario.yaml"
        csv_out = case_dir / "sim_for_eval.csv"
        eval_json = case_dir / "evaluator_results.json"

        case_dir.mkdir(parents=True, exist_ok=True)

        # Write patched scenario
        patched = patch_template_for_case(template, case, cfg)
        safe_dump_yaml(patched, scenario_out)

        new_load = new_load_first if idx == 0 else False

        print(f"[{idx+1:03d}/{len(cases):03d}] case {case.case_id} ship1_start_east_m={case.ship1_start_east_m:.1f} new_load={new_load}")



        csv_ok = False
        eval_ok = False

        try:
            _ep0 = run_sim_and_export_csv(
                scenario_out,
                csv_out,
                show_plots=show_plots,
                liveplot=liveplot,
                new_load_of_map_data=new_load,
            )
            csv_ok = csv_out.exists() and csv_out.stat().st_size > 0
        except Exception as ex:
            import traceback
            (case_dir / "error_sim.txt").write_text(traceback.format_exc())
            print(f"  !! SIM ERROR: {ex}")
            raise  # <-- viktig: da får du også traceback rett i terminalen


        scores: dict[str, float] = {}

        if run_eval and csv_ok:
            try:
                scores = run_evaluator_on_csv(
                    csv_out,
                    map_data_files=map_data_files,
                    utm_zone=utm_zone,
                    new_map_data_load=new_map_data_load,
                    out_json=eval_json,
                )
                eval_ok = True
            except Exception as ex:
                import traceback
                (case_dir / "error_eval.txt").write_text(traceback.format_exc())
                print(f"  !! EVAL ERROR: {ex}")


        row = {
            "case_id": case.case_id,
            "ship1_start_east_m": case.ship1_start_east_m,
            "scenario_yaml": str(scenario_out),
            "csv_path": str(csv_out) if csv_ok else "",
            "csv_ok": int(csv_ok),
            "eval_ok": int(eval_ok),
            "evaluator_json": str(eval_json) if eval_ok else "",
        }
        row.update(scores)
        append_summary_row(summary_csv, row)
        scores_df = extract_scores_from_result(row)
        scores_df.to_csv(case_dir / "scores.csv", index=False)
        write_score_report_from_df(scores_df, case_dir / "scores.txt")

    print("\n=== DONE ===")
    print("Summary:", summary_csv)
    plt.show(block=True)


if __name__ == "__main__":
    main()
