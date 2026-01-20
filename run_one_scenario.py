from __future__ import annotations

from pathlib import Path
import numpy as np

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim

from export_for_evaluator import export_episode_to_evaluator_csv


def check_motion_matches_cog(v, name: str) -> None:
    """
    Sjekker om faktisk bevegelsesretning stemmer med COG, i NAV-konvensjon:
      0 rad = North, +pi/2 = East, pi = South, -pi/2 = West.
    """
    xy = np.asarray(v.xy, dtype=float)  # (2, N)
    x = xy[0, :]
    y = xy[1, :]

    dx = np.diff(x)
    dy = np.diff(y)

    # NAV-bæring: 0 = nord, + = medurs
    theta_nav = np.arctan2(dx, dy)            # rad, lengde N-1
    cog = np.asarray(v.cog, dtype=float)[1:]  # align

    # Vinkelfeil wrap til [-pi, pi]
    err = (theta_nav - cog + np.pi) % (2 * np.pi) - np.pi

    print(f"\n{name}: motion vs COG (NAV convention)")
    print(f"  mean |err| = {np.mean(np.abs(err)):.4f} rad")
    print(f"  max  |err| = {np.max(np.abs(err)):.4f} rad")
    print(f"  first 5 theta_nav (deg): {np.rad2deg(theta_nav[:5])}")
    print(f"  first 5 cog       (deg): {np.rad2deg(cog[:5])}")

    # Fart fra posisjon vs SOG
    ts = np.asarray(v.timestamps, dtype=float)
    if len(ts) > 1:
        speed_from_pos = np.hypot(dx, dy) / np.diff(ts)
        sog = np.asarray(v.sog, dtype=float)[1:]
        print(f"  speed_from_pos first 5: {speed_from_pos[:5]}")
        print(f"  sog            first 5: {sog[:5]}")


def diagnose_motion_vs_cog(v, name: str) -> None:
    cog = np.asarray(v.cog, dtype=float)

    # Heuristikk: hvis cog ser ut som grader (store tall), konverter til rad
    if np.nanmax(np.abs(cog)) > 2 * np.pi + 0.5:
        cog = np.deg2rad(cog)
        cog_unit = "deg->rad"
    else:
        cog_unit = "rad"

    xy = np.asarray(v.xy, dtype=float)

    # To mulige akse-rekkefølger
    axis_cases = {
        "xy=(E,N)": (xy[0], xy[1]),
        "xy=(N,E)": (xy[1], xy[0]),  # swap
    }

    # To vanlige COG-konvensjoner:
    # A) NAV: 0 = North, +CW  (East = -90deg)
    # B) MATH: 0 = East, +CCW
    def fwd_nav(cog_rad):
        fx = -np.sin(cog_rad)  # east
        fy = np.cos(cog_rad)   # north
        return fx, fy

    def fwd_math(cog_rad):
        fx = np.cos(cog_rad)  # east
        fy = np.sin(cog_rad)  # north
        return fx, fy

    conv_cases = {
        "COG=nav(0=N,+CW)": fwd_nav,
        "COG=math(0=E,+CCW)": fwd_math,
        "COG=nav_alt(0=N,+CCW)": lambda c: (np.sin(c), np.cos(c)),
    }

    print(f"\n=== {name}: diagnose (cog interpreted as {cog_unit}) ===")

    for axis_label, (E, N) in axis_cases.items():
        dE = np.diff(E)
        dN = np.diff(N)

        speed = np.hypot(dE, dN)
        uE = dE / (speed + 1e-12)
        uN = dN / (speed + 1e-12)

        # Align cog to diff
        c = cog[1:]

        for conv_label, fwd_fun in conv_cases.items():
            fE, fN = fwd_fun(c)
            dot = uE * fE + uN * fN

            print(
                f"{axis_label:10s} | {conv_label:20s}  "
                f"dot(min/mean/max)={dot.min(): .3f}/{dot.mean(): .3f}/{dot.max(): .3f}  "
                f"frac(dot<0)={(dot < 0).mean() * 100:5.1f}%"
            )


def summarize_episode(ep0: dict) -> None:
    vessel_data = ep0["vessel_data"]
    sim_data = ep0["sim_data"]
    ship_info = ep0["ship_info"]

    print("\n=== Summary ===")
    print("Num vessels:", len(vessel_data))
    print("ship_info keys:", list(ship_info.keys()))
    print("sim_data shape:", sim_data.shape, "columns:", list(sim_data.columns))

    # Tid (bruk timestamps – datetimes_utc kan ha dubletter)
    ts = np.asarray(vessel_data[0].timestamps, dtype=float)
    if len(ts) > 1:
        dts = np.diff(ts)
        print(
            f"time: N={len(ts)}, t0={ts[0]:.1f}s, t_end={ts[-1]:.1f}s, "
            f"dt(min/mean/max)={dts.min():.3f}/{dts.mean():.3f}/{dts.max():.3f}"
        )

    # Start/slutt per skip
    for v in vessel_data:
        xy = np.asarray(v.xy, dtype=float)
        x0, y0 = xy[0, 0], xy[1, 0]
        x1, y1 = xy[0, -1], xy[1, -1]
        print(
            f"{v.name}: start=({x0:.1f},{y0:.1f}) end=({x1:.1f},{y1:.1f}) "
            f"sog0={v.sog[0]:.2f} m/s cog0={v.cog[0]:.4f} rad"
        )

    # Plot-extent-hint
    all_xy = np.hstack([np.asarray(v.xy, dtype=float) for v in vessel_data])
    xmin, xmax = all_xy[0].min(), all_xy[0].max()
    ymin, ymax = all_xy[1].min(), all_xy[1].max()
    pad = 50.0
    print(f"plot extent suggestion: x[{xmin-pad:.1f}, {xmax+pad:.1f}]  y[{ymin-pad:.1f}, {ymax+pad:.1f}]")

    # SOG / COG første steg + kritisk sjekk
    for v in vessel_data:
        print(f"{v.name} sog first 10:", v.sog[:10])
        print(f"{v.name} cog first 10:", v.cog[:10])
        check_motion_matches_cog(v, v.name)


def main() -> None:
    # === Velg scenario her ===
    scenario_file = Path("scenarios/crossing_give_way.yaml")
    # scenario_file = Path("scenarios/crossing_give_way.yaml")

    # === Eksportmål (absolutt path, så du alltid finner fila) ===
    out_csv = Path.home() / "colav-simulator" / "sim_for_eval.csv"

    # Sett True hvis du vil stoppe programmet rett etter eksport (nyttig for debugging)
    stop_after_export = False

    scenario_generator = sg.ScenarioGenerator()
    scenario_data = scenario_generator.generate(
        config_file=scenario_file,
        new_load_of_map_data=True,   # sett False når ENC er cached
        save_scenario=False,
        show_plots=False,
        n_episodes=1,
    )

    sbmpc_obj = ci.SBMPCWrapper()
    simulator = sim.Simulator()
    simulator.toggle_liveplot_visibility(True)

    output = simulator.run([scenario_data], colav_systems=[(0, sbmpc_obj)])
    ep0 = output[0]["episode_simdata_list"][0]

    # === Eksport til evaluator-format ===
    export_episode_to_evaluator_csv(ep0, out_csv)
    print("EXPORT DONE  ->", out_csv)
    print("EXPORT EXISTS->", out_csv.exists())
    print("EXPORT SIZE  ->", out_csv.stat().st_size if out_csv.exists() else None)

    if stop_after_export:
        raise SystemExit("Stopping after export (intentional)")

    # === Diagnostikk / oppsummering ===
    diagnose_motion_vs_cog(ep0["vessel_data"][0], "Ship0")
    diagnose_motion_vs_cog(ep0["vessel_data"][1], "Ship1")
    summarize_episode(ep0)


if __name__ == "__main__":
    main()
