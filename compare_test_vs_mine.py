from pathlib import Path
import numpy as np

import colav_simulator.core.colav.colav_interface as ci
import colav_simulator.scenario_generator as sg
import colav_simulator.simulator as sim


def get_ep0_from_output(output):
    scenario_out = output[0]
    ep_list = scenario_out.get("episode_simdata_list", None)
    if ep_list is None or len(ep_list) == 0:
        raise KeyError(f"Missing episode_simdata_list. Keys: {list(scenario_out.keys())}")
    ep0 = ep_list[0]
    for k in ["vessel_data", "sim_data", "ship_info"]:
        if k not in ep0:
            raise KeyError(f"ep0 missing {k}. Keys: {list(ep0.keys())}")
    return ep0


def forward_dot(v, xy_order="EN", cog_conv="nav_alt", cog_offset=0.0):
    """
    Return dot-product stats between velocity direction and "forward" direction implied by COG.
    xy_order: 'EN' means v.xy[0]=E, v.xy[1]=N ; 'NE' swaps.
    cog_conv:
      - nav_alt: 0=N, +CCW, (E,N) forward = (sin(cog), cos(cog))
      - nav:     0=N, +CW,  (E,N) forward = (-sin(cog), cos(cog))
      - math:    0=E, +CCW, (E,N) forward = (cos(cog), sin(cog))
    """
    xy = np.asarray(v.xy, float)
    if xy_order == "EN":
        E, N = xy[0], xy[1]
    else:
        N, E = xy[0], xy[1]

    dE, dN = np.diff(E), np.diff(N)
    speed = np.hypot(dE, dN) + 1e-12
    uE, uN = dE / speed, dN / speed

    cog = np.asarray(v.cog, float)[1:] + cog_offset

    if cog_conv == "nav_alt":       # 0=N, +CCW
        fE, fN = np.sin(cog), np.cos(cog)
    elif cog_conv == "nav":         # 0=N, +CW
        fE, fN = -np.sin(cog), np.cos(cog)
    elif cog_conv == "math":        # 0=E, +CCW
        fE, fN = np.cos(cog), np.sin(cog)
    else:
        raise ValueError("Unknown cog_conv")

    dot = uE * fE + uN * fN
    return {
        "dot_min": float(dot.min()),
        "dot_mean": float(dot.mean()),
        "dot_max": float(dot.max()),
        "frac_backwards": float((dot < 0).mean()),
    }


def diagnose_episode(ep0, title):
    vd = ep0["vessel_data"]
    print(f"\n================ {title} ================")
    for i, v in enumerate(vd):
        # Den kombinasjonen du fant som ga perfekt match for Ship1 i ditt tilfelle:
        base = forward_dot(v, xy_order="EN", cog_conv="nav_alt", cog_offset=0.0)
        flip = forward_dot(v, xy_order="EN", cog_conv="nav_alt", cog_offset=np.pi)

        xy = np.asarray(v.xy, float)
        E0, N0 = float(xy[0, 0]), float(xy[1, 0])
        E1, N1 = float(xy[0, -1]), float(xy[1, -1])

        print(f"\n{v.name} (idx {i})")
        print(f"  start EN=({E0:.1f},{N0:.1f})  end EN=({E1:.1f},{N1:.1f})")
        print(f"  cog0={float(v.cog[0]): .4f} rad   sog0={float(v.sog[0]):.3f} m/s")
        print(f"  nav_alt offset 0 : dot(min/mean/max)={base['dot_min']:+.3f}/{base['dot_mean']:+.3f}/{base['dot_max']:+.3f}  frac_back={100*base['frac_backwards']:.1f}%")
        print(f"  nav_alt offset pi: dot(min/mean/max)={flip['dot_min']:+.3f}/{flip['dot_mean']:+.3f}/{flip['dot_max']:+.3f}  frac_back={100*flip['frac_backwards']:.1f}%")


def run_configured_scenarios():
    sbmpc_obj = ci.SBMPCWrapper()
    scenario_generator = sg.ScenarioGenerator()
    scenario_data_list = scenario_generator.generate_configured_scenarios()

    simulator = sim.Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run(scenario_data_list, colav_systems=[(0, sbmpc_obj)])
    return get_ep0_from_output(output)


def run_my_file(path: Path):
    sbmpc_obj = ci.SBMPCWrapper()
    scenario_generator = sg.ScenarioGenerator()
    scenario_data = scenario_generator.generate(
        config_file=path,
        new_load_of_map_data=True,
        save_scenario=False,
        show_plots=False,
        n_episodes=1,
    )
    simulator = sim.Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run([scenario_data], colav_systems=[(0, sbmpc_obj)])
    return get_ep0_from_output(output)


def main():
    ep_test = run_configured_scenarios()
    diagnose_episode(ep_test, "TEST_SIMULATOR / generate_configured_scenarios()")

    ep_mine = run_my_file(Path("scenarios/my_minimal_cr.yaml"))
    diagnose_episode(ep_mine, "MY_MINIMAL_CR / run_one_scenario style")


if __name__ == "__main__":
    main()
