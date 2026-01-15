from pathlib import Path
from colav_evaluation_tool.evaluator import Evaluator

runs_root = Path("experiments/runs")
runs = sorted(runs_root.glob("sweep_angle_transition_*"), key=lambda p: p.name)
latest = runs[-1]
csv_path = latest / "runs" / "case_0001" / "sim_for_eval.csv"

print("Latest run:", latest)
print("CSV:", csv_path)

e = Evaluator()
e.load_data_from_file(
    csv_path,
    map_data_files=["More_og_Romsdal_utm33.gdb"],
    utm_zone=33,
    new_map_data_load=False
)
res = e.evaluate()

print("\nevaluate() returned:", type(res))

# Check containers
for name in ["rdata", "results", "vessels", "vessel_list", "vessel_data_list"]:
    print(f"has e.{name}:", hasattr(e, name))

# rdata score-like attributes
if hasattr(e, "rdata"):
    r = e.rdata
    score_like = [a for a in dir(r) if a.startswith(("S_", "P_", "C_"))]
    print("\nScore-like attrs on e.rdata (first 150):")
    print(score_like[:150])

# vessels score-like attributes
if hasattr(e, "vessels"):
    vs = e.vessels
    print("\nlen(e.vessels):", len(vs))
    v0 = vs[0]
    v_score_like = [a for a in dir(v0) if a.startswith(("S_", "P_", "C_")) or "score" in a.lower()]
    print("Score-like attrs on vessel0 (first 150):")
    print(v_score_like[:150])
