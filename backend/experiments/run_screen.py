import json, sys
import numpy as np
from experiments.mouth_tuning import BASE, run

def variant(i):
    rng = np.random.default_rng([2026, i])
    p = {k: float(v * np.exp(rng.uniform(-0.5, 0.5))) for k, v in BASE.items()}
    for k in p:
        if k.startswith("damping"):
            p[k] = min(p[k], 1.2)
    p["f2_top"], p["noise_top"] = min(p["f2_top"], 0.95), min(p["noise_top"], 1.0)
    p["f1_range"] = min(p["f1_range"], 0.95 - p["f1_base"])
    return p

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "baseline":
        jobs = [("base", BASE, seed, c, 200) for c in (0.0, 0.02, 0.1) for seed in range(6)]
        jobs += [("random-words", BASE, seed, 0.02, 0) for seed in range(6)]
        run(jobs, "experiments/out_baseline.jsonl")
    if stage == "screen":
        quick = dict(children=18, repeats=8)
        jobs = [("base", BASE, 100 + seed, 0.02, 80, quick) for seed in range(20)]
        jobs += [(f"v{i}", variant(i), 0, 0.02, 80, quick) for i in range(int(sys.argv[2]))]
        run(jobs, "experiments/out_screen.jsonl")
