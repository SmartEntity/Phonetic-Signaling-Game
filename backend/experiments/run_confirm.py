import json
from experiments.mouth_tuning import BASE, run
from experiments.run_screen import variant

if __name__ == "__main__":
    top = json.load(open("experiments/top.json"))[:8]
    guided = {**BASE, "period_lift": BASE["period_lift"] * 0.6, "period_jaw": BASE["period_jaw"] * 1.6}
    sets = [("base", BASE), ("guided", guided)] + [(n, variant(int(n[1:]))) for n in top]
    run([(n, p, 500 + seed, 0.02, 200) for seed in range(5) for n, p in sets], "experiments/out_confirm.jsonl")
