import json, numpy as np
from experiments.mouth_tuning import BASE
from experiments.run_screen import variant
rows = [json.loads(l) for l in open("experiments/out_screen.jsonl")]
base = [r for r in rows if r["name"] == "base"]
var = [r for r in rows if r["name"] != "base"]
keys = ["human", "vowels_voiced", "open_syllables", "reuse", "accuracy", "effort", "stop_share", "voiceless_obstruents", "syllables_per_word", "frames_per_syllable"]
val = lambda r, k: np.nan if r[k] is None else r[k]
print("measure                 baseline x20 (mean ± sd)    400 variants (mean ± sd, min..max)")
for k in keys:
    b, v = np.array([val(r, k) for r in base]), np.array([val(r, k) for r in var])
    print(f"{k:22s}  {np.nanmean(b):.2f} ± {np.nanstd(b):.2f}              {np.nanmean(v):.2f} ± {np.nanstd(v):.2f}   {np.nanmin(v):.2f}..{np.nanmax(v):.2f}")
names = list(BASE)
X = np.array([[np.log(variant(int(r["name"][1:]))[k] / BASE[k]) for k in names] for r in var])
X1 = np.column_stack([np.ones(len(X)), X])
for k in ["human", "vowels_voiced", "open_syllables", "accuracy", "frames_per_syllable", "voiceless_obstruents", "stop_share"]:
    y = np.array([val(r, k) for r in var]); ok = ~np.isnan(y)
    coef, *_ = np.linalg.lstsq(X1[ok], y[ok], rcond=None)
    resid = y[ok] - X1[ok] @ coef
    se = np.sqrt(resid.var(ddof=X1.shape[1]) * np.diag(np.linalg.inv(X1[ok].T @ X1[ok])))
    t = coef / se
    r2 = 1 - resid.var() / y[ok].var()
    top = sorted(zip(np.abs(t[1:]), names, coef[1:], t[1:]), reverse=True)[:6]
    print(f"\n{k}: R2 {r2:.2f}; strongest constants (change per e-fold, t):")
    print("   " + "; ".join(f"{n} {c:+.3f} (t={tt:+.1f})" for _, n, c, tt in top))
best = sorted(var, key=lambda r: -r["human"])[:12]
print("\ntop 12 by human score in the screen:")
for r in best:
    print(f"  {r['name']:5s} human {r['human']:.2f} voiced {r['vowels_voiced']:.2f} open {r['open_syllables']:.2f} reuse {r['reuse']:.2f} acc {r['accuracy']:.2f}")
json.dump([r["name"] for r in best], open("experiments/top.json", "w"))
