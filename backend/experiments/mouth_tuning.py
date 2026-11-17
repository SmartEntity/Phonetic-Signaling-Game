"""
Which words are best for a given mouth, and how human do they look?

For one set of mouth constants this searches (no neural nets) for a small lexicon
whose words survive the channel best for the least effort, then describes the
winning words with util/ipa and scores them against human sound patterns.
"""
import json
import sys
from collections import Counter
from multiprocessing import Pool

import numpy as np

from util import ipa
from world.anatomy.voicebox import VoiceBox
from world.datatypes import Posture, Sound

REST = VoiceBox.REST
NAMES = ["jaw", "lift", "place", "breath", "voice"]

BASE = {
    **{f"period_{n}": v for n, v in zip(NAMES, VoiceBox.PERIOD)},
    **{f"damping_{n}": v for n, v in zip(NAMES, VoiceBox.DAMPING)},
    **{f"mass_{n}": v for n, v in zip(NAMES, VoiceBox.MASS)},
    "reach": 1.5, "shut": 0.03, "press_build": 0.5, "press_release": 0.8, "press_leak": 0.05, "burst_sharp": 8.0,
    "leak_gap": 0.15, "whisper": 0.1, "hiss_centre": 0.08, "hiss_width": 0.06, "hiss_onset": 0.02,
    "f1_base": 0.1, "f1_range": 0.45, "f1_slope": 2.0, "f2_top": 0.9, "f2_range": 0.5,
    "peak_voiced": 0.09, "peak_whisper": 0.2, "f2_gain": 0.8, "noise_top": 0.95, "noise_range": 0.6, "noise_width": 0.15,
}
CENTRES = VoiceBox.CENTRES
WALL, SUBSTEPS = VoiceBox.WALL, VoiceBox.SUBSTEPS

WORDS, SLOTS, HOLD = 16, 6, 4
WOBBLE, SILENCE_FRAMES, SILENCE_LEVEL = 0.05, 8, 0.05
NOISE, MAX_DELAY, RATE, GAIN = 0.1, 6, 0.15, 0.2
PIECES = 4


def vec(p, prefix):
    return np.array([p[f"{prefix}_{n}"] for n in NAMES])


def opening(x):
    return np.logaddexp(0.0, 30.0 * (x[..., 0] - x[..., 1])) / 30.0


def sound(p, x, burst):
    gap = opening(x)
    place, breath, voice = (np.clip(x[..., i], 0.0, 1.0) for i in (2, 3, 4))
    gets_out = gap / (gap + p["leak_gap"])
    voiced = breath * voice * gets_out
    whisper = p["whisper"] * breath * (1 - voice) * gets_out
    hiss = breath * np.exp(-((gap - p["hiss_centre"]) / p["hiss_width"]) ** 2) * np.tanh(gap / p["hiss_onset"])
    f1 = p["f1_base"] + p["f1_range"] * np.tanh(p["f1_slope"] * gap)
    f2 = p["f2_top"] - p["f2_range"] * place
    peak = lambda centre, width: np.exp(-((CENTRES - centre[..., None]) / width) ** 2)
    tone = voiced[..., None] * (peak(f1, p["peak_voiced"]) + p["f2_gain"] * peak(f2, p["peak_voiced"])) \
        + whisper[..., None] * (peak(f1, p["peak_whisper"]) + p["f2_gain"] * peak(f2, p["peak_whisper"]))
    noise = (hiss + burst)[..., None] * peak(p["noise_top"] - p["noise_range"] * place, p["noise_width"])
    return tone + noise


def simulate(p, targets, wobble=0.0, rng=None, hold=HOLD):
    """targets (B, S, 5) -> positions (B,T,5), pressure, burst (B,T), spectrogram (B,T,8), effort per frame (B,T)."""
    B, S, _ = targets.shape
    mass, period, damping = vec(p, "mass"), vec(p, "period"), vec(p, "damping")
    spring = mass * (2 * np.pi / period) ** 2
    friction = 2 * damping * np.sqrt(spring * mass)
    max_force = p["reach"] * spring
    strength = max_force / max_force.sum()

    share = np.clip((targets - REST) / p["reach"], -1.0, 1.0)
    if wobble:
        share = np.clip(share + wobble * rng.standard_normal(share.shape), -1.0, 1.0)

    x, v = np.tile(REST, (B, 1)), np.zeros((B, 5))
    pressure = np.zeros(B)
    xs, ps, bs, es = [], [], [], []
    dt = 1.0 / SUBSTEPS
    for s in range(S):
        push = share[:, s] * max_force
        effort = (share[:, s] ** 2 * strength).sum(1)
        for _ in range(hold):
            for _ in range(SUBSTEPS):
                wall = WALL * spring * (np.maximum(-x, 0.0) - np.maximum(x - 1.0, 0.0))
                a = (push - spring * (x - REST) - friction * v + wall) / mass
                v = v + dt * a
                x = x + dt * v
            gap, breath = opening(x), np.clip(x[:, 3], 0.0, 1.0)
            shut = np.exp(-(gap / p["shut"]) ** 2)
            burst = pressure * np.tanh(p["burst_sharp"] * gap)
            pressure = np.clip(pressure + p["press_build"] * breath * shut - p["press_release"] * burst
                               - p["press_leak"] * pressure, 0.0, 1.0)
            xs.append(x); ps.append(pressure); bs.append(burst); es.append(effort)
    pos, burst = np.stack(xs, 1), np.stack(bs, 1)
    return pos, np.stack(ps, 1), burst, sound(p, pos, burst), np.stack(es, 1)


def word_span(spec):
    loud = spec.sum(-1) >= SILENCE_LEVEL
    B, T = loud.shape
    start = np.where(loud.any(1), loud.argmax(1), 0)
    end, quiet, done = np.zeros(B, int), np.zeros(B, int), ~loud.any(1)
    for t in range(T):
        active = ~done & (t >= start)
        end = np.where(active & loud[:, t], t + 1, end)
        quiet = np.where(active & loud[:, t], 0, np.where(active, quiet + 1, quiet))
        done = done | (active & (quiet >= SILENCE_FRAMES))
    return start, end


def channel(spec, start, end, rng):
    B, T, bands = spec.shape
    frames = end - start
    length = np.where(frames > 0, np.maximum(1, np.rint(frames * (1 + RATE * rng.uniform(-1, 1, B))).astype(int)), 0)
    before, after = rng.integers(MAX_DELAY + 1, size=B), rng.integers(1, MAX_DELAY + 2, size=B)
    L = int(np.ceil(T * (1 + RATE))) + 2 * MAX_DELAY + 2
    k = np.arange(L)[None] - before[:, None]
    inside = (k >= 0) & (k < length[:, None])
    at = np.clip((k + 0.5) * frames[:, None] / np.maximum(length, 1)[:, None] - 0.5, 0, np.maximum(frames - 1, 0)[:, None])
    at = np.where(inside, at, 0.0) + start[:, None]
    lo = np.clip(np.floor(at).astype(int), 0, T - 1)
    hi = np.clip(lo + 1, 0, T - 1)
    w = (at - lo)[..., None]
    out = np.take_along_axis(spec, lo[..., None], 1) * (1 - w) + np.take_along_axis(spec, hi[..., None], 1) * w
    out = out * inside[..., None]
    out = out * (1 + GAIN * rng.uniform(-1, 1, B))[:, None, None] + NOISE * rng.standard_normal(out.shape)
    valid = np.arange(L)[None] < (before + length + after)[:, None]
    return out, valid


def features(heard, valid):
    spec = np.maximum(heard, 0.0)
    loud = np.maximum(spec.sum(-1) - 0.4, 0.0) * valid
    total = loud.sum(1, keepdims=True)
    progress = (np.cumsum(loud, 1) - loud / 2) / np.maximum(total, 1e-12)
    piece = np.minimum((progress * PIECES).astype(int), PIECES - 1)
    onehot = (piece[..., None] == np.arange(PIECES)) * loud[..., None]
    out = np.einsum("btk,btf->bkf", onehot, spec).reshape(len(spec), -1)
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    return np.where(norm > 0, out / np.maximum(norm, 1e-12), 0.0)


def score(p, lexica, rng, repeats=12, effort_cost=0.02, temperature=0.05):
    """lexica (P, W, S, 5) -> fitness, soft accuracy, hard accuracy, effort, each (P,)."""
    P, W = lexica.shape[:2]
    targets = np.repeat(lexica.reshape(P * W, *lexica.shape[2:]), repeats, 0)
    _, _, _, spec, effort = simulate(p, targets, WOBBLE, rng)
    start, end = word_span(spec)
    effort = (effort * (np.arange(effort.shape[1])[None] < end[:, None])).sum(1).reshape(P, W, repeats).mean((1, 2))
    feats = features(*channel(spec, start, end, rng)).reshape(P, W, repeats, -1)
    half = repeats // 2
    soft = hard = 0.0
    for train, test in [(feats[:, :, :half], feats[:, :, half:]), (feats[:, :, half:], feats[:, :, :half])]:
        template = train.mean(2)
        d2 = ((test[:, :, :, None, :] - template[:, None, None, :, :]) ** 2).sum(-1)
        logits = -d2 / temperature
        prob = np.exp(logits - logits.max(-1, keepdims=True))
        prob = prob / prob.sum(-1, keepdims=True)
        truth = np.arange(W)[None, :, None]
        soft = soft + np.take_along_axis(prob, np.broadcast_to(truth, prob.shape[:3])[..., None], -1)[..., 0].mean((1, 2)) / 2
        hard = hard + (d2.argmin(-1) == truth).mean((1, 2)) / 2
    return soft - effort_cost * effort, soft, hard, effort


def search(p, seed, effort_cost=0.02, generations=200, parents=6, children=24, repeats=12):
    rng = np.random.default_rng(seed)
    pop = rng.uniform(0, 1, (parents, WORDS, SLOTS, 5))
    for g in range(generations):
        sigma = 0.3 * (0.04 / 0.3) ** (g / max(1, generations - 1))
        kids = pop[rng.integers(parents, size=children)].copy()
        donors = pop[rng.integers(parents, size=children)]
        swap = rng.random((children, WORDS)) < 0.1
        kids[swap] = donors[swap]
        mask = rng.random((children, WORDS, SLOTS, 1)) < 0.15
        kids = np.clip(kids + mask * sigma * rng.standard_normal(kids.shape), 0.0, 1.0)
        everyone = np.concatenate([pop, kids])
        fitness = score(p, everyone, rng, repeats=repeats, effort_cost=effort_cost)[0]
        pop = everyone[np.argsort(-fitness)[:parents]]
    final = score(p, pop, rng, repeats=64, effort_cost=effort_cost)
    best = int(np.argmax(final[0]))
    return pop[best], {"fitness": float(final[0][best]), "accuracy": float(final[2][best]), "effort": float(final[3][best])}


class TunedMouth(VoiceBox):
    def __init__(self, p):
        self.p = p

    def sound(self, state: Posture) -> Sound:
        return Sound.make(sound(self.p, state.position, np.float64(state.burst)))


STOP_SYMBOLS = {s for pair in ipa.STOPS.values() for s in pair}
FRICATIVE_SYMBOLS = {s for pair in ipa.FRICATIVES.values() for s in pair}
VOICELESS_OBSTRUENTS = {pair[0] for pair in ipa.STOPS.values()} | {pair[0] for pair in ipa.FRICATIVES.values()}
VOWEL_REGION = {}
for height, row in ipa.VOWELS.items():
    for back, symbol in row.items():
        VOWEL_REGION[symbol] = (height, back)


def spacing(points):
    """Mean distance from each sound to its nearest neighbour, corrected for how many sounds there are."""
    points = np.array(points)
    if len(points) < 3:
        return None
    d = np.linalg.norm(points[:, None] - points[None], axis=-1) + np.eye(len(points)) * 1e9
    return float(d.min(1).mean() * len(points) ** (1 / 3))


def describe(p, lexicon):
    """Words as IPA plus the human-likeness measures, from the noiseless trajectories."""
    pos, pressure, burst, spec, _ = simulate(p, lexicon)
    _, end = word_span(spec)
    mouth = TunedMouth(p)
    words, shapes, tokens, vowel_frames = [], [], [], []
    where = {"V": [], "C": []}
    for w in range(len(lexicon)):
        postures = [Posture(pos[w, t], np.zeros(5), float(pressure[w, t]), float(burst[w, t])) for t in range(end[w])]
        segs = [(t, kind, symbol) for t, kind, symbol in ipa.segments(postures, mouth) if symbol != ipa.BREAK]
        words.append(ipa.to_ipa(postures, mouth))
        shapes.append("".join(kind for _, kind, _ in segs))
        tokens += [(kind, symbol) for _, kind, symbol in segs]
        for t, kind, _ in segs:
            where[kind].append([min(float(opening(pos[w, t])), 1.0), *np.clip(pos[w, t, [2, 4]], 0, 1)])
        vowel_frames.append((int(end[w]), sum(kind == "V" for _, kind, _ in segs)))

    pairs = [(a, b) for s in shapes for a, b in zip(s, s[1:])]
    vowels = [s for kind, s in tokens if kind == "V"]
    consonants = [s for kind, s in tokens if kind == "C"]
    base = lambda s: s.replace(ipa.ASPIRATED, "").split(ipa.TIE)[0]
    stops = sum(base(s) in STOP_SYMBOLS for s in consonants)
    fricatives = sum(base(s) in FRICATIVE_SYMBOLS for s in consonants)
    obstruents = [base(s) for s in consonants if base(s) in STOP_SYMBOLS | FRICATIVE_SYMBOLS]
    regions = {VOWEL_REGION[s[0]] for s in vowels if s[0] in VOWEL_REGION} | \
              {VOWEL_REGION[s[1]] for s in vowels if s[0] == "h" and len(s) > 1 and s[1] in VOWEL_REGION}
    corners = [any(h in ("close", "close-mid") and b == "front" for h, b in regions),
               any(h in ("open", "open-mid") for h, b in regions),
               any(h in ("close", "close-mid") and b == "back" for h, b in regions)]
    syllables = sum(n for _, n in vowel_frames)

    cv_only = lambda s: len(s) >= 2 and len(s) % 2 == 0 and all(k == "CV"[i % 2] for i, k in enumerate(s))
    parts = {
        "vowels_voiced": float(np.mean([ipa.VOICELESS not in s for s in vowels])) if vowels else 0.0,
        "open_syllables": float(np.mean([cv_only(s) for s in shapes])),
    }
    return {
        "words": words, "shapes": shapes, **parts,
        "vowel_spacing": spacing(where["V"]), "consonant_spacing": spacing(where["C"]),
        "alternation": float(np.mean([a != b for a, b in pairs])) if pairs else 0.0,
        "vowel_corners": float(np.mean(corners)),
        "syllables_per_word": syllables / len(shapes),
        "frames_per_syllable": float(sum(f for f, _ in vowel_frames) / syllables) if syllables else None,
        "voiceless_obstruents": float(np.mean([s in VOICELESS_OBSTRUENTS for s in obstruents])) if obstruents else None,
        "stop_share": stops / len(consonants) if consonants else None,
        "fricative_share": fricatives / len(consonants) if consonants else None,
        "distinct_sounds": len({s for _, s in tokens}), "tokens": len(tokens),
        "inventory": dict(Counter(s for _, s in tokens).most_common()),
    }


def random_control(p, seed, lexica=12):
    """The same measures for unoptimized random words in the same mouth: what the mouth and labeller give for free."""
    rng = np.random.default_rng([seed, 999])
    rows = [describe(p, rng.uniform(0, 1, (WORDS, SLOTS, 5))) for _ in range(lexica)]
    keys = ["vowels_voiced", "open_syllables", "vowel_spacing", "consonant_spacing", "stop_share", "fricative_share",
            "voiceless_obstruents", "syllables_per_word", "frames_per_syllable"]
    return {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys}


def human_score(found, control):
    """Three parts, each 0-1: vowels voiced, words made only of CV syllables, and sounds reused
    (found sounds sit closer together than random words' sounds do in the same mouth)."""
    reuse = []
    for key in ("vowel_spacing", "consonant_spacing"):
        if found[key] is not None and control.get(key):
            reuse.append(float(np.clip(1 - found[key] / control[key], 0.0, 1.0)))
    parts = {"vowels_voiced": found["vowels_voiced"], "open_syllables": found["open_syllables"],
             "reuse": float(np.mean(reuse)) if reuse else 0.0}
    return {"human": float(np.mean(list(parts.values()))), "reuse": parts["reuse"]}


def one_run(job):
    name, p, seed, effort_cost, generations = job[:5]
    extra = job[5] if len(job) > 5 else {}
    if generations == 0:
        lexicon = np.random.default_rng(seed).uniform(0, 1, (WORDS, SLOTS, 5))
        stats = dict(zip(["fitness", "soft", "accuracy", "effort"],
                         (float(v[0]) for v in score(p, lexicon[None], np.random.default_rng(seed), repeats=64, effort_cost=effort_cost))))
        stats.pop("soft")
    else:
        lexicon, stats = search(p, seed, effort_cost, generations, **extra)
    found, control = describe(p, lexicon), random_control(p, seed)
    return {"name": name, "seed": seed, "effort_cost": effort_cost, "generations": generations, **stats, **found,
            **human_score(found, control), "control": control}


def run(jobs, out, workers=2):
    with Pool(workers) as pool, open(out, "a") as f:
        for result in pool.imap_unordered(one_run, jobs):
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
