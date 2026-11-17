import time, numpy as np
from experiments.mouth_tuning import *
from world.players.testers import RandomSpeaker, TemplateListener
from world.channels.base import ChannelIMPL
from world.objects import ObjectSpace
from world.datatypes import Posture

# 1. batch sim == user's VoiceBox through ArticulatedSpeaker (no wobble)
rng = np.random.default_rng(1)
targets = rng.uniform(0, 1, (5, SLOTS, 5))
pos, pressure, burst, spec, effort = simulate(BASE, targets)
start, end = word_span(spec)
class Fixed(RandomSpeaker):
    def plan_postures(self, obj): return [Posture.make(t) for t in self.plan]
sp = Fixed(max_slots=SLOTS, mouth=VoiceBox(), hold=HOLD, wobble=0.0)
for i in range(5):
    sp.plan = targets[i]
    sounds, eff = sp.act(None)
    ref = np.array([s.toTensor() for s in sounds])
    assert sp.span == (start[i], end[i]), (sp.span, start[i], end[i])
    assert np.allclose(ref, spec[i, start[i]:end[i]], atol=1e-9)
    assert abs(eff - effort[i, :end[i]].sum()) < 1e-9
print("simulator matches VoiceBox + ArticulatedSpeaker")

# 2. features == TemplateListener.features on a channel output
tl = TemplateListener(ObjectSpace({"a": ["x", "y"]}))
heard, valid = channel(spec, start, end, np.random.default_rng(0))
f = features(heard, valid)
for i in range(5):
    n = int(valid[i].sum())
    ref = tl.features([Sound.make(row) for row in heard[i, :n]])
    assert np.allclose(ref, f[i], atol=1e-9)
print("features match TemplateListener")

# 3. channel statistics vs ChannelIMPL (lengths)
ch = ChannelIMPL(seed=0)
i = 0
ref_len = [len(ch.travel([Sound.make(r) for r in spec[i, start[i]:end[i]]])) for _ in range(4000)]
big = np.repeat(spec[i:i+1], 4000, 0)
_, valid = channel(big, np.repeat(start[i:i+1], 4000), np.repeat(end[i:i+1], 4000), np.random.default_rng(0))
print("channel length mean/sd  mine %.2f %.2f   ChannelIMPL %.2f %.2f" % (valid.sum(1).mean(), valid.sum(1).std(), np.mean(ref_len), np.std(ref_len)))

t = time.time(); lex, stats = search(BASE, 0, generations=30); print("30 generations: %.1fs" % (time.time() - t), stats)
