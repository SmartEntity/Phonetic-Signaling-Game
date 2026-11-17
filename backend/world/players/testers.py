import numpy as np

from world.channels.base import Channel
from world.datatypes import Object, Posture, Sound
from world.objects import ObjectSpace
from world.players.base import ArticulatedSpeaker, Listener, Speaker


class RandomSpeaker(ArticulatedSpeaker):
    def __init__(self, plan_seed: int = 0, **kw) -> None:
        super().__init__(**kw)
        self.plan_seed = plan_seed

    def plan_postures(self, obj: Object) -> list[Posture]:
        rng = np.random.default_rng([self.plan_seed, *obj.values])
        return [Posture.make(target) for target in rng.uniform(0.0, 1.0, (self.max_slots, 5))]


class RandomListener(Listener):
    def __init__(self, space: ObjectSpace, seed=None) -> None:
        self.objects = space.all_objects()
        self.rng = np.random.default_rng(seed)

    def act(self, msg: list[Sound]) -> Object:
        return self.objects[self.rng.integers(len(self.objects))]


class TemplateListener(Listener):
    def __init__(self, space: ObjectSpace, pieces: int = 4) -> None:
        self.objects = space.all_objects()
        self.pieces = pieces
        self.templates : dict[Object, np.ndarray] = {}

    def features(self, msg: list[Sound]) -> np.ndarray:
        spec = np.maximum(np.array([s.toArray() for s in msg]), 0.0)
        loud = np.maximum(spec.sum(1) - 0.4, 0.0)
        if loud.sum() == 0:
            return np.zeros(self.pieces * spec.shape[1])
        progress = (np.cumsum(loud) - loud / 2) / loud.sum()
        piece = np.minimum((progress * self.pieces).astype(int), self.pieces - 1)
        out = np.array([(spec[piece == k] * loud[piece == k, None]).sum(0) for k in range(self.pieces)]).ravel()
        return out / np.linalg.norm(out)

    def study(self, speaker: Speaker, channel: Channel, objects: list[Object], repeats: int = 10) -> None:
        for obj in objects:
            heard = [self.features(channel.travel(speaker.act(obj)[0])) for _ in range(repeats)]
            self.templates[obj] = np.mean(heard, axis=0)

    def act(self, msg: list[Sound]) -> Object:
        if not self.templates:
            return self.objects[0]
        heard = self.features(msg)
        return min(self.templates, key=lambda obj: np.linalg.norm(self.templates[obj] - heard))
