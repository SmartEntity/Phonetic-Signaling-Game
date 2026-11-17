from abc import ABC, abstractmethod
from world.datatypes import Sound
import numpy as np

"""
what the mouth articulated -> what actually gets conveyed?
"""


class Channel(ABC):
    @abstractmethod
    def travel(self, msg: list[Sound]) -> list[Sound]:
        return msg

class ChannelIMPL(Channel):
    def __init__(self, noise: float = 0.1, max_delay: int = 6, rate: float = 0.15, gain: float = 0.2,
                 seed: int | np.random.Generator | None = None) -> None:
        self.noise = noise
        self.max_delay = max_delay
        self.rate = rate
        self.gain = gain
        self.rng = np.random.default_rng(seed)

    def travel(self, msg: list[Sound]) -> list[Sound]:
        spec = np.array([s.toArray() for s in msg]) if msg else np.zeros((0, 8))
        frames, bands = spec.shape

        if frames:
            length = max(1, int(round(frames * (1 + self.rate * self.rng.uniform(-1, 1)))))
            at = np.clip((np.arange(length) + 0.5) * frames / length - 0.5, 0, frames - 1)
            spec = np.stack([np.interp(at, np.arange(frames), spec[:, b]) for b in range(bands)], axis=1)

        before = self.rng.integers(self.max_delay + 1)
        after = self.rng.integers(1, self.max_delay + 2)
        out = np.concatenate([np.zeros((before, bands)), spec, np.zeros((after, bands))])

        out = out * (1 + self.gain * self.rng.uniform(-1, 1))
        out = out + self.noise * self.rng.standard_normal(out.shape)
        return [Sound.make(frame) for frame in out]
