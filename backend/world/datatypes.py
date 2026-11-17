"""
Object
Message

RoundData
- 
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import numpy as np
import torch

if TYPE_CHECKING:
    from world.objects import ObjectSpace

@dataclass(frozen=True)
class Object:
    values: tuple[int, ...]
    space: "ObjectSpace" = field(repr=False)

    def toTensor(self) -> torch.Tensor:
        return self.space.to_tensor(self)

    def __getattr__(self, attribute: str) -> str:
        if attribute.startswith("_") or attribute not in self.space.attributes:
            raise AttributeError(attribute)
        return self.space.value(self, attribute)

    def __str__(self) -> str:
        return self.space.name(self)

@dataclass(frozen=True)
class Motion:
    #TODO points of articulation
    #Jaw, Lift, Place, Breath, Voice

    jaw: float
    lift: float
    place: float
    breath: float
    voice: float

    def toArray(self) -> np.ndarray:
        return np.array([self.jaw, self.lift, self.place, self.breath, self.voice], dtype=float)

    def toTensor(self) -> torch.Tensor:
        return torch.tensor(self.toArray(), dtype=torch.float32)

    @staticmethod
    def make(v) -> "Motion":
        return Motion(*(float(x) for x in v))

@dataclass(frozen=True, eq=False)
class Posture:

    position: np.ndarray
    velocity: np.ndarray
    pressure: float = 0.0
    burst: float = 0.0

    @staticmethod
    def make(v) -> "Posture":
        return Posture(np.array(v, dtype=float), np.zeros(5))

@dataclass(frozen=True)
class Sound:
    #TODO points of articulation

    f: tuple[float, float, float, float, float, float, float, float]

    def toArray(self) -> np.ndarray:
        return np.array(self.f, dtype=float)

    def toTensor(self) -> torch.Tensor:
        return torch.tensor(self.f, dtype=torch.float32)

    @staticmethod
    def stackTensor(sounds: list["Sound"]) -> torch.Tensor:
        return torch.tensor([s.f for s in sounds], dtype=torch.float32)

    @staticmethod
    def make(v) -> "Sound":
        a, b, c, d, e, f, g, h = (float(x) for x in v)
        return Sound((a, b, c, d, e, f, g, h))


@dataclass(frozen=True)
class RoundSummary:
    obj: Object
    spoken: list[Sound]
    effort: float
    heard: list[Sound]
    guess: Object
    acc: float
    positions: np.ndarray
    planned: list[Posture]
