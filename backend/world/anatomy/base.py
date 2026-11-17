"""
ear? no for now
"""


"""
mouth
move() -> effort


"""

from abc import ABC, abstractmethod

from world.datatypes import Motion, Posture, Sound

class Mouth(ABC):
    @property
    @abstractmethod
    def rest(self) -> Posture: ...

    # The push that takes the mouth from where it is towards a target, for one frame.
    @abstractmethod
    def steer(self, state: Posture, target: Posture) -> Motion: ...

    # Is the mouth close enough to the target, and nearly still?
    @abstractmethod
    def arrived(self, state: Posture, target: Posture, tolerance: float) -> bool: ...

    @abstractmethod
    def move(self, state: Posture, motion: Motion) -> tuple[Posture, float]: ...

    @abstractmethod
    def sound(self, state: Posture) -> Sound: ...
