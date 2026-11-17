#TODO
"""
Speaker
- speak(obj)->msg abstract
- start_word

Listener
- guess(msg)->ans
"""


from abc import ABC, abstractmethod

from world.anatomy.base import Mouth
from world.datatypes import Sound, Object, Motion, Posture

import numpy as np

class Speaker(ABC):
    def __init__(self, max_slots: int) -> None:
        self.max_slots = max_slots

    @abstractmethod
    def act(self, obj: Object) -> tuple[list[Sound], float]:
        ...

    def get_positions(self) -> np.ndarray:
        return np.zeros((0, 5))

    def get_targets(self) -> list[Posture]:
        return []


class ArticulatedSpeaker(Speaker):
    def __init__(self, max_slots: int, mouth: Mouth, tolerance: float = 0.05, max_travel: int = 30, wobble: float = 0.15,
                 silence_frames: int = 8, silence_level: float = 0.05, seed=None, max_tolerance: float | None = None) -> None:
        super().__init__(max_slots)
        self.mouth = mouth
        self.tolerance = tolerance    # how close to a target counts as there
        self.max_tolerance = tolerance if max_tolerance is None else max(tolerance, max_tolerance)  # EXPERIMENT: chosen-tolerance (the sloppiest the speaker may choose to be)
        self.max_travel = max_travel  # give up on a target after this many frames
        self.wobble = wobble
        self.silence_frames = silence_frames
        self.silence_level = silence_level
        self.rng = np.random.default_rng(seed)
        self.start_new()

    def act(self, obj: Object) -> tuple[list[Sound], float]:
        self.start_new()
        self.targets = self.plan_postures(obj)[:self.max_slots]
        postures, efforts = self.process_motion(self.targets)
        sounds = self.render_sound(postures)
        self.span = self.word_span(sounds)
        start, end = self.span
        return sounds[start:end], float(sum(efforts[:end]))

    def start_new(self):
        self.targets : list[Posture] = []
        self.state : list[Posture] = [self.mouth.rest]
        self.span : tuple[int, int] = (0, 0)
        self.efforts : list[float] = []
        self.holds : list[int] = []  # frames each target took
        self.tolerances : list[float] = []  # EXPERIMENT: chosen-tolerance (how close each target has to be reached; where none is given, self.tolerance)

    @abstractmethod
    def plan_postures(self, obj: Object) -> list[Posture]: #TODO space for RL model to spit something out
        ...

    def process_motion(self, targets: list[Posture]) -> tuple[list[Posture], list[float]]:
        for slot, target in enumerate(targets):
            self.execute(target, self.tolerances[slot] if slot < len(self.tolerances) else self.tolerance)  # EXPERIMENT: chosen-tolerance (was self.execute(target))

        return self.state[1:], self.efforts

    def execute(self, target: Posture, tolerance: float) -> None:
        # Keep steering towards the target until the mouth is there, however many frames that takes.
        # The wobble is a slightly wrong aim: the mouth goes precisely to a spot near the target.
        slip = self.wobble * self.rng.standard_normal(5)
        aim = Posture.make(np.clip(target.position + slip, 0.0, 1.0))
        frames = 0
        while frames < self.max_travel:
            frames += 1
            posture, delta = self.mouth.move(self.state[-1], self.mouth.steer(self.state[-1], aim))
            self.state.append(posture)
            self.efforts.append(delta)
            if self.mouth.arrived(self.state[-1], aim, tolerance):
                break
        self.holds.append(frames)

    # EXPERIMENT: chosen-tolerance
    # The speaker picks how careful to be with each target: 0 is as careful as it gets (tolerance), 1 is as sloppy as allowed (max_tolerance).
    def chooses_tolerance(self) -> bool:
        return self.max_tolerance > self.tolerance

    # EXPERIMENT: chosen-tolerance
    def tolerance_for(self, hurry: float) -> float:
        return self.tolerance + float(np.clip(hurry, 0.0, 1.0)) * (self.max_tolerance - self.tolerance)

    def render_sound(self, postures: list[Posture]) -> list[Sound]:
        return [self.mouth.sound(p) for p in postures]

    def word_span(self, sounds: list[Sound]) -> tuple[int, int]:
        loud = [sum(s.f) >= self.silence_level for s in sounds]
        if True not in loud:
            return (0, 0)
        start = loud.index(True)
        end, quiet = start + 1, 0
        for t in range(start + 1, len(loud)):
            if loud[t]:
                end, quiet = t + 1, 0
            else:
                quiet += 1
                if quiet >= self.silence_frames:
                    break
        return (start, end)

    def get_targets(self) -> list[Posture]:
        return list(self.targets)

    # EXPERIMENT: chosen-tolerance
    def get_tolerances(self) -> list[float]:
        return list(self.tolerances)

    # How many frames each target took.
    def get_holds(self) -> list[int]:
        return list(self.holds)

    # How many targets lasted into the audible word, so learn() only credits those.
    def used_slots(self) -> int:
        ends = np.cumsum(self.holds)
        return max(1, min(int(np.searchsorted(ends, self.span[1])) + 1, len(ends)))

    def get_postures(self) -> list[Posture]:
        return self.state[1:][:self.span[1]]

    def get_positions(self) -> np.ndarray:
        start, end = self.span
        return np.array([p.position for p in self.state[1:][start:end]]).reshape(-1, 5)


class Listener(ABC):
    @abstractmethod
    def act(self, msg: list[Sound]) -> Object:
        pass
