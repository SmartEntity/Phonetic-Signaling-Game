from dataclasses import dataclass, field

import numpy as np

from world.anatomy.base import Mouth
from world.anatomy.voicebox import VoiceBox
from world.channels.base import Channel, ChannelIMPL
from world.datatypes import Object
from world.objects import ObjectSpace


@dataclass(frozen=True)
class Config:
    attributes: dict = field(default_factory=lambda: {
        "color": ["red", "yellow", "green", "cyan", "blue", "magenta", "black", "white"],
        "shape": ["circle", "square", "triangle"],
        "size": ["small", "medium", "large", "XL"]
    })
    held_out: float = 0.0
    split_seed: int = 0

    max_slots: int = 10
    wobble: float = 0.05
    tolerance: float = 0.05  # how close to a target counts as there
    max_tolerance: float = 0.3  # EXPERIMENT: chosen-tolerance (the speaker picks, per target, how close counts as there, between tolerance and this; set equal to tolerance to switch off)
    max_travel: int = 30  # most frames spent on one target

    silence_frames: int = 8
    silence_level: float = 0.05

    noise: float = 0.1
    max_delay: int = 6
    rate: float = 0.17
    gain: float = 0.2

    hidden: int = 64
    speaker_lr: float = 1e-3
    listener_lr: float = 3e-3
    explore: float = 0.5
    explore_end: float = 0.1
    reward: str = "logprob"
    reward_floor: bool = False  # EXPERIMENT: reward-floor (True: being misheard is never scored worse than a blind guess)
    pool: int = 4
    batch: int = 32
    steps: int = 100000
    effort_cost: float = 0.05
    time_cost: float = 0.01  # per frame; this and effort_cost are scaled by how often the listener gets it right
    report_every: int = 100

    arch: str = "gru"
    run_dir: str = "runs"
    run_name: str = ""
    save_every: int = 1000

    seed: int = 0


@dataclass
class World:
    config: Config
    space: ObjectSpace
    seen: list[Object]
    held: list[Object]
    mouth: Mouth
    channel: Channel
    speaker_rng: np.random.Generator
    listener_rng: np.random.Generator
    train_rng: np.random.Generator

    def speaker_args(self) -> dict:
        c = self.config
        return dict(max_slots=c.max_slots, mouth=self.mouth, tolerance=c.tolerance, max_tolerance=c.max_tolerance, max_travel=c.max_travel, wobble=c.wobble,
                    silence_frames=c.silence_frames, silence_level=c.silence_level, seed=self.speaker_rng)


def build(config: Config) -> World:
    channel_rng, speaker_rng, listener_rng, train_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(4))
    space = ObjectSpace(config.attributes)
    seen, held = space.split(config.held_out, config.split_seed)
    channel = ChannelIMPL(config.noise, config.max_delay, config.rate, config.gain, seed=channel_rng)
    return World(config, space, seen, held, VoiceBox(), channel, speaker_rng, listener_rng, train_rng)
