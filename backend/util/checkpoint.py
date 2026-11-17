import importlib
from typing import Any
import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
from torch import nn

from util.ipa import lexicon
from world.config import Config, World, build
from world.players.base import ArticulatedSpeaker, Listener

ARCHITECTURES = {
    "rnn": ("world.players.RNNagents", "RNNSpeaker", "RNNListener"),
    "gru": ("world.players.GRUagents", "GRUSpeaker", "GRUListener"),
}

CAN_CHANGE_WHEN_RESUMING = {"steps", "report_every", "save_every", "run_dir", "run_name"}


class TrainableSpeaker(ArticulatedSpeaker):
    """What a learning speaker adds on top of ArticulatedSpeaker."""
    exploring: bool
    explore: float
    cell: Any
    expected: dict
    optimizer: Any

    def learn(self, obj, reward: float) -> None: ...
    def update(self) -> None: ...


class TrainableListener(Listener):
    optimizer: Any

    def learn(self, obj) -> None: ...
    def confidence(self, obj) -> float: ...
    def log_confidence(self, obj) -> float: ...
    def update(self) -> None: ...


@dataclass
class Run:
    config: Config
    world: World
    speaker: TrainableSpeaker
    listener: TrainableListener
    history: list[dict]
    step: int


def run_path(config: Config) -> Path:
    return Path(config.run_dir) / (config.run_name or f"{config.arch}-seed{config.seed}-pool{config.pool}")


def make_players(config: Config, world: World) -> tuple[TrainableSpeaker, TrainableListener]:
    if config.arch not in ARCHITECTURES:
        raise ValueError(f"unknown arch {config.arch!r}; choose from {list(ARCHITECTURES)}")
    module, speaker_class, listener_class = ARCHITECTURES[config.arch]
    module = importlib.import_module(module)
    speaker = getattr(module, speaker_class)(world.space, config.hidden, config.speaker_lr, config.explore, **world.speaker_args())
    listener = getattr(module, listener_class)(world.space, config.hidden, config.listener_lr, config.pool)
    return speaker, listener


def networks(player) -> dict[str, nn.Module]:
    return {name: value for name, value in vars(player).items() if isinstance(value, nn.Module)}


def config_from_dict(data: dict) -> Config:
    if "tolerance" not in data:
        raise ValueError("this run was saved before the mouth drove itself to each target; open it with the code in archive/")
    data = {"max_tolerance": data["tolerance"], **data}  # EXPERIMENT: chosen-tolerance (runs saved before it existed had a fixed tolerance)
    return Config(**{f.name: data[f.name] for f in fields(Config) if f.name in data})


def write_atomically(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    os.replace(temporary, path)


def save_json(path: Path, data) -> None:
    write_atomically(path, lambda tmp: tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False)))


def save_torch(path: Path, data) -> None:
    write_atomically(path, lambda tmp: torch.save(data, tmp))


def player_state(player) -> dict:
    return {"networks": {name: net.state_dict() for name, net in networks(player).items()},
            "optimizer": player.optimizer.state_dict()}


# Runs saved while the listener used the hand-written MyGRUCell: same maths and weight layout as nn.GRUCell, other names.
HAND_WRITTEN_GRU = {"from_input.weight": "weight_ih", "from_input.bias": "bias_ih",
                    "from_memory.weight": "weight_hh", "from_memory.bias": "bias_hh"}


def renamed(saved: dict, net: nn.Module) -> dict:
    if isinstance(net, nn.GRUCell) and set(saved) == set(HAND_WRITTEN_GRU):
        return {HAND_WRITTEN_GRU[key]: value for key, value in saved.items()}
    return saved


def load_player_state(player, state: dict, where: str) -> None:
    nets = networks(player)
    if set(nets) != set(state["networks"]):
        raise ValueError(f"{where} has networks {sorted(state['networks'])} but this player has {sorted(nets)}")
    migrated = False
    for name, net in nets.items():
        saved = state["networks"][name]
        fixed = renamed(saved, net)
        migrated = migrated or fixed is not saved
        net.load_state_dict(fixed)
    if not migrated:
        player.optimizer.load_state_dict(state["optimizer"])


def rng_states(run: Run) -> dict:
    states = {"train": run.world.train_rng.bit_generator.state, "speaker": run.speaker.rng.bit_generator.state}
    channel_rng = getattr(run.world.channel, "rng", None)
    if channel_rng is not None:
        states["channel"] = channel_rng.bit_generator.state
    return states


def restore_rng_states(run: Run, states: dict) -> None:
    run.world.train_rng.bit_generator.state = states["train"]
    run.speaker.rng.bit_generator.state = states["speaker"]
    channel_rng = getattr(run.world.channel, "rng", None)
    if channel_rng is not None and "channel" in states:
        channel_rng.bit_generator.state = states["channel"]


def words(run: Run) -> list[dict]:
    speaker = run.speaker
    exploring, before = speaker.exploring, rng_states(run)
    speaker.exploring = False
    try:
        spoken = lexicon(speaker, run.world.space.all_objects())
    finally:
        speaker.exploring = exploring
        restore_rng_states(run, before)
    return [{"object": str(obj), "values": list(obj.values), "held_out": obj in run.world.held, "meant": meant, "said": said}
            for obj, (meant, said) in spoken.items()]


def save_run(path: Path, run: Run) -> None:
    path.mkdir(parents=True, exist_ok=True)
    baseline = {",".join(map(str, obj.values)): value for obj, value in getattr(run.speaker, "expected", {}).items()}
    save_torch(path / "speaker.pt", {**player_state(run.speaker), "baseline": baseline})
    save_torch(path / "listener.pt", player_state(run.listener))
    save_torch(path / "training.pt", {"torch_rng": torch.get_rng_state(), "numpy_rngs": rng_states(run)})
    save_json(path / "log.json", run.history)
    save_json(path / "lexicon.json", words(run))
    save_json(path / "run.json", {"step": run.step, "config": asdict(run.config)})


def load_run(path: Path | str, config: Config | None = None) -> Run:
    path = Path(path)
    meta = json.loads((path / "run.json").read_text())
    stored = config_from_dict(meta["config"])
    if config is None:
        config = stored
    else:
        changed = [f.name for f in fields(Config) if f.name not in CAN_CHANGE_WHEN_RESUMING
                   and getattr(config, f.name) != getattr(stored, f.name)]
        if changed:
            raise ValueError(f"{path} was saved with different settings for {changed}; "
                             f"only {sorted(CAN_CHANGE_WHEN_RESUMING)} can change when resuming")

    world = build(config)
    speaker, listener = make_players(config, world)
    run = Run(config, world, speaker, listener, json.loads((path / "log.json").read_text()), meta["step"])

    saved = torch.load(path / "speaker.pt", weights_only=True)
    load_player_state(speaker, saved, "speaker.pt")
    if hasattr(speaker, "expected"):
        speaker.expected = {world.space.from_indices(int(i) for i in key.split(",")): value
                            for key, value in saved["baseline"].items()}
    load_player_state(listener, torch.load(path / "listener.pt", weights_only=True), "listener.pt")

    training = torch.load(path / "training.pt", weights_only=True)
    torch.set_rng_state(training["torch_rng"])
    restore_rng_states(run, training["numpy_rngs"])
    return run


def exists(path: Path) -> bool:
    return (path / "run.json").exists()
