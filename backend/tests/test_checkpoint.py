import json
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from train import train
from util.checkpoint import load_run, run_path
from world.config import Config

SMALL = {"shape": ["a", "b", "c"], "colour": ["x", "y", "z"]}


def small(tmp: str, **extra) -> Config:
    base = dict(attributes=SMALL, held_out=0.0, steps=4, batch=4, report_every=2, save_every=2, hidden=16, run_dir=tmp)
    return Config(**{**base, **extra})


@pytest.mark.parametrize("arch", ["gru", "rnn"])
def test_round_trip_gives_same_words(arch):
    with tempfile.TemporaryDirectory() as tmp:
        config = small(tmp, arch=arch)
        world, speaker, listener = train(config, log=lambda _: None)
        path = run_path(config)
        for name in ["run.json", "speaker.pt", "listener.pt", "training.pt", "log.json", "lexicon.json"]:
            assert (path / name).exists(), name

        run = load_run(path)
        assert run.step == 4 and len(run.history) == 3
        for net_a, net_b in [(vars(speaker), vars(run.speaker)), (vars(listener), vars(run.listener))]:
            for name, value in net_a.items():
                if isinstance(value, torch.nn.Module):
                    for pa, pb in zip(value.parameters(), net_b[name].parameters()):
                        assert torch.equal(pa, pb)
        assert set(speaker.expected) == set(run.speaker.expected)
        assert all(run.speaker.expected[o] == v for o, v in speaker.expected.items())


def test_lexicon_file_lists_every_object():
    with tempfile.TemporaryDirectory() as tmp:
        config = small(tmp, steps=2)
        train(config, log=lambda _: None)
        words = json.loads((run_path(config) / "lexicon.json").read_text())
        assert len(words) == 9 and {"object", "held_out", "meant", "said"} <= set(words[0])


def test_resume_continues_from_saved_step():
    with tempfile.TemporaryDirectory() as tmp:
        lines: list[str] = []
        train(small(tmp, steps=4), log=lines.append)
        lines.clear()
        train(small(tmp, steps=8), log=lines.append, resume=True)
        assert lines[0].startswith("resuming") and "from step 4" in lines[0]
        steps = [int(l.split()[1]) for l in lines if l.startswith("step")]
        assert steps == [6, 8]
        assert load_run(run_path(small(tmp))).step == 8


def test_refuses_to_overwrite_and_to_change_settings():
    with tempfile.TemporaryDirectory() as tmp:
        config = small(tmp, steps=2)
        train(config, log=lambda _: None)
        with pytest.raises(FileExistsError):
            train(config, log=lambda _: None)
        with pytest.raises(ValueError):
            load_run(run_path(config), replace(config, hidden=32))


def test_unknown_arch_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            train(small(tmp, arch="lstm"), log=lambda _: None)


def test_run_dir_empty_writes_nothing():
    with tempfile.TemporaryDirectory() as tmp:
        import os
        os.chdir(tmp)
        train(Config(attributes=SMALL, held_out=0.0, steps=2, batch=2, report_every=1, run_dir="", hidden=16), log=lambda _: None)
        assert not Path("runs").exists()


def test_log_reward_and_explore_decay_run_and_resume():
    with tempfile.TemporaryDirectory() as tmp:
        config = small(tmp, steps=4, reward="logprob", explore=0.5, explore_end=0.1)
        world, speaker, listener = train(config, log=lambda _: None)
        assert abs(speaker.explore - 0.1) < 1e-9
        obj = world.space.all_objects()[0]
        assert listener.log_confidence(obj) < 0
        assert abs(2.718281828 ** listener.log_confidence(obj) - listener.confidence(obj)) < 1e-5
        with pytest.raises(ValueError):
            load_run(run_path(config), replace(config, reward="confidence"))


def test_bad_reward_name_is_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            train(small(tmp, reward="nope"), log=lambda _: None)
