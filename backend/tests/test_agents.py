from dataclasses import replace
import math

import numpy as np
import torch

from train import heard_score, train
from world.config import Config, build
from world.game import Game
from world.players.RNNagents import RNNListener, RNNSpeaker

SMALL = {"color": ["red", "blue"], "shape": ["circle", "square"]}


def players(config):
    torch.manual_seed(config.seed)
    world = build(config)
    speaker = RNNSpeaker(world.space, config.hidden, config.speaker_lr, config.explore, **world.speaker_args())
    listener = RNNListener(world.space, config.hidden, config.listener_lr, config.pool)
    return world, speaker, listener


def test_plan_is_steady_without_exploring_and_varies_with_it():
    world, speaker, _ = players(Config())
    obj = world.space.all_objects()[0]
    plan = lambda: np.array([p.position for p in speaker.plan_postures(obj)])
    assert plan().shape == (world.config.max_slots, 5) and not np.allclose(plan(), plan())
    speaker.exploring = False
    assert np.allclose(plan(), plan())


def test_a_round_runs_and_learning_changes_both_players():
    world, speaker, listener = players(Config())
    game = Game(speaker, listener, world.channel)
    before = [p.detach().clone() for p in [*speaker.cell.parameters(), *listener.rnn.parameters()]]
    for obj in world.seen[:8]:
        summary = game.play_rnd(obj)
        assert 0 <= listener.confidence(obj) <= 1 and summary.guess.space == world.space
        listener.learn(obj)
        speaker.learn(obj, listener.confidence(obj))
    speaker.update()
    listener.update()
    after = [*speaker.cell.parameters(), *listener.rnn.parameters()]
    assert all(not torch.equal(a, b) for a, b in zip(before, after)) and not speaker.losses and not listener.losses


def test_training_runs_and_reports():
    lines = []
    world, speaker, listener = train(Config(attributes=SMALL, held_out=0.0, steps=2, batch=4, report_every=1, run_dir=""), log=lines.append)
    assert len(lines) == 3 and all("seen" in line for line in lines) and len(world.seen) == 4


# EXPERIMENT: no-silence
def test_saying_nothing_always_scores_worse_than_saying_something():
    config = Config(attributes=SMALL, hidden=16, run_dir="", reward_floor=True)
    world, speaker, listener = players(config)
    obj = world.seen[0]
    Game(speaker, listener, world.channel).play_rnd(obj)
    blind_guess = math.log(1 / len(world.space.all_objects()))
    assert heard_score(config, listener, obj, [], blind_guess) < blind_guess <= heard_score(config, listener, obj, ["a sound"], blind_guess)


# EXPERIMENT: reward-floor
def test_reward_floor_only_applies_when_switched_on():
    config = Config(attributes=SMALL, hidden=16, run_dir="")
    world, speaker, listener = players(config)
    obj = world.seen[0]
    Game(speaker, listener, world.channel).play_rnd(obj)
    unfloored = heard_score(config, listener, obj, ["a sound"], 0.0)
    assert unfloored == listener.log_confidence(obj)
    assert heard_score(replace(config, reward_floor=True), listener, obj, ["a sound"], 0.0) == max(unfloored, 0.0)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok  ", name)
