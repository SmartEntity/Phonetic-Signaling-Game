import json
import tempfile

import numpy as np
import pytest
import torch

from main import ScriptedSpeaker
from train import cost, train
from util.checkpoint import load_run, make_players, run_path
from util.ipa import plan_to_ipa, to_ipa
from world.anatomy.voicebox import VoiceBox
from world.config import Config, build
from world.datatypes import Posture

SMALL = {"shape": ["a", "b", "c"], "colour": ["x", "y", "z"]}
WORLD = build(Config())
OBJ = WORLD.space.all_objects()[0]


def at(jaw=0.25, lift=0.0, place=0.5, breath=0.0, voice=0.0):
    return Posture.make([jaw, lift, place, breath, voice])


def spoken(plan, **kw):
    speaker = ScriptedSpeaker(plan, **{"wobble": 0.0, "tolerance": 0.05, **kw})
    speaker.act(OBJ)
    return speaker


def test_the_mouth_arrives_at_every_target_without_overshooting():
    rng = np.random.default_rng(0)
    for _ in range(50):
        plan = [Posture.make(rng.random(5)) for _ in range(6)]
        speaker = spoken(plan)
        ends = np.cumsum(speaker.get_holds())
        for end, target in zip(ends, plan):
            assert np.abs(speaker.state[end].position - target.position).max() < 0.05
        assert max(speaker.get_holds()) < speaker.max_travel
        positions = np.array([p.position for p in speaker.state])
        assert positions.min() > -0.01 and positions.max() < 1.01


def test_far_targets_take_longer_and_cost_more_than_near_ones():
    near, far = spoken([at(jaw=0.35)]), spoken([at(jaw=1.0, place=1.0)])
    assert far.get_holds()[0] > near.get_holds()[0]
    assert sum(far.efforts) > sum(near.efforts)


def test_a_target_the_mouth_is_already_at_takes_one_frame():
    there = at(jaw=1.0, breath=1.0, voice=1.0)
    speaker = spoken([there, there, there])
    assert speaker.get_holds()[1:] == [1, 1]


def test_a_tighter_tolerance_is_slower():
    plan = [at(jaw=1.0, place=0.0, breath=1.0, voice=1.0)]
    assert spoken(plan, tolerance=0.02).get_holds()[0] > spoken(plan, tolerance=0.1).get_holds()[0]


def test_what_is_said_matches_what_was_meant():
    shut = at(jaw=0, lift=1, place=0.0, breath=1)
    plan = [shut, shut, shut, at(jaw=1, place=0.0, breath=1, voice=1), shut, shut, shut, at(jaw=1, place=0.0, breath=1, voice=1), at()]
    speaker = spoken(plan)
    assert to_ipa(speaker.get_postures(), speaker.mouth) == "papa"
    assert plan_to_ipa(plan, speaker.mouth, speaker.get_holds()) == "papa"


def test_wobble_is_a_slightly_wrong_aim_and_differs_each_time():
    plan = [at(jaw=1.0, breath=1.0, voice=1.0)]
    first, second = spoken(plan, wobble=0.1, seed=1), spoken(plan, wobble=0.1, seed=2)
    assert not np.allclose(first.state[-1].position, second.state[-1].position)
    assert max(first.get_holds() + second.get_holds()) < first.max_travel


def test_effort_and_time_are_free_until_the_listener_understands():
    config = Config(time_cost=0.01, effort_cost=0.05)
    assert cost(config, 0.0, 2.0, 40) == 0.0
    assert cost(config, 0.5, 2.0, 40) == pytest.approx(0.25)
    assert cost(config, 1.0, 2.0, 40) == pytest.approx(0.5)


def test_runs_save_and_load_and_runs_from_the_old_mouth_are_refused():
    with tempfile.TemporaryDirectory() as tmp:
        config = Config(attributes=SMALL, held_out=0.0, steps=2, batch=4, report_every=2, save_every=2, hidden=16, run_dir=tmp, arch="rnn")
        train(config, log=lambda _: None)
        assert load_run(run_path(config)).speaker.tolerance == 0.05
        meta_path = run_path(config) / "run.json"
        meta = json.loads(meta_path.read_text())
        for name in ("tolerance", "max_travel", "time_cost"):
            del meta["config"][name]
        meta_path.write_text(json.dumps(meta))
        with pytest.raises(ValueError):
            load_run(run_path(config))


# EXPERIMENT: chosen-tolerance
def test_a_speaker_that_chooses_its_tolerance_can_hurry_or_take_care():
    config = Config(attributes=SMALL, hidden=16, run_dir="", wobble=0.0, tolerance=0.05, max_tolerance=0.3)
    world = build(config)
    speaker, _ = make_players(config, world)
    assert speaker.chooses_tolerance() and speaker.out.out_features == 6
    assert speaker.tolerance_for(-1.0) == speaker.tolerance_for(0.0) == 0.05
    assert speaker.tolerance_for(1.0) == speaker.tolerance_for(2.0) == pytest.approx(0.3)
    speaker.act(world.seen[0])
    assert len(speaker.get_tolerances()) == len(speaker.get_holds()) == config.max_slots
    assert all(0.05 <= t <= 0.3 for t in speaker.get_tolerances())

    far = [at(jaw=1.0, place=0.0, breath=1.0, voice=1.0)]
    careful, hurried = ScriptedSpeaker(far, wobble=0.0), ScriptedSpeaker(far, wobble=0.0)
    careful.execute(far[0], 0.05)
    hurried.execute(far[0], 0.3)
    assert hurried.get_holds()[0] < careful.get_holds()[0]


# EXPERIMENT: chosen-tolerance
def test_with_max_tolerance_equal_to_tolerance_nothing_is_chosen():
    config = Config(attributes=SMALL, hidden=16, run_dir="", max_tolerance=0.05)
    world = build(config)
    speaker, _ = make_players(config, world)
    assert not speaker.chooses_tolerance() and speaker.out.out_features == 5
    speaker.act(world.seen[0])
    assert speaker.get_tolerances() == []


# EXPERIMENT: chosen-tolerance
def test_runs_saved_before_the_speaker_chose_its_tolerance_still_load():
    with tempfile.TemporaryDirectory() as tmp:
        config = Config(attributes=SMALL, held_out=0.0, steps=2, batch=4, report_every=2, save_every=2, hidden=16, run_dir=tmp, max_tolerance=0.05)
        train(config, log=lambda _: None)
        path = run_path(config) / "run.json"
        saved = json.loads(path.read_text())
        del saved["config"]["max_tolerance"]
        path.write_text(json.dumps(saved))
        assert not load_run(run_path(config)).speaker.chooses_tolerance()
