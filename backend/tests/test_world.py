#Thank you Claude

import numpy as np

from main import ScriptedSpeaker
from world.anatomy.voicebox import VoiceBox
from world.channels.base import ChannelIMPL
from world.datatypes import Posture, Sound
from world.config import Config, build
from world.game import Game
from world.objects import ObjectSpace
from world.players.testers import RandomListener, RandomSpeaker, TemplateListener

SPACE = ObjectSpace({
    "color": ["red", "orange", "yellow", "green", "blue", "purple", "black", "white"],
    "shape": ["circle", "square", "triangle"],
})
OBJ = SPACE.make(color="red", shape="circle")
STILL = Posture.make(VoiceBox.REST)


def at(jaw=0.25, lift=0.0, place=0.5, breath=0.0, voice=0.0):
    return Posture.make([jaw, lift, place, breath, voice])


def say(plan, **kw):
    speaker = ScriptedSpeaker(plan, wobble=0.0, **kw)
    postures, efforts = speaker.process_motion(plan)
    spec = np.array([s.toArray() for s in speaker.render_sound(postures)])
    return np.array([p.position for p in postures]), spec, float(np.mean(efforts))


def word(plan, **kw):
    speaker = ScriptedSpeaker(plan, wobble=0.0, **kw)
    spoken, effort = speaker.act(OBJ)
    return spoken, effort, speaker


SHUT, OPEN = at(jaw=0, lift=1, breath=1), at(jaw=1, breath=1, voice=1)


def test_doing_nothing_is_still_silent_and_free():
    x, spec, effort = say([STILL] * 5)
    assert np.allclose(x, VoiceBox.REST) and np.allclose(spec, 0) and effort == 0


def test_shut_mouth_is_silent_open_voiced_mouth_is_loud():
    _, shut, _ = say([at(jaw=0, lift=1, breath=1, voice=1)] * 8)
    _, open_, _ = say([at(jaw=1, breath=1, voice=1)] * 8)
    assert shut[-1].sum() < 0.01 and open_[-1].sum() > 1.0


def test_whisper_is_much_quieter_than_voice():
    _, whisper, _ = say([at(jaw=1, breath=1, voice=0)] * 8)
    _, voiced, _ = say([at(jaw=1, breath=1, voice=1)] * 8)
    assert whisper[-1].max() < 0.2 * voiced[-1].max()


def test_where_a_closure_is_made_is_heard_only_when_it_opens():
    closed, opened = [], []
    for place in (0.0, 1.0):
        plan = [at(jaw=0, lift=1, place=place, breath=1)] * 6 + [at(jaw=1, place=place, breath=1, voice=1)] * 3
        speaker = ScriptedSpeaker(plan, wobble=0.0)
        postures, _ = speaker.process_motion(plan)
        spec = np.array([s.toArray() for s in speaker.render_sound(postures)])
        shut = sum(speaker.get_holds()[:6])
        closed.append(spec[shut - 5:shut])
        opened.append(spec[shut:shut + 4])
    assert np.abs(closed[0] - closed[1]).max() < 0.01
    assert np.abs(opened[0] - opened[1]).max() > 0.1


def test_nothing_jumps_and_parts_fall_back_to_rest():
    x, _, _ = say([at(jaw=1)] * 2 + [STILL] * 20)
    assert x[0, 0] - VoiceBox.REST[0] < 0.5
    assert abs(x[-1, 0] - VoiceBox.REST[0]) < 0.02


def test_channel_changes_length_and_adds_noise():
    spoken = [Sound.make(np.ones(8))] * 40
    channel = ChannelIMPL(seed=1)
    lengths = {len(channel.travel(spoken)) for _ in range(20)}
    heard = np.array([s.toArray() for s in channel.travel(spoken)])
    assert len(lengths) > 5 and 35 <= min(lengths) and max(lengths) <= 59 and heard.std() > 0.05
    assert 1 <= len(channel.travel([])) <= 13


def test_objects_cover_every_combination_once():
    everything = SPACE.all_objects()
    assert len(everything) == np.prod(SPACE.sizes()) == len(set(everything))
    for obj in everything:
        v = obj.toTensor()
        assert len(v) == sum(SPACE.sizes()) and v.sum() == len(SPACE.attributes)
        assert SPACE.from_indices(obj.values) == obj
    blue_square = SPACE.make(color="blue", shape="square")
    assert str(blue_square) == "blue square" and blue_square.color == "blue" and blue_square.shape == "square"


def test_split_is_repeatable_and_shows_every_value():
    seen, held = SPACE.split(0.2, seed=3)
    assert (seen, held) == SPACE.split(0.2, seed=3)
    assert not set(seen) & set(held) and set(seen) | set(held) == set(SPACE.all_objects())
    assert len(held) == int(0.2 * len(SPACE.all_objects()))
    for a, n in enumerate(SPACE.sizes()):
        assert len({obj.values[a] for obj in seen}) == n


def test_score_gives_partial_credit():
    red_circle, red_square, blue_square = (SPACE.make(color=c, shape=s) for c, s in [("red", "circle"), ("red", "square"), ("blue", "square")])
    assert SPACE.score(red_circle, red_circle) == 1 and SPACE.score(red_circle, red_square) == 0.5 and SPACE.score(red_circle, blue_square) == 0


def test_adding_an_attribute_is_one_more_row():
    bigger = ObjectSpace({**SPACE.attributes, "size": ["tiny", "small", "big", "huge"]})
    obj = bigger.make(color="green", shape="triangle", size="huge")
    assert len(bigger.all_objects()) == 96 and len(obj.toTensor()) == 15 and obj.size == "huge" and str(obj) == "green triangle huge"


def test_short_closure_stays_inside_the_word_long_one_ends_it():
    short, _, _ = word([OPEN, OPEN, SHUT, OPEN, OPEN, STILL])
    long, _, _ = word([OPEN, OPEN, SHUT, SHUT, SHUT, SHUT, OPEN, OPEN, STILL])
    alone, _, _ = word([OPEN, OPEN, STILL])
    assert len(short) > len(alone) + 8 and abs(len(long) - len(alone)) <= 3


def test_word_is_trimmed_to_its_sound_and_lines_up_with_positions():
    spoken, _, speaker = word([SHUT, SHUT, OPEN, OPEN, STILL, STILL, STILL, STILL])
    loud = [sum(s.f) >= speaker.silence_level for s in spoken]
    assert loud[0] and loud[-1] and len(spoken) < 32 and speaker.get_positions().shape == (len(spoken), 5)


def test_silent_plan_is_an_empty_free_word():
    spoken, effort, speaker = word([STILL] * 6)
    assert spoken == [] and effort == 0 and speaker.get_positions().shape == (0, 5)


def test_longer_word_costs_more():
    _, short, _ = word([OPEN] * 2 + [STILL] * 3)
    _, long, _ = word([OPEN] * 12 + [STILL] * 3)
    assert long > 1.5 * short


def test_same_seed_same_results_and_baselines_rank_as_expected():
    def run(seed):
        world = build(Config(seed=seed))
        objects = world.space.all_objects()
        speaker = RandomSpeaker(**world.speaker_args())
        matcher = TemplateListener(world.space)
        matcher.study(speaker, world.channel, objects)
        return (Game(speaker, RandomListener(world.space, world.listener_rng), world.channel).evaluate(objects, 10),
                Game(speaker, matcher, world.channel).evaluate(objects, 10))
    guessing, matching = run(0)
    assert (guessing, matching) == run(0) and (guessing, matching) != run(1)
    assert guessing["correct"] < 0.12 and matching["correct"] > 2 * guessing["correct"] + 0.1


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok  ", name)


def test_arrays_for_the_world_tensors_for_the_networks():
    import torch
    from world.datatypes import Motion, Sound
    motion, sound = Motion.make([0.1, 0.2, 0.3, 0.4, 0.5]), Sound.make(range(8))
    for item, size in [(motion, 5), (sound, 8)]:
        assert isinstance(item.toArray(), np.ndarray) and item.toArray().shape == (size,)
        assert isinstance(item.toTensor(), torch.Tensor) and item.toTensor().dtype == torch.float32
        assert np.allclose(item.toArray(), item.toTensor().numpy())
    assert isinstance(SPACE.all_objects()[0].toTensor(), torch.Tensor)
