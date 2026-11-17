#Thank you Claude

import numpy as np

from main import ScriptedSpeaker
from util.ipa import NAMES, explain, inventory, lexicon, name_of, plain, plan_to_ipa, shape, to_ipa
from world.config import Config, build
from world.datatypes import Posture
from world.players.testers import RandomSpeaker

WORLD = build(Config())
OBJ = WORLD.space.all_objects()[0]


def at(jaw=0.25, lift=0.0, place=0.5, breath=0.0, voice=0.0):
    return Posture.make([jaw, lift, place, breath, voice])


def say(plan):
    speaker = ScriptedSpeaker(plan, wobble=0.0)
    speaker.act(OBJ)
    return to_ipa(speaker.get_postures()), shape(speaker.get_postures())


STILL = at()


def stop(place, voice=0.0):
    return at(jaw=0, lift=1, place=place, breath=1, voice=voice)


def vowel(jaw, place, voice=1.0):
    return at(jaw=jaw, place=place, breath=1, voice=voice)


ALVEOLAR = 2.5 / 7


def test_stops_by_place_and_voicing():
    assert say([stop(0.0)] * 5 + [vowel(1, 0.0)] * 2 + [STILL]) == ("pa", "CV")
    assert say([stop(ALVEOLAR)] * 2 + [vowel(1, ALVEOLAR)] * 2 + [STILL]) == ("tɐ", "CV")
    assert say([stop(1.0)] * 5 + [vowel(1, 1.0)] * 2 + [STILL]) == ("qɑ", "CV")
    assert say([stop(0.0, voice=1)] * 5 + [vowel(1, 0.0)] * 2 + [STILL]) == ("ba", "CV")


def test_vowels_by_height_backness_voice_and_length():
    assert say([vowel(1.0, 0.0)] * 2 + [STILL])[0] == "a"
    assert say([vowel(0.45, 1.0)] * 2 + [STILL])[0] == "ʌ"
    assert say([vowel(1.0, 0.5, voice=0.0)] * 2 + [STILL])[0] == "ɐ\u0325"
    assert say([vowel(1.0, 0.5)] * 6 + [STILL])[0] == "ɐː"


def test_syllables_fricatives_affricates_and_breath():
    assert say([stop(0.0)] * 5 + [vowel(1, 0.0)] * 2 + [stop(0.0)] + [vowel(1, 0.0)] * 2 + [STILL]) == ("papa", "CVCV")
    hiss = at(jaw=0.2, lift=0.12, place=ALVEOLAR, breath=1, voice=0)
    word, _ = say([vowel(1, ALVEOLAR)] * 2 + [hiss] * 3 + [vowel(1, ALVEOLAR)] * 2 + [STILL])
    assert word.startswith("ɐ") and word.endswith("sɐ")
    assert say([stop(ALVEOLAR)] * 2 + [hiss] * 4 + [vowel(1, ALVEOLAR)] * 2 + [STILL])[0] == "t\u0361sɐ"
    assert say([stop(ALVEOLAR)] * 2 + [vowel(1, ALVEOLAR, voice=0)] + [vowel(1, ALVEOLAR)] * 2 + [STILL])[0] == "tʰɐ"
    assert say([vowel(1, 0.5, voice=0)] + [vowel(1, 0.5)] * 2 + [STILL])[0] == "hɐː"


def test_silence_is_no_word_and_lexicon_is_repeatable():
    assert say([STILL] * 5) == ("", "")
    speaker = RandomSpeaker(**WORLD.speaker_args())
    objects = WORLD.space.all_objects()
    words = lexicon(speaker, objects)
    assert words == lexicon(speaker, objects) and speaker.wobble == WORLD.config.wobble
    assert len(words) == len(objects) and len({said for _, said in words.values()}) > len(objects) // 2


def test_plan_reads_what_was_aimed_for_and_the_mouth_says_it():
    plan = [stop(0.0)] * 3 + [vowel(1, 0.0)] + [stop(1.0)] * 3 + [vowel(1, 1.0), STILL]
    speaker = ScriptedSpeaker(plan, wobble=0.0)
    speaker.act(OBJ)
    assert speaker.get_targets() == plan
    assert plan_to_ipa(plan, speaker.mouth, speaker.get_holds()) == "paqɑ" == to_ipa(speaker.get_postures())


def test_a_planned_stop_is_only_released_if_something_follows_it():
    assert plan_to_ipa([stop(0.0)] * 2) == "p̚"
    assert plan_to_ipa([stop(0.0), vowel(1, 0.0)]) == "pa"
    assert plan_to_ipa([stop(0.0), STILL]) == "p"
    assert plan_to_ipa([at(jaw=0, lift=1)] * 2) == ""


def test_a_shut_mouth_leaking_air_is_not_a_vowel():
    assert plan_to_ipa([stop(0.0)] * 3) == "p̚"
    assert "V" not in say([stop(0.0)] * 3)[1]


def test_symbols_can_be_spelled_out():
    assert name_of("t̠") == "voiceless postalveolar stop"
    assert name_of("ɨ̥ː") == "long voiceless close central vowel"
    assert name_of("tʰ") == "aspirated voiceless alveolar stop"
    assert name_of("t͡s") == "voiceless alveolar affricate"
    assert name_of("p̚") == "unreleased voiceless bilabial stop"
    assert name_of("hɐ") == "breath then open central vowel"
    assert name_of("ɹ̠̥") == "voiceless postalveolar approximant"
    assert len(set(NAMES.values())) == len(NAMES)
    speaker = ScriptedSpeaker([stop(0.0)] * 5 + [vowel(1, 0.0)] * 2 + [STILL], wobble=0.0)
    speaker.act(OBJ)
    assert explain(speaker.get_postures(), speaker.mouth) == "p voiceless bilabial stop, a open front vowel"


def test_every_symbol_a_speaker_produces_has_a_name():
    speaker = RandomSpeaker(**WORLD.speaker_args())
    symbols = set(inventory(speaker, WORLD.space.all_objects() * 4))
    assert symbols and all("?" not in name_of(symbol) for symbol in symbols)


def test_simple_spelling_uses_everyday_letters():
    assert plain("t̠") == "t" and plain("ɐ") == "ah" and plain("ʃ") == "sh" and plain("ɨ̥") == "~ih~"
    assert plain("p̚") == "(p)" and plain("tʰ") == "t'" and plain("t͡s") == "ts" and plain("ɐː") == "ah:" and plain("ɨ̥ː") == "~ih:~"
    assert all("?" not in plain(symbol) for symbol in NAMES)
    assert say([stop(0.0)] * 5 + [vowel(1, 0.0)] * 2 + [STILL])[0] == "pa"
    speaker = ScriptedSpeaker([stop(ALVEOLAR)] * 2 + [vowel(1, ALVEOLAR)] * 2 + [STILL], wobble=0.0)
    speaker.act(OBJ)
    assert to_ipa(speaker.get_postures(), speaker.mouth) == "tɐ"
    assert to_ipa(speaker.get_postures(), speaker.mouth, simple=True) == "tah"
    assert plan_to_ipa([stop(0.0)] * 2, simple=True) == "(p)"
    words = lexicon(RandomSpeaker(**WORLD.speaker_args()), WORLD.space.all_objects(), simple=True)
    assert all("?" not in word for pair in words.values() for word in pair)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok  ", name)
