import numpy as np

from world.anatomy.voicebox import VoiceBox
from world.config import Config, build
from world.datatypes import Object, Posture
from world.game import Game
from world.players.base import ArticulatedSpeaker
from util.ipa import lexicon, plan_to_ipa, shape, to_ipa
from world.players.testers import RandomListener, RandomSpeaker, TemplateListener


class ScriptedSpeaker(ArticulatedSpeaker):

    def __init__(self, plan: list[Posture], **kw) -> None:
        super().__init__(len(plan), VoiceBox(), **kw)
        self.plan = plan

    def plan_postures(self, obj: Object) -> list[Posture]:
        return self.plan


def main():
    shut = Posture.make([0.0, 0.9, 0.0, 0.9, 0.0])
    open_ = Posture.make([1.0, 0.0, 0.5, 0.9, 1.0])
    relax = Posture.make(VoiceBox.REST)
    speaker = ScriptedSpeaker([shut, shut, open_, open_, open_, shut, shut, open_, open_, relax], wobble=0.0)

    world = build(Config())
    obj = world.space.all_objects()[0]
    spoken, effort = speaker.act(obj)
    heard = world.channel.travel(spoken)

    print("frame   jaw  lift place breath voice  opening  loudness")
    for t, (x, s) in enumerate(zip(speaker.get_positions(), spoken)):
        loud = s.toArray().sum()
        print(f"{t:5d} {x[0]:5.2f} {x[1]:5.2f} {x[2]:5.2f} {x[3]:6.2f} {x[4]:5.2f} {VoiceBox.opening(x):8.2f} {loud:9.2f}  " + "#" * int(round(10 * loud)))
    print(f"\n{obj}   meant {plan_to_ipa(speaker.get_targets(), simple=True)}   said {to_ipa(speaker.get_postures(), simple=True)}   {shape(speaker.get_postures())}   effort {effort:.3f}   spoken {len(spoken)} frames   heard {len(heard)} frames")

    objects = world.space.all_objects()
    babbler = RandomSpeaker(**world.speaker_args())
    guesser = RandomListener(world.space, world.listener_rng)
    matcher = TemplateListener(world.space)
    matcher.study(babbler, world.channel, objects)
    print(f"\n{len(objects)} objects, random words")
    for obj, (meant, said) in list(lexicon(babbler, objects, simple=True).items())[:8]:
        print(f"  {str(obj):16s} meant {meant}   said {said}")
    for label, listener in [("random guesses", guesser), ("template matching", matcher)]:
        r = Game(babbler, listener, world.channel).evaluate(objects, repeats=10)
        print(f"{label:18s} correct {r['correct']:.2f}   acc {r['acc']:.2f}   effort {r['effort']:.2f}   frames {r['frames']:.1f}")


if __name__ == "__main__":
    main()
