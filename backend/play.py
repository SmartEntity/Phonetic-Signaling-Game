import sys
from pathlib import Path

import torch

from util.checkpoint import exists, load_run, make_players
from util.ipa import explain, plan_to_ipa, to_ipa
from world.config import Config, build
from world.game import Game


def load(path: str | None):
    if path and exists(Path(path)):
        run = load_run(path)
        return run.world, run.speaker, run.listener
    config = Config(run_dir="")
    world = build(config)
    speaker, listener = make_players(config, world)
    return world, speaker, listener


def choose(name: str, options: list[str]) -> int | None:
    print(f"\n{name}:")
    for i, option in enumerate(options, 1):
        print(f"  {i}) {option}")
    while True:
        answer = input("number (q to quit): ").strip().lower()
        if answer == "q":
            return None
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"  pick a number from 1 to {len(options)}")


def bars(sounds) -> str:
    return "".join(" .:-=+*#%@"[min(9, int(sum(s.f) * 4))] for s in sounds)


def main():
    world, speaker, listener = load(sys.argv[1] if len(sys.argv) > 1 else None)
    speaker.exploring = False
    game = Game(speaker, listener, world.channel)
    held = set(world.held)
    while True:
        picks = []
        for name, options in world.space.attributes.items():
            pick = choose(name, options)
            if pick is None:
                return
            picks.append(pick)
        obj = world.space.from_indices(picks)
        with torch.no_grad():
            r = game.play_rnd(obj)
            confidence = listener.confidence(r.guess)
        postures = speaker.get_postures()
        print(f"\nYou chose:      {obj}" + ("   (held out during training)" if obj in held else ""))
        print(f"Speaker meant:  {plan_to_ipa(r.planned, speaker.mouth, speaker.get_holds(), simple=True)}")
        print(f"Speaker said:   {to_ipa(postures, speaker.mouth, simple=True)}   ({len(r.spoken)} frames, effort {r.effort:.2f})")
        print(f"In words:       {explain(postures, speaker.mouth) or 'silence'}")
        print(f"Sound:          {bars(r.spoken)}")
        print(f"Listener heard: {bars(r.heard)}")
        print(f"Listener guess: {r.guess}   (its confidence in that guess {confidence:.0%})")
        print("RIGHT" if r.acc == 1.0 else f"WRONG ({r.acc:.0%} of attributes right)")


if __name__ == "__main__":
    main()
