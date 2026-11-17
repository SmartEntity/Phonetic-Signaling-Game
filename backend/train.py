import math
import sys
from typing import Any
from dataclasses import fields, replace
from pathlib import Path

import torch

from util.checkpoint import Run, TrainableListener, TrainableSpeaker, exists, load_run, make_players, run_path, save_run
from util.ipa import lexicon
from world.config import Config, World, build
from world.datatypes import Object
from world.game import Game


UNDERSTOOD_RATE = 0.001  # how quickly "the listener gets it right" is averaged: about 1000 rounds


def cost(config: Config, understood: float, effort: float, frames: int) -> float:
    """What a word costs the speaker in effort and time. Free while nobody understands it, the full price once the listener does.

    Without the scaling, silence wins: before the listener has learned anything, speaking costs something and earns nothing.
    """
    return understood * (config.effort_cost * effort + config.time_cost * frames)


SILENCE_PENALTY = 3.0  # EXPERIMENT: no-silence (how much worse than a blind guess saying nothing scores)


def heard_score(config: Config, listener: Any, obj: Object, spoken: list, blind_guess: float) -> float:
    """How well the listener understood the word, as the speaker's reward before costs."""
    if not spoken:  # EXPERIMENT: no-silence (silence is not a word: it always scores worse than anything said aloud)
        return blind_guess - SILENCE_PENALTY  # EXPERIMENT: no-silence
    if config.reward != "logprob":
        return listener.confidence(obj)
    logprob = listener.log_confidence(obj)
    if config.reward_floor:  # EXPERIMENT: reward-floor
        return max(logprob, blind_guess)  # EXPERIMENT: reward-floor (being misheard is never worse than a blind guess)
    return logprob


def train(config: Config, log=print, resume: bool = False) -> tuple[World, TrainableSpeaker, TrainableListener]:
    path = run_path(config) if config.run_dir else None

    if resume and path is not None and exists(path):
        run = load_run(path, config)
        world, speaker, listener, history, start = run.world, run.speaker, run.listener, run.history, run.step + 1
        log(f"resuming {path} from step {run.step}")
    else:
        if path is not None and exists(path):
            raise FileExistsError(f"{path} already holds a run. Add resume=1 to continue it, or set run_name=<new name>.")
        torch.manual_seed(config.seed)
        world = build(config)
        speaker, listener = make_players(config, world)
        history, start = [], 0
    if config.reward not in ("confidence", "logprob"):
        raise ValueError(f"reward must be 'confidence' or 'logprob', not {config.reward!r}")
    game = Game(speaker, listener, world.channel)

    def save(step: int) -> None:
        if path is not None:
            save_run(path, Run(config, world, speaker, listener, history, step))

    understood = history[-1]["seen"] if history else 0.0
    blind_guess = math.log(1 / len(world.space.all_objects()))
    last = start - 1
    for step in range(start, config.steps + 1):
        speaker.explore = config.explore + (config.explore_end - config.explore) * min(1.0, step / max(1, config.steps))
        for i in world.train_rng.integers(len(world.seen), size=config.batch):
            obj = world.seen[i]
            summary = game.play_rnd(obj)
            listener.learn(obj)
            heard = heard_score(config, listener, obj, summary.spoken, blind_guess)
            understood += UNDERSTOOD_RATE * ((summary.acc == 1.0) - understood)
            speaker.learn(obj, heard - cost(config, understood, summary.effort, len(summary.spoken)))
        speaker.update()
        listener.update()
        last = step

        if step % config.report_every == 0:
            speaker.exploring = False
            seen, held = game.evaluate(world.seen, 3), game.evaluate(world.held, 3)
            speaker.exploring = True
            history.append(dict(step=step, seen=seen["correct"], held_out=held["correct"], effort=seen["effort"], frames=seen["frames"]))
            log(f"step {step:6d}   seen {seen['correct']:.2f}   held out {held['correct']:.2f}   "
                f"effort {seen['effort']:.1f}   frames {seen['frames']:.1f}")
        if config.save_every and step % config.save_every == 0:
            save(step)

    if last >= start and not (config.save_every and last % config.save_every == 0):
        save(last)
    if path is not None and last >= start:
        log(f"saved to {path}")
    return world, speaker, listener


def main():
    defaults = Config()
    names = {f.name for f in fields(Config)}
    changes, resume = {}, False
    for arg in sys.argv[1:]:
        name, value = arg.split("=", 1)
        if name == "resume":
            resume = value not in ("0", "false", "no")
            continue
        if name not in names:
            raise SystemExit(f"unknown setting {name!r}; choose from {sorted(names | {'resume'})}")
        kind = type(getattr(defaults, name))
        if kind is bool:
            changes[name] = value.lower() in ("1", "true", "yes", "on")
            continue
        if kind not in (int, float, str):
            raise SystemExit(f"{name} can't be set from the command line")
        changes[name] = kind(value)
    world, speaker, listener = train(replace(defaults, **changes), resume=resume)

    speaker.exploring = False
    print()
    for obj, (meant, said) in lexicon(speaker, world.space.all_objects(), simple=True).items():
        print(f"{'held out' if obj in world.held else '':9s} {str(obj):16s} meant {meant}   said {said}")


if __name__ == "__main__":
    main()
