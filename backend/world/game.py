"""

play_round
- wrapped in speaker
    - begin speak
    - full speak 
    - render
- Add noise(channel)
- wrapped in listener
    - listen
    - guess
- validate
- return round details
"""


from world.channels.base import Channel
from world.datatypes import Object, RoundSummary, Sound
from world.players.base import Listener, Speaker

import numpy as np

class Game:
    def __init__(self, speaker: Speaker, listener:Listener, channel: Channel):
        self.speaker = speaker
        self.listener = listener
        self.channel = channel
        return

    def play_rnd(self, obj: Object) -> RoundSummary:
        spoken, effort = self.speaker.act(obj)
        heard = self.channel.travel(spoken)
        guess = self.listener.act(heard)

        acc = obj.space.score(obj, guess)
        
        return RoundSummary(
            obj,
            spoken,
            effort,
            heard,
            guess,
            acc,
            self.speaker.get_positions(),
            self.speaker.get_targets()
        )

    def evaluate(self, objects: list[Object], repeats: int = 1) -> dict:
        rounds = [self.play_rnd(obj) for _ in range(repeats) for obj in objects]
        if not rounds:
            return {"correct": 0.0, "acc": 0.0, "effort": 0.0, "frames": 0.0}
        return {
            "correct": float(np.mean([r.acc == 1.0 for r in rounds])),
            "acc": float(np.mean([r.acc for r in rounds])),
            "effort": float(np.mean([r.effort for r in rounds])),
            "frames": float(np.mean([len(r.spoken) for r in rounds])),
        }
