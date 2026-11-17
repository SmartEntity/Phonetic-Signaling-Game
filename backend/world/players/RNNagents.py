#TODO
"""
look into RNN

Speaker: Each in -> All out
Listern: All in -> Last out


"""

import math

import numpy as np
import torch
from torch import nn

from world.datatypes import Object, Posture, Sound
from world.objects import ObjectSpace
from world.players.base import ArticulatedSpeaker, Listener


class RNNSpeaker(ArticulatedSpeaker):
    def __init__(self, space: ObjectSpace, hidden: int = 64, lr: float = 1e-3, explore: float = 0.1, **kw) -> None:
        super().__init__(**kw)
        self.outputs = 6 if self.chooses_tolerance() else 5  # EXPERIMENT: chosen-tolerance (the sixth output is how sloppy to be with this target)
        self.cell = nn.RNNCell(sum(space.sizes()) + self.outputs + 1, hidden)  # EXPERIMENT: slot-counter (the + 1)
        self.out = nn.Linear(hidden, self.outputs)
        self.optimizer = torch.optim.Adam([*self.cell.parameters(), *self.out.parameters()], lr)
        self.explore = explore
        self.exploring = True
        self.expected : dict[Object, float] = {}
        self.slot_log_probs : list[torch.Tensor] = []
        self.losses : list[torch.Tensor] = []

    def plan_postures(self, obj: Object) -> list[Posture]:
        seen = obj.toTensor()
        memory = torch.zeros(self.cell.hidden_size)
        target = torch.zeros(self.outputs)
        targets, self.slot_log_probs = [], []
        for slot in range(self.max_slots):
            count = torch.tensor([slot / self.max_slots])  # EXPERIMENT: slot-counter (how far through the word it is, counted in targets)
            memory = self.cell(torch.cat([seen, target, count]), memory)  # EXPERIMENT: slot-counter (was cat([seen, target]))
            centre = torch.sigmoid(self.out(memory))
            spread = torch.distributions.Normal(centre, self.explore)
            target = spread.sample() if self.exploring else centre.detach()
            self.slot_log_probs.append(spread.log_prob(target).sum())
            targets.append(Posture.make(target[:5].numpy()))
            if self.chooses_tolerance():  # EXPERIMENT: chosen-tolerance
                self.tolerances.append(self.tolerance_for(float(target[5])))  # EXPERIMENT: chosen-tolerance
        return targets

    def learn(self, obj: Object, reward: float) -> None:
        used = self.used_slots()
        log_prob = torch.stack(self.slot_log_probs[:used]).sum()
        expected = self.expected.get(obj, 0.0)
        self.losses.append(-(reward - expected) * log_prob)
        self.expected[obj] = 0.9 * expected + 0.1 * reward

    def update(self) -> None:
        if self.losses:
            self.optimizer.zero_grad()
            torch.stack(self.losses).mean().backward()
            nn.utils.clip_grad_norm_([*self.cell.parameters(), *self.out.parameters()], 1.0)
            self.optimizer.step()
            self.losses = []


class RNNListener(Listener):
    def __init__(self, space: ObjectSpace, hidden: int = 64, lr: float = 3e-3, pool: int = 4) -> None:
        self.space = space
        self.pool = pool
        self.rnn = nn.RNN(8, hidden)
        self.heads = nn.ModuleList([nn.Linear(hidden, n) for n in space.sizes()])
        self.optimizer = torch.optim.Adam([*self.rnn.parameters(), *self.heads.parameters()], lr)
        self.scores : list[torch.Tensor] = []
        self.losses : list[torch.Tensor] = []

    def act(self, msg: list[Sound]) -> Object:
        frames = Sound.stackTensor(msg)
        chunks = -(-len(frames) // self.pool)
        padded = torch.cat([frames, torch.zeros(chunks * self.pool - len(frames), frames.shape[1])])
        heard = padded.reshape(chunks, self.pool, -1).mean(1)
        _, memory = self.rnn(heard)
        self.scores = [head(memory[0]) for head in self.heads]
        return self.space.from_indices(int(s.argmax()) for s in self.scores)

    def confidence(self, obj: Object) -> float:
        return float(np.prod([torch.softmax(s, 0)[i].item() for s, i in zip(self.scores, obj.values)]))

    def log_confidence(self, obj: Object) -> float:
        return float(sum(torch.log_softmax(s, 0)[i].item() for s, i in zip(self.scores, obj.values)))

    def learn(self, obj: Object) -> None:
        per_attribute = [nn.functional.cross_entropy(s, torch.tensor(i)) for s, i in zip(self.scores, obj.values)]
        self.losses.append(torch.stack(per_attribute).sum())

    def update(self) -> None:
        if self.losses:
            self.optimizer.zero_grad()
            torch.stack(self.losses).mean().backward()
            nn.utils.clip_grad_norm_([*self.rnn.parameters(), *self.heads.parameters()], 1.0)
            self.optimizer.step()
            self.losses = []
