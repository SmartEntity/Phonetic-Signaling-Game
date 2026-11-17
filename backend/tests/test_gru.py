import numpy as np
import torch
from torch import nn

from world.config import Config, build
from world.game import Game
from world.players.GRUagents import MyGRUCell, GRUListener, GRUSpeaker


def test_hand_written_cell_matches_pytorch():
    torch.manual_seed(0)
    mine, ref = MyGRUCell(13, 16), nn.GRUCell(13, 16)
    ref.weight_ih.data.copy_(mine.from_input.weight)
    ref.bias_ih.data.copy_(mine.from_input.bias)
    ref.weight_hh.data.copy_(mine.from_memory.weight)
    ref.bias_hh.data.copy_(mine.from_memory.bias)
    for shape in [(), (5,)]:
        x, h = torch.randn(*shape, 13), torch.randn(*shape, 16)
        assert mine(x, h).shape == ref(x, h).shape and torch.allclose(mine(x, h), ref(x, h), atol=1e-6)


def test_gates_keep_or_replace_memory():
    cell = MyGRUCell(3, 4)
    x, h = torch.randn(3), torch.randn(4)
    cell.from_memory.bias.data[4:8] = 50.0
    assert torch.allclose(cell(x, h), h, atol=1e-4)
    cell.from_memory.bias.data[4:8] = -50.0
    assert not torch.allclose(cell(x, h), h, atol=1e-2)
    assert cell.update_gate.shape == (4,) and cell.reset_gate.shape == (4,)


def test_gru_players_run_and_learn():
    config = Config()
    torch.manual_seed(0)
    world = build(config)
    speaker = GRUSpeaker(world.space, config.hidden, config.speaker_lr, config.explore, **world.speaker_args())
    listener = GRUListener(world.space, config.hidden, config.listener_lr, pool=1)
    game = Game(speaker, listener, world.channel)
    before = [p.detach().clone() for p in [*speaker.cell.parameters(), *listener.cell.parameters()]]
    for obj in world.seen[:8]:
        summary = game.play_rnd(obj)
        assert 0 <= listener.confidence(obj) <= 1 and summary.guess.space == world.space
        listener.learn(obj)
        speaker.learn(obj, listener.confidence(obj))
    speaker.update()
    listener.update()
    after = [*speaker.cell.parameters(), *listener.cell.parameters()]
    assert all(not torch.equal(a, b) for a, b in zip(before, after))


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok  ", name)
