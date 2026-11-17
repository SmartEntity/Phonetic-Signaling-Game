import json
import tempfile
from pathlib import Path

import server
from train import train
from world.config import Config

SMALL = {"shape": ["a", "b", "c"], "colour": ["x", "y", "z"]}


def test_a_round_from_the_server_holds_everything_the_page_shows():
    with tempfile.TemporaryDirectory() as tmp:
        train(Config(attributes=SMALL, held_out=0.0, steps=2, batch=4, report_every=2, save_every=2, hidden=16, run_dir=tmp, run_name="tiny"), log=lambda _: None)
        server.RUNS, server.loaded = Path(tmp), {}

        assert [(run["name"], run["playable"]) for run in server.describe_runs()] == [("tiny", True)]

        round_ = server.play_round("tiny", [1, 2])
        json.dumps(round_)  # everything in it can be sent
        assert round_["object"]["attributes"] == {"shape": "b", "colour": "z"}
        speaker, listener = round_["speaker"], round_["listener"]
        assert len(speaker["sound"]) <= len(speaker["mouth"]) and len(speaker["targets"]) == 10
        assert all(speaker["tightest"] <= target["tolerance"] <= speaker["loosest"] for target in speaker["targets"])
        assert [belief["attribute"] for belief in listener["beliefs"]] == ["shape", "colour"]
        assert all(abs(sum(option["chance"] for option in belief["options"]) - 1) < 1e-4 for belief in listener["beliefs"])
        assert listener["right"] == (listener["guess"]["values"] == [1, 2])

        assert server.play_round("tiny", None)["object"]["values"] in [list(obj.values) for obj in server.get_run("tiny").world.space.all_objects()]
