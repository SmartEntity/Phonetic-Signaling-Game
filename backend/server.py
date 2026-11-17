"""
A small web server for the play page in ../frontend/.

GET  /api/runs               the saved runs that can be played
POST /api/round              play one round: {"run": name, "values": [colour, shape, ...] or null for a random object}

It plays the whole round at once and sends everything back. The page decides what to show and when.

    uv run python server.py            (then, in ../frontend/:  npm run dev)
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch

from util.checkpoint import Run, exists, load_run
from util.ipa import explain, plan_to_ipa, to_ipa
from world.game import Game
from world.players.base import ArticulatedSpeaker

RUNS = Path("runs")
PORT = 8000
ARTICULATORS = ["jaw", "lift", "place", "breath", "voice"]

loaded: dict[str, Run] = {}
picker = np.random.default_rng()


def describe_runs() -> list[dict]:
    """Every run in runs/ that the current code can load, newest first."""
    out = []
    folders = sorted((p for p in RUNS.iterdir() if exists(p)), key=lambda p: (p / "run.json").stat().st_mtime, reverse=True) if RUNS.is_dir() else []
    for folder in folders:
        try:
            meta = json.loads((folder / "run.json").read_text())
            log = json.loads((folder / "log.json").read_text())
            get_run(folder.name)
        except Exception as problem:  # an old run the current code can't open
            out.append({"name": folder.name, "playable": False, "why": str(problem)})
            continue
        out.append({"name": folder.name, "playable": True, "step": meta["step"], "accuracy": log[-1]["seen"] if log else 0.0,
                    "attributes": meta["config"]["attributes"]})
    return out


def get_run(name: str) -> Run:
    if name not in loaded:
        folder = RUNS / name
        if folder.parent != RUNS or not exists(folder):
            raise ValueError(f"there is no run called {name!r}")
        run = load_run(folder)
        run.speaker.exploring = False
        loaded[name] = run
    return loaded[name]


def describe_object(obj) -> dict:
    return {"name": str(obj), "values": list(obj.values), "attributes": {a: obj.space.value(obj, a) for a in obj.space.attributes}}


def play_round(name: str, values: list[int] | None) -> dict:
    run = get_run(name)
    world, speaker, listener = run.world, run.speaker, run.listener
    assert isinstance(speaker, ArticulatedSpeaker)
    objects = world.space.all_objects()
    obj = world.space.from_indices(values) if values is not None else objects[int(picker.integers(len(objects)))]

    with torch.no_grad():
        r = Game(speaker, listener, world.channel).play_rnd(obj)
        beliefs = [torch.softmax(scores, 0).tolist() for scores in listener.scores]

    postures, holds = speaker.get_postures(), speaker.get_holds()
    tolerances = speaker.get_tolerances() or [speaker.tolerance] * len(holds)
    start, end = speaker.span
    targets, frame = [], 0
    for target, frames, tolerance in zip(r.planned, holds, tolerances):
        targets.append({"position": target.position.tolist(), "frames": frames, "tolerance": tolerance, "starts_at": frame, "heard": frame < end})
        frame += frames

    return {
        "run": name,
        "step": run.step,
        "object": {**describe_object(obj), "held_out": obj in world.held},
        "speaker": {
            "meant": plan_to_ipa(r.planned, speaker.mouth, holds, simple=True),
            "meant_ipa": plan_to_ipa(r.planned, speaker.mouth, holds),
            "said": to_ipa(postures, speaker.mouth, simple=True),
            "said_ipa": to_ipa(postures, speaker.mouth),
            "in_words": explain(postures, speaker.mouth),
            "effort": r.effort,
            "word_starts_at": start,
            "sound": [list(s.f) for s in r.spoken],
            "mouth": [p.position.tolist() for p in postures],
            "articulators": ARTICULATORS,
            "targets": targets,
            "tightest": speaker.tolerance,
            "loosest": speaker.max_tolerance,
        },
        "listener": {
            "heard": [list(s.f) for s in r.heard],
            "guess": describe_object(r.guess),
            "beliefs": [{"attribute": a, "options": [{"value": v, "chance": p} for v, p in zip(options, chances)]}
                        for (a, options), chances in zip(world.space.attributes.items(), beliefs)],
            "right": r.acc == 1.0,
            "share_right": r.acc,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def reply(self, status: int, body) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.reply(200, {})

    def do_GET(self) -> None:
        if self.path == "/api/runs":
            self.reply(200, describe_runs())
        else:
            self.reply(404, {"error": f"nothing at {self.path}"})

    def do_POST(self) -> None:
        if self.path != "/api/round":
            return self.reply(404, {"error": f"nothing at {self.path}"})
        try:
            asked = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            self.reply(200, play_round(asked["run"], asked.get("values")))
        except (KeyError, ValueError, TypeError) as problem:
            self.reply(400, {"error": str(problem)})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    print(f"serving rounds on http://localhost:{port}   (start the page with: cd ../frontend && npm run dev)")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
