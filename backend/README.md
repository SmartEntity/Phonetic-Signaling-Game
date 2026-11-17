# Backend

The game, training and a small web server. Run everything from this folder.

```bash
uv sync
uv run python train.py pool=1 run_name=first     # train; settings are in world/config.py
uv run python play.py runs/first                 # play a round in the terminal
uv run python server.py                          # serve rounds to the web page in ../frontend
uv run python main.py                            # demo of the mouth with a scripted speaker, no training
uv run --with pytest python -m pytest tests --ignore=tests/test_gru.py
```

| Folder | Holds |
|---|---|
| `world/` | config, objects, the mouth (`anatomy/`), the channel, the players, one round of the game |
| `util/` | saving and loading runs, and reading mouth movements as IPA |
| `tests/` | tests |
| `experiments/` | old scripts from the fixed-hold mouth |
| `runs/` | trained runs (not in git) |

See the [main README](../README.md) for what the project is.
