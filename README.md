# Phonetic Signaling Game

A side project where two neural networks try to invent a spoken language from scratch.

## The idea

One network is the **speaker** and one is the **listener**.

1. The speaker gets shown an object (like a "big red triangle").
2. It has to say a "word" for it, but it can only do that by moving a fake mouth.
3. The sound goes through a noisy channel (like a bad phone line).
4. The listener hears it and guesses what the object was.
5. If the guess is right, both of them get rewarded.

Neither one knows any language at the start. They have to agree on one on their own.

**What I'm trying to find out:** if the mouth has real-ish physics and every movement costs effort, do things like syllables, vowels and consonants show up on their own? Nobody tells the speaker what a vowel is.

## How the fake mouth works

The mouth has 5 parts, each one a weight on a spring:

- **jaw**: how open the mouth is
- **lift**: how high the tongue goes
- **place**: how far back in the mouth the tongue is
- **breath**: air pushing out
- **voice**: vocal cords on or off

The speaker doesn't control the parts directly. It picks target positions, and the springs pull each part toward its target. Springs take time to move and can overshoot, so some sounds are harder than others. Holding a part away from its resting spot costs effort, and resting is free.

Some sounds fall out of the physics. If the jaw closes while air is building up and then opens, you get a burst, which works like a `p` or `b`.

## The objects

Each object has a color (8), a shape (3) and a size (4), so there are 96 objects total. The speaker has to describe all three things in one word.

## How they learn

- The speaker uses **REINFORCE**, a basic reinforcement learning method. Words that got guessed right become more likely.
- Reward = how much the listener believed the right answer, minus effort and time costs.
- The costs only count in proportion to how often the listener is already right. Early on, nobody understands anybody, so the speaker isn't punished for trying.
- Saying nothing always scores worse than saying something, so it can't just stay silent.
- It explores a lot at the start and less later.

## Running it

You need [uv](https://docs.astral.sh/uv/) and Node 20+.

**Train**
```bash
cd backend
uv sync
uv run python train.py pool=1 wobble=0.05 noise=0.1 run_name=first
```
Any setting in `backend/world/config.py` can be changed like `name=value`. Results are saved in `backend/runs/first/`.

**Play in the terminal**
```bash
uv run python play.py runs/first
```

**Play in the browser**
```bash
# terminal 1 (in backend/)
uv run python server.py

# terminal 2
cd frontend
npm install
npm run dev
```
Then go to http://localhost:3000. Click through: reveal the object, make the speaker talk, let the listener guess.

**Tests**
```bash
cd backend
uv run --with pytest python -m pytest tests --ignore=tests/test_gru.py
```

## Folders

```
backend/    the game, training, small server, tests
  world/    config, objects, mouth, noisy channel, speaker/listener
  util/     saving runs, turning mouth movements into IPA
  tests/
frontend/   the Next.js page for playing a round
```

## Settings that matter

| setting | default | what it does |
|---|---|---|
| `wobble` | 0.05 | how shaky the mouth is |
| `noise` | 0.1 | how noisy the channel is |
| `effort_cost` | 0.05 | price of moving the mouth |
| `time_cost` | 0.01 | price of taking longer |
| `max_slots` | 10 | how many mouth positions per word |
| `pool` | 4 | how many objects per training round |
| `reward_floor` | false | never score worse than a blind guess |

Experimental changes in the code are marked `# EXPERIMENT: name` so they're easy to find and remove.

## Results so far

Only one seed per run, so take these as hints and not proof.

- With a gentle mouth (`wobble=0.05`, `noise=0.1`) it reached about 92% accuracy on all 96 objects with 49 different words.
- Size got learned first, then shape, then color.
- It mostly used two stops (a `b` and a back-of-throat one) and two vowels.
- More wobble and noise made learning much slower and the words collapsed into long vowels.
- Two loopholes showed up and I patched them: staying silent, and mumbling too quietly to hear.

## Known issues

- `tests/test_gru.py` is broken (it imports a class that was removed).
- `backend/experiments/` is old and may not run on the current code.
- A saved run in `backend/runs/` may stop loading if the network's input size changes. Runs aren't in git.
- The quiet-mumble loophole isn't fully closed yet.
