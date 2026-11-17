# Frontend

A Next.js page for walking through one round of the game: reveal the object, tell the speaker to speak, let the listener guess.

It needs the Python server running, from `../backend`:

```bash
uv run python server.py
```

Then here:

```bash
npm install
npm run dev
```

Open http://localhost:3000. The page asks `/api/...`, and `next.config.mjs` passes that on to the server at http://127.0.0.1:8000. To use another address, set `GAME_SERVER` before `npm run dev`.

The server plays the whole round at once. The page decides what to show and when.
