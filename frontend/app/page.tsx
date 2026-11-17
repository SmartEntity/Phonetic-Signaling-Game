"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Beliefs } from "@/components/Beliefs";
import { MouthTraces } from "@/components/MouthTraces";
import { ObjectFigure } from "@/components/ObjectFigure";
import { SoundStrip } from "@/components/SoundStrip";
import type { Round, RunInfo } from "@/lib/types";

// How far through the round the page has shown. The whole round is played when the object is revealed;
// the later steps only uncover what is already there.
type Shown = "nothing" | "object" | "word" | "guess";
const ANY = -1;

const SERVER_HELP = "Can't reach the game server. In the backend folder, run: uv run python server.py";

export default function Page() {
  const [runs, setRuns] = useState<RunInfo[] | null>(null);
  const [runName, setRunName] = useState("");
  const [picks, setPicks] = useState<number[]>([]);
  const [round, setRound] = useState<Round | null>(null);
  const [shown, setShown] = useState<Shown>("nothing");
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");
  const [tally, setTally] = useState({ played: 0, right: 0 });

  useEffect(() => {
    fetch("/api/runs")
      .then((reply) => (reply.ok ? reply.json() : Promise.reject()))
      .then((found: RunInfo[]) => {
        setRuns(found);
        const first = found.find((run) => run.playable);
        if (first) setRunName(first.name);
      })
      .catch(() => setProblem(SERVER_HELP));
  }, []);

  const run = useMemo(() => runs?.find((r) => r.name === runName && r.playable) as Extract<RunInfo, { playable: true }> | undefined, [runs, runName]);
  const attributes = useMemo(() => Object.entries(run?.attributes ?? {}), [run]);
  const sizes = run?.attributes["size"];

  // A new run means a new set of attributes to choose from, and any round on screen belongs to the old one.
  useEffect(() => {
    setPicks(attributes.map(() => ANY));
    setRound(null);
    setShown("nothing");
    setTally({ played: 0, right: 0 });
  }, [attributes]);

  const reveal = useCallback(
    async (values?: number[]) => {
      if (!run) return;
      setBusy(true);
      setProblem("");
      const chosen = values ?? attributes.map(([, options], a) => (picks[a] === ANY || picks[a] === undefined ? Math.floor(Math.random() * options.length) : picks[a]));
      try {
        const reply = await fetch("/api/round", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ run: run.name, values: chosen }) });
        const body = await reply.json();
        if (!reply.ok) throw new Error(body.error ?? "The server could not play that round.");
        setRound(body as Round);
        setShown("object");
      } catch (error) {
        setProblem(error instanceof TypeError ? SERVER_HELP : (error as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [run, attributes, picks],
  );

  const showGuess = () => {
    if (!round) return;
    setShown("guess");
    setTally((t) => ({ played: t.played + 1, right: t.right + (round.listener.right ? 1 : 0) }));
  };

  const startOver = () => {
    setRound(null);
    setShown("nothing");
  };

  const spoken = shown === "word" || shown === "guess";
  const guessed = shown === "guess";
  const width = round ? Math.max(round.speaker.sound.length, round.listener.heard.length, 1) : 1;
  const loudest = round ? Math.max(0.2, ...round.speaker.sound.flat(), ...round.listener.heard.flat()) : 1;
  const missed = round ? Object.keys(round.object.attributes).filter((a) => round.object.attributes[a] !== round.listener.guess.attributes[a]) : [];

  return (
    <main>
      <header className="top">
        <h1>One round</h1>
        <label className="field">
          <span>Trained pair</span>
          <select value={runName} onChange={(event) => setRunName(event.target.value)} disabled={!runs || busy}>
            {!runs && <option>Loading</option>}
            {runs?.map((r) =>
              r.playable ? (
                <option key={r.name} value={r.name}>
                  {r.name} (step {r.step.toLocaleString()}, {Math.round(r.accuracy * 100)}% right in training)
                </option>
              ) : (
                <option key={r.name} value={r.name} disabled>
                  {r.name} (saved by older code)
                </option>
              ),
            )}
          </select>
        </label>
        <p className="tally" aria-live="polite">
          {tally.played === 0 ? "No rounds played yet" : `${tally.right} of ${tally.played} guessed right`}
        </p>
      </header>

      {problem && (
        <p className="problem" role="alert">
          {problem}
        </p>
      )}
      {runs && runs.length > 0 && !run && !problem && <p className="problem">None of the saved runs can be opened by the current code. Train a new one first.</p>}
      {runs && runs.length === 0 && <p className="problem">There are no saved runs yet. Train one with: uv run python train.py run_name=first</p>}

      <div className="round">
        <section className="panel" aria-labelledby="step-object">
          <h2 id="step-object">
            <span className="step">1</span> The object
          </h2>
          <ObjectFigure thing={shown === "nothing" ? null : (round?.object ?? null)} sizes={sizes} label="The object the speaker is shown" />
          {shown === "nothing" ? (
            <>
              <div className="choosers">
                {attributes.map(([name, options], a) => (
                  <label className="field" key={name}>
                    <span>{name}</span>
                    <select value={picks[a] ?? ANY} onChange={(event) => setPicks((old) => old.map((p, i) => (i === a ? Number(event.target.value) : p)))}>
                      <option value={ANY}>any</option>
                      {options.map((option, i) => (
                        <option key={option} value={i}>
                          {option}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
              <button className="act" onClick={() => reveal()} disabled={!run || busy}>
                {busy ? "Playing the round" : "Reveal object"}
              </button>
            </>
          ) : (
            round && (
              <p className="caption">
                {round.object.name}
                {round.object.held_out && <span className="aside"> (never seen in training)</span>}
              </p>
            )
          )}
        </section>

        <section className="panel panel-word" aria-labelledby="step-word">
          <h2 id="step-word">
            <span className="step">2</span> The word
          </h2>
          {!spoken && <p className="waiting">{shown === "nothing" ? "The speaker hasn't been shown anything yet." : "The speaker has seen the object and is waiting."}</p>}
          {shown === "object" && (
            <button className="act" onClick={() => setShown("word")}>
              Tell the speaker to speak
            </button>
          )}
          {spoken && round && (
            <>
              <p className="word" lang="und">
                {round.speaker.said || <span className="word-none">no sound</span>}
              </p>
              <p className="ipa">
                {round.speaker.said_ipa ? <span lang="und-fonipa">/{round.speaker.said_ipa}/</span> : "The mouth moved, but too quietly for any sound to be written down."}
                {round.speaker.meant !== round.speaker.said && round.speaker.meant && <span className="aside"> It was aiming for “{round.speaker.meant}”.</span>}
              </p>
              <h3>As it left the mouth</h3>
              <SoundStrip frames={round.speaker.sound} width={width} loudest={loudest} sweep={shown === "word"} label="The sound the speaker made" />
              <details>
                <summary>How the mouth moved</summary>
                <p className="detail">
                  {round.speaker.sound.length} frames of sound from {round.speaker.targets.filter((t) => t.heard).length} mouth targets, costing {round.speaker.effort.toFixed(1)} effort.
                  {round.speaker.in_words && <> In words: {round.speaker.in_words}.</>}
                </p>
                <MouthTraces speaker={round.speaker} />
              </details>
            </>
          )}
        </section>

        <section className="panel" aria-labelledby="step-guess">
          <h2 id="step-guess">
            <span className="step">3</span> The guess
          </h2>
          {!guessed && <ObjectFigure thing={null} hidden label="The listener's guess, not shown yet" />}
          {!spoken && <p className="waiting">Nothing has been said yet.</p>}
          {shown === "word" && (
            <button className="act" onClick={showGuess}>
              Let the listener guess
            </button>
          )}
          {guessed && round && (
            <>
              <ObjectFigure thing={round.listener.guess} sizes={sizes} label="The listener's guess" />
              <p className="caption">{round.listener.guess.name}</p>
              <p className={round.listener.right ? "verdict verdict-right" : "verdict verdict-wrong"} role="status">
                {round.listener.right ? "Right" : `Wrong ${missed.join(" and ")}`}
              </p>
              <h3>As it reached the listener</h3>
              <SoundStrip frames={round.listener.heard} width={width} loudest={loudest} label="The sound after noise and echo" />
              <h3>How sure it was</h3>
              <Beliefs round={round} />
            </>
          )}
        </section>
      </div>

      {guessed && round && (
        <div className="after">
          <button className="act" onClick={startOver}>
            New round
          </button>
          <button className="act act-quiet" onClick={() => reveal(round.object.values)} disabled={busy}>
            Same object again
          </button>
        </div>
      )}
    </main>
  );
}
