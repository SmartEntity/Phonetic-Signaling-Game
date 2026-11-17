import type { Round } from "@/lib/types";

// One line per part of the mouth, across the frames of the word. The ticks underneath mark where each new target began.
export function MouthTraces({ speaker }: { speaker: Round["speaker"] }) {
  const frames = speaker.mouth.length;
  if (frames === 0) return null;
  const step = 12;
  const height = 30;
  const width = Math.max(1, frames - 1) * step;

  return (
    <div className="traces">
      {speaker.articulators.map((name, part) => (
        <div className="trace" key={name}>
          <span className="trace-name">{name}</span>
          <svg viewBox={`0 0 ${width} ${height}`} style={{ width: `min(100%, ${width}px)` }} preserveAspectRatio="none" role="img" aria-label={`${name} over the word`}>
            <line x1="0" y1={height - 1} x2={width} y2={height - 1} className="trace-floor" />
            {speaker.targets
              .filter((target) => target.heard && target.starts_at > 0 && target.starts_at < frames)
              .map((target) => (
                <line key={target.starts_at} x1={target.starts_at * step} y1="0" x2={target.starts_at * step} y2={height} className="trace-tick" />
              ))}
            <polyline
              className="trace-line"
              points={speaker.mouth.map((position, t) => `${t * step},${(height - 3) * (1 - Math.max(0, Math.min(1, position[part]))) + 1.5}`).join(" ")}
            />
          </svg>
        </div>
      ))}
    </div>
  );
}
