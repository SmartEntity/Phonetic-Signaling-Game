"use client";

import { useEffect, useState } from "react";

// A sound drawn the way phonetics labs print it: time runs left to right, pitch bottom to top, and darker means louder.
// `frames` is one row of band loudnesses per frame. `width` (in frames) keeps two strips on the same time scale.
export function SoundStrip({ frames, width, loudest, sweep, label }: { frames: number[][]; width: number; loudest: number; sweep?: boolean; label: string }) {
  const bands = frames[0]?.length ?? 8;
  const cell = 12;
  const row = 9;
  const [shown, setShown] = useState(sweep ? 0 : frames.length);

  // When asked to sweep, uncover the sound one frame at a time, as if it were being spoken now.
  useEffect(() => {
    const still = typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!sweep || still) {
      setShown(frames.length);
      return;
    }
    setShown(0);
    let frame = 0;
    const timer = setInterval(() => {
      frame += 1;
      setShown(frame);
      if (frame >= frames.length) clearInterval(timer);
    }, 45);
    return () => clearInterval(timer);
  }, [frames, sweep]);

  if (frames.length === 0) return <p className="quiet-note">Nothing audible.</p>;

  return (
    <svg className="strip" viewBox={`0 0 ${width * cell} ${bands * row}`} style={{ width: `min(100%, ${width * cell}px)` }} role="img" aria-label={label}>
      <rect x="0" y="0" width={width * cell} height={bands * row} className="strip-paper" />
      {frames.slice(0, shown).map((frame, t) =>
        frame.map((value, band) => {
          const ink = Math.max(0, Math.min(1, value / loudest));
          return ink < 0.03 ? null : <rect key={`${t}-${band}`} x={t * cell} y={(bands - 1 - band) * row} width={cell} height={row} className="strip-ink" opacity={ink} />;
        }),
      )}
    </svg>
  );
}
