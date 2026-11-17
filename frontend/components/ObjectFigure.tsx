import type { Thing } from "@/lib/types";

// The colours the game's objects come in. Anything else is drawn in grey with its name underneath.
const COLOURS: Record<string, string> = {
  red: "#D8322B",
  yellow: "#F2CF1D",
  green: "#2FA35A",
  cyan: "#27C4D9",
  blue: "#2647C9",
  magenta: "#D43AB6",
  black: "#0E1418",
  white: "#FBFCFD",
  orange: "#EE8A1F",
  purple: "#7B3FC4",
};

function find(attributes: Record<string, string>, ...names: string[]): string | undefined {
  for (const name of names) if (name in attributes) return attributes[name];
  return undefined;
}

// How big to draw it: the position of its size among all the sizes, smallest first.
function scale(thing: Thing, sizes: string[] | undefined): number {
  const size = find(thing.attributes, "size");
  if (!size || !sizes || sizes.length < 2) return 0.7;
  const place = Math.max(0, sizes.indexOf(size));
  return 0.34 + (0.62 * place) / (sizes.length - 1);
}

export function ObjectFigure({ thing, sizes, hidden, label }: { thing: Thing | null; sizes?: string[]; hidden?: boolean; label: string }) {
  if (hidden || !thing) {
    return (
      <div className="stage stage-hidden" role="img" aria-label={label}>
        <svg viewBox="0 0 200 150" aria-hidden="true">
          <defs>
            <pattern id="cloth" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(35)">
              <line x1="0" y1="0" x2="0" y2="8" stroke="currentColor" strokeWidth="1.5" />
            </pattern>
          </defs>
          <path d="M40 132 L52 46 Q100 14 148 46 L160 132 Z" fill="url(#cloth)" opacity="0.55" />
        </svg>
      </div>
    );
  }

  const colour = COLOURS[find(thing.attributes, "color", "colour") ?? ""] ?? "#AEB8C0";
  const shape = find(thing.attributes, "shape");
  const r = 62 * scale(thing, sizes);
  const cx = 100;
  const cy = 75;
  const common = { fill: colour, stroke: "rgba(14,20,24,0.35)", strokeWidth: 1 };

  return (
    <div className="stage" role="img" aria-label={`${label}: ${thing.name}`}>
      <svg viewBox="0 0 200 150" aria-hidden="true">
        {shape === "circle" && <circle cx={cx} cy={cy} r={r} {...common} />}
        {shape === "square" && <rect x={cx - r} y={cy - r} width={2 * r} height={2 * r} {...common} />}
        {shape === "triangle" && (
          <polygon points={`${cx},${cy - r} ${cx + r * 0.95},${cy + r * 0.78} ${cx - r * 0.95},${cy + r * 0.78}`} {...common} />
        )}
        {!["circle", "square", "triangle"].includes(shape ?? "") && <rect x={cx - r} y={cy - r * 0.6} width={2 * r} height={1.2 * r} rx={r * 0.3} {...common} />}
      </svg>
    </div>
  );
}
