import type { Round } from "@/lib/types";

// For each attribute, how sure the listener was of every option. The one it chose is filled; the true one is named.
export function Beliefs({ round }: { round: Round }) {
  return (
    <div className="beliefs">
      {round.listener.beliefs.map(({ attribute, options }) => {
        const truth = round.object.attributes[attribute];
        const guess = round.listener.guess.attributes[attribute];
        const ranked = [...options].sort((a, b) => b.chance - a.chance).filter((o, i) => i < 3 || o.value === truth);
        return (
          <div className="belief" key={attribute}>
            <h4>
              {attribute}
              {guess !== truth && <span className="belief-miss"> was {truth}</span>}
            </h4>
            <ul>
              {ranked.map((option) => (
                <li key={option.value} className={option.value === guess ? "chosen" : ""}>
                  <span className="belief-value">{option.value}</span>
                  <span className="belief-bar" aria-hidden="true">
                    <span style={{ width: `${Math.round(option.chance * 100)}%` }} />
                  </span>
                  <span className="belief-chance">{Math.round(option.chance * 100)}%</span>
                </li>
              ))}
            </ul>
          </div>
        );
      })}
    </div>
  );
}
