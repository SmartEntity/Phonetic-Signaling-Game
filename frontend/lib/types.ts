// What server.py sends back. One round holds everything; the page chooses what to show.

export type RunInfo =
  | { name: string; playable: true; step: number; accuracy: number; attributes: Record<string, string[]> }
  | { name: string; playable: false; why: string };

export type Thing = { name: string; values: number[]; attributes: Record<string, string> };

export type Target = { position: number[]; frames: number; tolerance: number; starts_at: number; heard: boolean };

export type Round = {
  run: string;
  step: number;
  object: Thing & { held_out: boolean };
  speaker: {
    meant: string;
    meant_ipa: string;
    said: string;
    said_ipa: string;
    in_words: string;
    effort: number;
    word_starts_at: number;
    sound: number[][];
    mouth: number[][];
    articulators: string[];
    targets: Target[];
    tightest: number;
    loosest: number;
  };
  listener: {
    heard: number[][];
    guess: Thing;
    beliefs: { attribute: string; options: { value: string; chance: number }[] }[];
    right: boolean;
    share_right: number;
  };
};
