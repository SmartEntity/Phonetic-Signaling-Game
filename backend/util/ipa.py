#Thank you Claude
"""Writes what a mouth did as IPA.

Two readings share one classifier:
    to_ipa(postures)       what the mouth actually did, so what a listener could hear
    plan_to_ipa(targets)   what the speaker aimed for, read as if the mouth were perfect

Every reading follows the same steps:
    1. Track    one row of measurements per frame (gap, loudness, place, breath, voice, burst)
    2. landmarks   the frames where the mouth opens wide (vowel peaks) or closes (consonants)
    3. classify    each landmark becomes one symbol: vowel, stop, tap, fricative or approximant
    4. name_of     any symbol can be spelled out in words, "t̠" is a voiceless postalveolar stop
"""
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

from world.anatomy.base import Mouth
from world.anatomy.voicebox import VoiceBox, PLACE, BREATH, VOICE
from world.datatypes import Object, Posture
from world.players.base import ArticulatedSpeaker

PLACES = ["bilabial", "dental", "alveolar", "postalveolar", "palatal", "velar", "uvular"]
CORONAL = ("dental", "alveolar", "postalveolar")

STOPS = {
    "bilabial": ("p", "b"), "dental": ("t̪", "d̪"), "alveolar": ("t", "d"), "postalveolar": ("t̠", "d̠"),
    "palatal": ("c", "ɟ"), "velar": ("k", "g"), "uvular": ("q", "ɢ"),
}
FRICATIVES = {
    "bilabial": ("ɸ", "β"), "dental": ("θ", "ð"), "alveolar": ("s", "z"), "postalveolar": ("ʃ", "ʒ"),
    "palatal": ("ç", "ʝ"), "velar": ("x", "ɣ"), "uvular": ("χ", "ʁ"),
}
APPROXIMANTS = {
    "bilabial": "β̞", "dental": "ð̞", "alveolar": "ɹ", "postalveolar": "ɹ̠",
    "palatal": "j", "velar": "ɰ", "uvular": "ʁ̞",
}
TAP = "ɾ"
PLAIN_VOICELESS_STOPS = {voiceless for voiceless, _ in STOPS.values()}

BACKNESS = ["front", "central", "back"]
HEIGHTS = [(0.2, "close"), (0.35, "close-mid"), (0.55, "open-mid"), (np.inf, "open")]
VOWELS = {
    "close": {"front": "i", "central": "ɨ", "back": "ɯ"},
    "close-mid": {"front": "e", "central": "ɘ", "back": "ɤ"},
    "open-mid": {"front": "ɛ", "central": "ɜ", "back": "ʌ"},
    "open": {"front": "a", "central": "ɐ", "back": "ɑ"},
}

VOICELESS = "̥"
TIE = "͡"
ASPIRATED = "ʰ"
LONG = "ː"
UNRELEASED = "̚"
BREAK = "."

# How wide the gap between the articulators is, as `opening` measures it
SHUT = 0.03        # below this the mouth is closed and no air gets out
NARROW = 0.15      # below this (but not shut) the air is squeezed: hiss or glide
LOOSE = 0.3        # above this the mouth is a vowel, never a consonant

# How much the vocal folds buzz
VOICED = 0.5       # voice above this is a buzz
WHISPERED = 0.3    # voice below this is no buzz at all, only breath

QUIET = 0.05       # total loudness below this counts as silence
MIN_SWING = 0.08   # the gap must move this much to count as a gesture, smaller is a wobble
RELEASE_BREATH = 0.1   # breath above this puts air pressure behind a closure
BURST_HEARD = 0.02     # burst above this is an audible pop when a closure opens
VOWEL_EDGE = 0.5       # a vowel lasts while the gap stays above this fraction of its peak

TAP_FRAMES = 2         # a closure this short or shorter, voiced and coronal, is a tap
AFFRICATE_FRAMES = 3   # hiss after a stop this long makes it an affricate
ASPIRATION_FRAMES = 3  # breathy frames this long make a puff of air, ʰ or h
LONG_FRAMES = 12       # a vowel this long gets ː


@dataclass(frozen=True)
class Track:
    """What the mouth was doing in each frame, one array entry per frame."""

    opening: np.ndarray
    loud: np.ndarray
    burst: np.ndarray
    place: np.ndarray
    breath: np.ndarray
    voice: np.ndarray
    whispery: np.ndarray     # air passes with no buzz, so a whisper or an h
    hissing: np.ndarray      # air passes through a narrow gap, so a hiss
    duration: np.ndarray     # frames each entry lasts: 1 for a trajectory, the travel time of each planned target

    @staticmethod
    def of(postures: list[Posture], mouth: Mouth, frames: int | Sequence[int] = 1) -> "Track":
        x = np.array([p.position for p in postures])
        opening = np.array([VoiceBox.opening(row) for row in x])
        loud = np.array([sum(mouth.sound(p).f) for p in postures])
        place, breath, voice = (np.clip(x[:, i], 0, 1) for i in (PLACE, BREATH, VOICE))
        return Track(
            opening=opening, loud=loud, burst=np.array([p.burst for p in postures]),
            place=place, breath=breath, voice=voice,
            whispery=(loud >= QUIET) & (opening >= NARROW) & (voice < WHISPERED),
            hissing=(opening >= SHUT) & (opening < NARROW) & (breath > RELEASE_BREATH) & (loud >= QUIET),
            duration=np.broadcast_to(np.asarray(frames, dtype=float), (len(postures),)).copy(),
        )

    def __len__(self) -> int:
        return len(self.opening)

    def place_name(self, t: int) -> str:
        return PLACES[min(int(self.place[t] * len(PLACES)), len(PLACES) - 1)]

    def backness(self, t: int) -> str:
        return BACKNESS[min(int(self.place[t] * len(BACKNESS)), len(BACKNESS) - 1)]


class Segment(NamedTuple):
    frame: int
    kind: str        # "V" for a vowel peak, "C" for a closing or narrowing
    symbol: str


def turning_points(opening: np.ndarray, min_swing: float = MIN_SWING) -> list[tuple[int, str]]:
    """Frames where the gap peaks ("open") or dips ("narrow"), ignoring wobbles under min_swing."""
    points = []
    rising: bool | None = None       # None until the gap has moved by min_swing for the first time
    extreme = 0                      # most extreme frame so far in the current direction
    for t in range(1, len(opening)):
        if rising is None:
            if abs(opening[t] - opening[0]) > min_swing:
                rising, extreme = opening[t] > opening[0], t
        elif rising:
            if opening[t] > opening[extreme]:
                extreme = t
            elif opening[extreme] - opening[t] > min_swing:
                points.append((extreme, "open"))
                rising, extreme = False, t
        else:
            if opening[t] < opening[extreme]:
                extreme = t
            elif opening[t] - opening[extreme] > min_swing:
                points.append((extreme, "narrow"))
                rising, extreme = True, t
    if rising is not None:
        points.append((extreme, "open" if rising else "narrow"))
    return points


def landmarks(track: Track, rest: float) -> tuple[list[tuple[int, str]], bool]:
    """The frames that become symbols, plus whether the word was cut off while still closing."""
    # the mouth starts at rest, so the first gesture is measured from there
    points = [(t - 1, kind) for t, kind in turning_points(np.concatenate([[rest], track.opening])) if t > 0]
    cut_off = bool(points) and points[-1] == (len(track) - 1, "narrow")
    # a dip that stays wide is a vowel changing, not a consonant
    points = [(t, kind) for t, kind in points if kind == "open" or track.opening[t] < LOOSE]
    cut_off = cut_off and bool(points) and points[-1][1] == "narrow"

    # two openings in a row with no dip between them are one vowel, keep the wider
    merged = []
    for t, kind in points:
        if merged and merged[-1][1] == kind == "open":
            if track.opening[t] > track.opening[merged[-1][0]]:
                merged[-1] = (t, kind)
        else:
            merged.append((t, kind))
    # a sound with no opening gesture (a breath at rest) is still one vowel-like peak at its loudest,
    # but a shut or squeezed mouth leaking air is not a vowel
    loudest = int(track.loud.argmax()) if len(track) else 0
    if not any(kind == "open" for _, kind in merged) and len(track) and track.loud[loudest] >= QUIET and track.opening[loudest] >= NARROW:
        merged.append((loudest, "open"))
        merged.sort()
    return merged, cut_off


def run_around(mask: np.ndarray, t: int) -> tuple[int, int]:
    """The stretch of consecutive True frames in mask containing t, as (first, one past last)."""
    if not mask[t]:
        return t, t + 1
    lo = hi = t
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    while hi < len(mask) - 1 and mask[hi + 1]:
        hi += 1
    return lo, hi + 1


def vowel(track: Track, t: int, before: int, out: list[Segment]) -> str:
    """The vowel at frame t. May also give the previous stop a puff of air, in out."""
    if track.loud[t] < QUIET:
        return BREAK
    height = next(name for limit, name in HEIGHTS if track.opening[t] < limit)
    symbol = VOWELS[height][track.backness(t)]
    lo, hi = run_around((track.opening >= max(NARROW, VOWEL_EDGE * track.opening[t])) & (track.loud >= QUIET), t)
    breathy = (track.whispery * track.duration)[before:hi].sum() >= ASPIRATION_FRAMES
    previous = out[-1].symbol if out else BREAK

    if track.voice[lo:hi].mean() < WHISPERED:
        symbol += VOICELESS
    elif breathy and previous in PLAIN_VOICELESS_STOPS:
        out[-1] = out[-1]._replace(kind="C", symbol=previous + ASPIRATED)
    elif breathy and previous == BREAK:
        symbol = "h" + symbol
    if track.duration[lo:hi].sum() >= LONG_FRAMES:
        symbol += LONG
    return symbol


def is_released(track: Track, t: int, closed_until: int, until: int, planned: bool) -> bool:
    """Does the closure at t open again with a pop?

    The plan is read as an ideal mouth would do it: air is pushed and something follows the closure.
    The actual mouth has to make a measurable burst.
    """
    if planned:
        return bool(track.breath[t] > RELEASE_BREATH) and closed_until < len(track)
    return bool(track.burst[t:until + 1].max() > BURST_HEARD)


def closure(track: Track, t: int, until: int, planned: bool) -> str:
    """The consonant where the mouth shuts: a stop, a tap, or nothing if no air got out."""
    lo, hi = run_around(track.opening < SHUT, t)
    held = track.duration[lo:hi].sum()
    where = track.place_name(hi - 1)
    voiced = int(track.voice[(lo + hi) // 2] > VOICED)

    if held <= TAP_FRAMES and voiced and where in CORONAL:
        return TAP
    if not is_released(track, t, hi, until, planned):
        # a planned stop at the end of the word is still a stop, just never opened: p̚
        pressed = track.breath[t] > RELEASE_BREATH
        return STOPS[where][voiced] + UNRELEASED if planned and pressed else BREAK
    symbol = STOPS[where][voiced]
    if (track.hissing * track.duration)[t + 1:until + 1].sum() >= AFFRICATE_FRAMES:
        symbol += TIE + FRICATIVES[where][voiced]
    return symbol


def narrowing(track: Track, t: int, cut_off: bool) -> str:
    """The consonant where the mouth narrows without shutting: a fricative or approximant."""
    where = track.place_name(t)
    voiced = int(track.voice[t] > VOICED)
    if track.loud[t] < QUIET:
        return BREAK
    if track.opening[t] < NARROW and track.breath[t] > RELEASE_BREATH:
        return FRICATIVES[where][voiced]
    if cut_off:
        return BREAK     # the word ended before the mouth finished closing, so no gesture was completed
    return APPROXIMANTS[where] + (VOICELESS if track.voice[t] < WHISPERED else "")


def segments(postures: list[Posture], mouth: Mouth | None = None, planned: bool = False,
             frames_per_posture: int | Sequence[int] = 1) -> list[Segment]:
    """One Segment per vowel peak or consonant gesture, silent ones included with symbol BREAK.

    planned=True reads targets instead of the executed trajectory. frames_per_posture is how long each target is held,
    one number for all of them or one per target.
    """
    if not postures:
        return []
    mouth = mouth or VoiceBox()
    track = Track.of(postures, mouth, frames_per_posture)
    points, cut_off = landmarks(track, VoiceBox.opening(mouth.rest.position))

    out: list[Segment] = []
    for i, (t, kind) in enumerate(points):
        before = points[i - 1][0] + 1 if i else 0
        until = points[i + 1][0] if i + 1 < len(points) else len(track) - 1
        if kind == "open":
            symbol = vowel(track, t, before, out)
        elif track.opening[t] < SHUT:
            symbol = closure(track, t, until, planned)
        else:
            symbol = narrowing(track, t, cut_off and i == len(points) - 1)
        out.append(Segment(t, "V" if kind == "open" else "C", symbol))
    return out


def sounds(postures: list[Posture], mouth: Mouth | None = None, planned: bool = False, frames_per_posture: int | Sequence[int] = 1) -> list[str]:
    return [s.symbol for s in segments(postures, mouth, planned, frames_per_posture) if s.symbol != BREAK]


def to_ipa(postures: list[Posture], mouth: Mouth | None = None, planned: bool = False, frames_per_posture: int | Sequence[int] = 1,
           simple: bool = False) -> str:
    """The word in IPA, or with simple=True in everyday letters: "pa" and "pah" instead of "pa" and "pɐ"."""
    word = ""
    for _, _, symbol in segments(postures, mouth, planned, frames_per_posture):
        if symbol != BREAK or not word.endswith(BREAK):
            word += plain(symbol) if simple else symbol
    return word.strip(BREAK)


def plan_to_ipa(targets: list[Posture], mouth: Mouth | None = None, hold: int | Sequence[int] = 4, simple: bool = False) -> str:
    return to_ipa(targets, mouth, planned=True, frames_per_posture=hold, simple=simple)


def shape(postures: list[Posture], mouth: Mouth | None = None, planned: bool = False, frames_per_posture: int | Sequence[int] = 1) -> str:
    """The word's rhythm as C and V, "CVCV" for papa."""
    return "".join(s.kind for s in segments(postures, mouth, planned, frames_per_posture) if s.symbol != BREAK)


def _names() -> dict[str, str]:
    names = {TAP: "alveolar tap"}
    for place in PLACES:
        for voiced, label in enumerate(("voiceless", "voiced")):
            names[STOPS[place][voiced]] = f"{label} {place} stop"
            names[FRICATIVES[place][voiced]] = f"{label} {place} fricative"
        names[APPROXIMANTS[place]] = f"{place} approximant"
    for height, row in VOWELS.items():
        for backness, symbol in row.items():
            names[symbol] = f"{height} {backness} vowel"
    return names


NAMES = _names()


def name_of(symbol: str) -> str:
    """A symbol in words: "t̠" is "voiceless postalveolar stop", "ɨ̥ː" is "long voiceless close central vowel"."""
    if symbol in NAMES:
        return NAMES[symbol]
    if symbol.endswith(LONG):
        return "long " + name_of(symbol[:-len(LONG)])
    if symbol.endswith(VOICELESS):
        return "voiceless " + name_of(symbol[:-len(VOICELESS)])
    if symbol.endswith(ASPIRATED):
        return "aspirated " + name_of(symbol[:-len(ASPIRATED)])
    if symbol.endswith(UNRELEASED):
        return "unreleased " + name_of(symbol[:-len(UNRELEASED)])
    if TIE in symbol:
        stop, _ = symbol.split(TIE)
        return name_of(stop).replace(" stop", " affricate")
    if symbol.startswith("h") and len(symbol) > 1:
        return "breath then " + name_of(symbol[1:])
    return "?"


# A plainer spelling for reading at a glance, no special symbols. The IPA stays the record, this is for display.
PLAIN_STOPS = {"palatal": ("ky", "gy"), "uvular": ("q", "G")}
PLAIN_FRICATIVES = {
    "bilabial": ("f", "v"), "dental": ("th", "dh"), "alveolar": ("s", "z"), "postalveolar": ("sh", "zh"),
    "palatal": ("hy", "zy"), "velar": ("kh", "gh"), "uvular": ("qh", "rh"),
}
PLAIN_APPROXIMANTS = {
    "bilabial": "w", "dental": "r", "alveolar": "r", "postalveolar": "r", "palatal": "y", "velar": "gw", "uvular": "rh",
}
PLAIN_VOWELS = {
    "close": {"front": "ee", "central": "ih", "back": "oo"},
    "close-mid": {"front": "ay", "central": "eu", "back": "o"},
    "open-mid": {"front": "eh", "central": "er", "back": "uh"},
    "open": {"front": "a", "central": "ah", "back": "aw"},
}


def _plain() -> dict[str, str]:
    plain = {BREAK: BREAK, TAP: "dd"}
    for place in PLACES:
        for voiced in (0, 1):
            plain[STOPS[place][voiced]] = PLAIN_STOPS.get(place, (STOPS[place][0][0], STOPS[place][1][0]))[voiced]
            plain[FRICATIVES[place][voiced]] = PLAIN_FRICATIVES[place][voiced]
        plain[APPROXIMANTS[place]] = PLAIN_APPROXIMANTS[place]
    for height, row in VOWELS.items():
        for backness, symbol in row.items():
            plain[symbol] = PLAIN_VOWELS[height][backness]
    return plain


PLAIN = _plain()


def plain(symbol: str) -> str:
    """A symbol in everyday letters: "t̠" is "t", "ɐ" is "ah", "ɨ̥" is "~ih~" (whispered), "p̚" is "(p)" (never opened)."""
    if symbol in PLAIN:
        return PLAIN[symbol]
    if symbol.endswith(LONG):
        inner = plain(symbol[:-len(LONG)])
        return inner[:-1] + ":~" if inner.endswith("~") else inner + ":"
    if symbol.endswith(VOICELESS):
        return "~" + plain(symbol[:-len(VOICELESS)]) + "~"
    if symbol.endswith(ASPIRATED):
        return plain(symbol[:-len(ASPIRATED)]) + "'"
    if symbol.endswith(UNRELEASED):
        return "(" + plain(symbol[:-len(UNRELEASED)]) + ")"
    if TIE in symbol:
        return "".join(plain(part) for part in symbol.split(TIE))
    if symbol.startswith("h") and len(symbol) > 1:
        return "h" + plain(symbol[1:])
    return "?"


def explain(postures: list[Posture], mouth: Mouth | None = None, planned: bool = False, frames_per_posture: int | Sequence[int] = 1) -> str:
    """The word spelled out sound by sound: "p voiceless bilabial stop, ɐ open central vowel"."""
    heard = [s.symbol for s in segments(postures, mouth, planned, frames_per_posture) if s.symbol != BREAK]
    return ", ".join(f"{symbol} {name_of(symbol)}" for symbol in heard)


def lexicon(speaker: ArticulatedSpeaker, objects: list[Object], simple: bool = False) -> dict[Object, tuple[str, str]]:
    wobble, speaker.wobble = speaker.wobble, 0.0
    try:
        words = {}
        for obj in objects:
            speaker.act(obj)
            words[obj] = (plan_to_ipa(speaker.get_targets(), speaker.mouth, speaker.get_holds(), simple),
                          to_ipa(speaker.get_postures(), speaker.mouth, simple=simple))
        return words
    finally:
        speaker.wobble = wobble


def inventory(speaker: ArticulatedSpeaker, objects: list[Object]) -> Counter:
    wobble, speaker.wobble = speaker.wobble, 0.0
    try:
        count = Counter()
        for obj in objects:
            speaker.act(obj)
            count.update(sounds(speaker.get_postures(), speaker.mouth))
        return count
    finally:
        speaker.wobble = wobble
