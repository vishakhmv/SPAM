"""TIMIT phone reading and closure-merge logic.

This module is intentionally standalone: the closure-merge helpers are useful
beyond evaluation (e.g. preparing ground-truth segments to train supervised
segmentation/recognition models). It reads the distributed TIMIT ``.phn``
files directly — sample-indexed ``start stop phone`` lines at 16 kHz — and
maps the 61-symbol TIMIT phone set to IPA.

The closure handling mirrors the standard evaluation convention: TIMIT writes
most stops/affricates as a *closure* interval (``bcl``, ``dcl``, ...)
immediately followed by a *release* (``b``, ``jh``, ...). ``merge_stop_closures``
folds each closure into the following release span and drops the closure row;
a stranded closure falls back to its bare stop label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SR = 16000

# Silence token used downstream (matches the prepared-CSV convention).
SILENCE = "_"

# Non-speech TIMIT symbols stripped at utterance edges before scoring.
TIMIT_SILENCE_PHNS = frozenset({"h#", "pau", "epi"})

TIMIT_TO_IPA = {
    # Stops
    "b": "b",
    "d": "d",
    "g": "ɡ",
    "p": "p",
    "t": "t",
    "k": "k",
    "dx": "ɾ",
    "q": "ʔ",
    # Affricates
    "jh": "d͡ʒ",
    "ch": "t͡ʃ",
    # Fricatives
    "s": "s",
    "sh": "ʃ",
    "z": "z",
    "zh": "ʒ",
    "f": "f",
    "th": "θ",
    "v": "v",
    "dh": "ð",
    # Nasals
    "m": "m",
    "n": "n",
    "ng": "ŋ",
    "em": "m̩",
    "en": "n̩",
    "eng": "ŋ̩",
    "nx": "ɾ̃",
    # Semivowels and glides
    "l": "l",
    "r": "ɹ",
    "w": "w",
    "y": "j",
    "hh": "h",
    "hv": "ɦ",
    "el": "l̩",
    # Vowels
    "iy": "i",
    "ih": "ɪ",
    "eh": "ɛ",
    "ae": "æ",
    "aa": "ɑ",
    "ah": "ʌ",
    "ao": "ɔ",
    "uh": "ʊ",
    "uw": "u",
    "ux": "ʉ",
    "er": "ɜ˞",
    "ax": "ə",
    "ix": "ɨ",
    "axr": "ə˞",
    "ax-h": "ə̥",
    # Diphthongs
    "ey": "eɪ",
    "aw": "aʊ",
    "ay": "aɪ",
    "oy": "ɔɪ",
    "ow": "oʊ",
    # Stop closures: dropped after being merged into the succeeding stop, or
    # falling back to the bare stop label when stranded.
    "bcl": "b",
    "dcl": "d",
    "gcl": "ɡ",
    "pcl": "p",
    "tcl": "t",
    "kcl": "k",
    # Non-speech, mapped to the silence token.
    "pau": SILENCE,
    "epi": SILENCE,
    "h#": SILENCE,
}

# Release token -> its preceding closure token.
TIMIT_CLOSURE_OF = {
    "b": "bcl",
    "d": "dcl",
    "g": "gcl",
    "p": "pcl",
    "t": "tcl",
    "k": "kcl",
    "jh": "dcl",
    "ch": "tcl",
}
# The closure tokens themselves (the values of TIMIT_CLOSURE_OF).
TIMIT_CLOSURE_PHNS = frozenset(TIMIT_CLOSURE_OF.values())


@dataclass
class Seg:
    """A single phone interval (seconds).

    ``raw_label`` is the dataset-native token (TIMIT phone such as ``tcl``;
    or the VoxAngeles TextGrid token). ``ipa_label`` is the IPA mapping, the
    silence token ``"_"`` for non-speech, or ``None`` for an interval left
    deliberately unlabeled (an unmerged TIMIT closure — the ``timit-raw``
    training substrate).
    """

    start: float
    end: float
    raw_label: str
    ipa_label: str | None


def read_phn(phn_path: str | Path) -> list[tuple[int, int, str]]:
    """Read a distributed TIMIT ``.phn`` file as ``(start, stop, phn)`` rows.

    ``start``/``stop`` are raw sample indices at 16 kHz.
    """
    rows = []
    with open(phn_path) as f:
        for line in f:
            start_str, stop_str, phn = line.strip().split()
            rows.append((int(start_str), int(stop_str), phn))
    return rows


def merge_stop_closures(segs: list[Seg]) -> list[Seg]:
    """Fold each stop closure into the stop/affricate release after it.

    When a closure (``bcl``, ``dcl``, ...) is immediately followed by its
    matching release (``b``, ``jh``, ...), the two merge into one segment
    spanning ``[closure.start, release.end]``, labeled as the release; the
    closure row is dropped. A stranded closure is kept and surfaces as its
    bare stop label (already mapped via ``TIMIT_TO_IPA``).

    ``segs`` must be time-ordered and belong to a single utterance.
    """
    merged: list[Seg] = []
    for seg in segs:
        phn = seg.raw_label
        if merged and phn in TIMIT_CLOSURE_OF and merged[-1].raw_label == TIMIT_CLOSURE_OF[phn]:
            # Previous row is this release's closure: extend the release back
            # over the closure's span and replace the closure row with it.
            merged[-1] = Seg(merged[-1].start, seg.end, seg.raw_label, seg.ipa_label)
        else:
            merged.append(seg)
    return merged


def merge_adjacent_silence(segs: list[Seg]) -> list[Seg]:
    """Coalesce consecutive silence segments into one interval."""
    merged: list[Seg] = []
    for seg in segs:
        if merged and merged[-1].ipa_label == SILENCE and seg.ipa_label == SILENCE:
            merged[-1] = Seg(merged[-1].start, seg.end, merged[-1].raw_label, SILENCE)
        else:
            merged.append(seg)
    return merged


def timit_segments(phn_path: str | Path, merge_closures: bool = True) -> list[Seg]:
    """Read a TIMIT ``.phn`` into IPA-labeled segments (seconds).

    With ``merge_closures=True`` (default) closures are folded into their
    releases — the standard evaluation-boundary convention. With
    ``merge_closures=False`` every ``.phn`` interval is kept as its own
    segment (closures separate), so the closure->release boundaries survive;
    each standalone closure interval is left ``ipa_label=None`` (the
    ``timit-raw`` training substrate, where closure/release labeling is
    resolved downstream from ``raw_label``). Adjacent silence is always
    coalesced.
    """
    segs = []
    for start, stop, phn in read_phn(phn_path):
        # Unmerged closures stay unlabeled (ipa_label=None); merging instead
        # folds them into the following release via merge_stop_closures.
        ipa = None if (not merge_closures and phn in TIMIT_CLOSURE_PHNS) else TIMIT_TO_IPA[phn]
        segs.append(Seg(start / SR, stop / SR, phn, ipa))
    if merge_closures:
        segs = merge_stop_closures(segs)
    return merge_adjacent_silence(segs)
