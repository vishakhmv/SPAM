"""TIMIT dataset loader and phoneme alignment parser for SPAM reproduction.

Implements data loading, 61-phone to IPA mapping, stop closure/release labeling
for phonological vector training, stop-closure merging for evaluation, and center-frame
calculation c(s).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import soundfile as sf
import numpy as np

from src.config import Config, default_config
from phone_metrics.timit import (
    SR,
    SILENCE,
    TIMIT_TO_IPA,
    TIMIT_CLOSURE_OF,
    TIMIT_CLOSURE_PHNS,
    TIMIT_SILENCE_PHNS,
    read_phn,
    merge_stop_closures,
    merge_adjacent_silence,
    Seg,
)
from phone_metrics.datasets import boundary_secs, canonical_ipa

# Tokens that represent stop/affricate releases when preceded by closure
TIMIT_RELEASE_PHNS = frozenset(TIMIT_CLOSURE_OF.keys())

# Precomputed mapping from TIMIT phone to canonical IPA for O(1) lookups
TIMIT_TO_CANONICAL_IPA = {phn: canonical_ipa(ipa) for phn, ipa in TIMIT_TO_IPA.items()}


@dataclass
class PhoneSegment:
    """Represents an individual phone segment in an utterance."""
    start_sample: int
    stop_sample: int
    raw_label: str
    ipa_label: Optional[str]
    is_silence: bool
    is_closure: bool
    is_release: bool

    @property
    def start_time(self) -> float:
        return self.start_sample / SR

    @property
    def stop_time(self) -> float:
        return self.stop_sample / SR

    @property
    def center_sample(self) -> int:
        return (self.start_sample + self.stop_sample) // 2

    @property
    def center_time(self) -> float:
        return (self.start_time + self.stop_time) / 2.0

    def center_frame(self, stride_samples: int = 320) -> int:
        """Frame index corresponding to temporal center c(s)."""
        return self.center_sample // stride_samples


@dataclass
class TimitUtterance:
    """Represents a single TIMIT audio file with phone alignments."""
    utterance_id: str
    wav_path: Path
    phn_path: Path
    split: str  # "train" or "test"
    speaker_id: str
    dialect_region: str
    is_sa_sentence: bool
    num_samples: int
    segments_training: List[PhoneSegment]  # Unmerged closures for channel estimation
    segments_eval: List[Seg]  # Merged closures for evaluation

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / SR

    @property
    def eval_boundaries(self) -> np.ndarray:
        """Ground-truth boundary timestamps in seconds, outer silence stripped."""
        return boundary_secs(self.segments_eval)

    @property
    def eval_ipa_sequence(self) -> List[str]:
        """Sequence of IPA labels for recognition evaluation (non-silence)."""
        return [s.ipa_label for s in self.segments_eval if s.ipa_label and s.ipa_label != SILENCE]

    def load_audio(self) -> np.ndarray:
        """Loads the audio waveform as a 1D float32 numpy array normalized to [-1, 1]."""
        audio, sr = sf.read(self.wav_path, dtype="float32")
        if sr != SR:
            raise ValueError(f"Expected sample rate {SR}, got {sr} in {self.wav_path}")
        return audio


def parse_training_segments(phn_path: Path) -> List[PhoneSegment]:
    """Parse .PHN into PhoneSegment objects keeping closures separate for channel estimation.
    
    Identifies:
    - Silence: h#, pau, epi
    - Closures: bcl, dcl, gcl, pcl, tcl, kcl
    - Releases: b, d, g, p, t, k, jh, ch when preceded by matching closure
    """
    raw_rows = read_phn(phn_path)
    segments: List[PhoneSegment] = []

    for i, (start, stop, phn) in enumerate(raw_rows):
        is_silence = phn in TIMIT_SILENCE_PHNS
        is_closure = phn in TIMIT_CLOSURE_PHNS
        
        # Check if this phone is a release preceded by its closure
        is_release = False
        if phn in TIMIT_CLOSURE_OF and i > 0:
            prev_phn = raw_rows[i - 1][2]
            if prev_phn == TIMIT_CLOSURE_OF[phn]:
                is_release = True

        ipa = None if (is_closure or is_silence) else TIMIT_TO_CANONICAL_IPA.get(phn)

        segments.append(
            PhoneSegment(
                start_sample=start,
                stop_sample=stop,
                raw_label=phn,
                ipa_label=ipa,
                is_silence=is_silence,
                is_closure=is_closure,
                is_release=is_release,
            )
        )
    return segments


def parse_eval_segments(phn_path: Path) -> List[Seg]:
    """Parse .PHN into Seg objects with stop closures merged into succeeding releases."""
    raw_rows = read_phn(phn_path)
    segs: List[Seg] = []
    for start, stop, phn in raw_rows:
        ipa = TIMIT_TO_CANONICAL_IPA.get(phn, SILENCE)
        segs.append(Seg(start / SR, stop / SR, phn, ipa))
    
    # Merge closures into following release
    segs = merge_stop_closures(segs)
    # Coalesce consecutive silence
    segs = merge_adjacent_silence(segs)
    return segs


def load_timit_split(
    timit_root: Path,
    split: str = "train",
    include_sa: bool = True,
    max_utterances: Optional[int] = None,
    cache_dir: Optional[Path] = None,
) -> List[TimitUtterance]:
    """Loads all TIMIT utterances for a given split ('train' or 'test').
    
    Args:
        timit_root: Path to TIMIT dataset root containing TRAIN and TEST folders.
        split: 'train' or 'test' (case-insensitive).
        include_sa: Whether to include SA (dialect calibration) sentences.
                    The official 4620 training set includes SA sentences.
        max_utterances: Optional limit on the number of utterances to load.
        cache_dir: Optional directory to cache parsed utterances to disk (.pkl).
    """
    import pickle

    # Check cache if full dataset requested
    if cache_dir is not None and max_utterances is None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"timit_{split.lower()}_sa{int(include_sa)}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                cache_file.unlink(missing_ok=True)

    timit_root = Path(timit_root)
    split_dir = timit_root / split.upper()
    if not split_dir.exists():
        split_dir = timit_root / split.lower()
    if not split_dir.exists():
        raise FileNotFoundError(f"Cannot find split directory {split} in {timit_root}")

    wav_paths = sorted(split_dir.glob("**/*.WAV")) or sorted(split_dir.glob("**/*.wav"))
    if not wav_paths:
        raise FileNotFoundError(f"No WAV files found in {split_dir}")

    utterances: List[TimitUtterance] = []
    for wav_path in wav_paths:
        if max_utterances is not None and len(utterances) >= max_utterances:
            break

        is_sa = wav_path.stem.upper().startswith("SA")
        if not include_sa and is_sa:
            continue

        phn_path = wav_path.with_suffix(".PHN")
        if not phn_path.exists():
            phn_path = wav_path.with_suffix(".phn")
        if not phn_path.exists():
            raise FileNotFoundError(f"Missing .PHN file for {wav_path}")

        train_segs = parse_training_segments(phn_path)
        eval_segs = parse_eval_segments(phn_path)
        num_samples = train_segs[-1].stop_sample if train_segs else 0
        speaker_id = wav_path.parent.name
        dialect_region = wav_path.parent.parent.name
        utt_id = f"{dialect_region}_{speaker_id}_{wav_path.stem}"

        utterances.append(
            TimitUtterance(
                utterance_id=utt_id,
                wav_path=wav_path,
                phn_path=phn_path,
                split=split.lower(),
                speaker_id=speaker_id,
                dialect_region=dialect_region,
                is_sa_sentence=is_sa,
                num_samples=num_samples,
                segments_training=train_segs,
                segments_eval=eval_segs,
            )
        )

    if cache_dir is not None and max_utterances is None:
        tmp_file = cache_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "wb") as f:
                pickle.dump(utterances, f)
            tmp_file.replace(cache_file)
        except Exception:
            tmp_file.unlink(missing_ok=True)

    return utterances
