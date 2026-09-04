import pytest

from phone_metrics.datasets import Utterance
from phone_metrics.oracle import oracle_phone_accuracy
from phone_metrics.timit import Seg


def _utt(audio_path, language, labels):
    return Utterance(
        audio_path=audio_path,
        language=language,
        split="test",
        segments=[
            Seg(float(i), float(i + 1), raw_label=label, ipa_label=label)
            for i, label in enumerate(labels)
        ],
    )


def test_oracle_accuracy_strips_edges_but_scores_internal_silence():
    utt = _utt("u1.wav", "eng", ["_", "p", "_", "t", "_"])

    result = oracle_phone_accuracy(
        [utt],
        ["_", "p", "t", "t", "_"],
        label="ipa",
    )

    assert result.correct == 2
    assert result.total == 3
    assert result.accuracy == pytest.approx(2 / 3)


def test_oracle_accuracy_drops_gt_edge_silence_positionally():
    # Predictions are 1:1 with GT segments, and the recognizer may emit a
    # non-silence label at a GT edge-silence segment. That segment is dropped
    # positionally (with its aligned prediction), not by the prediction's value,
    # so it never counts and the remaining predictions stay aligned.
    utt = _utt("u1.wav", "eng", ["_", "p", "t"])

    result = oracle_phone_accuracy(
        [utt],
        ["a", "p", "t"],
        label="ipa",
    )

    assert result.correct == 2
    assert result.total == 2
    assert result.accuracy == pytest.approx(1.0)
