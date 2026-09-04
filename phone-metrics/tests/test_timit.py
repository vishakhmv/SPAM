"""Tests for TIMIT phone reading and closure-merge logic.

Moved here from the phonological-posteriogram repo's test_smoke.py when the
closure-merge logic was extracted into phone_metrics. Adapted from the
dict-based prepare_datasets rows to phone_metrics' Seg dataclass
(``raw_label``/``ipa_label`` instead of ``timit_phn``/``ipa``).
"""

from phone_metrics.timit import Seg, merge_adjacent_silence, merge_stop_closures


def _seg(timit_phn, mn, mx, ipa):
    """A single segment, mirroring a merged-mode TIMIT interval."""
    return Seg(mn, mx, timit_phn, ipa)


def test_merge_stop_closures_merges_closure_into_release():
    """A closure immediately before its release is folded into the release:
    one segment spanning [closure.start, release.end], labeled as the release."""
    segs = [
        _seg("bcl", 0.00, 0.05, "b"),
        _seg("b", 0.05, 0.10, "b"),
        _seg("iy", 0.10, 0.20, "i"),
    ]
    out = merge_stop_closures(segs)
    assert [s.raw_label for s in out] == ["b", "iy"]
    # The merged release spans the closure's start through its own end.
    assert out[0].start == 0.00 and out[0].end == 0.10
    assert out[0].ipa_label == "b"


def test_merge_stop_closures_standalone_closure_falls_back_to_stop():
    """A closure with no following release is kept and surfaces as the bare
    stop, NOT dropped and NOT a distinct closure symbol."""
    # bcl followed by a vowel (not its release /b/): stays on its own.
    segs = [
        _seg("bcl", 0.00, 0.05, "b"),
        _seg("iy", 0.05, 0.15, "i"),
    ]
    out = merge_stop_closures(segs)
    assert [s.raw_label for s in out] == ["bcl", "iy"]
    # The lone closure keeps its own span and reads out as the stop "b".
    assert out[0].start == 0.00 and out[0].end == 0.05
    assert out[0].ipa_label == "b"


def test_merge_stop_closures_affricate_after_cluster_closure():
    """Affricate closures merge (dcl→jh, tcl→ch), but only the *canonical*
    closure is folded: a preceding non-matching cluster closure stays
    standalone. Uses ``tcl dcl jh`` — an attested TIMIT cluster (a stop
    closure ``tcl`` followed by the affricate's own ``dcl`` closure and ``jh``);
    note ``dcl dcl`` never occurs in TIMIT."""
    segs = [
        _seg("tcl", 0.00, 0.04, "t"),  # standalone: not jh's canonical closure
        _seg("dcl", 0.04, 0.08, "d"),  # closure of the following jh
        _seg("jh", 0.08, 0.15, "d͡ʒ"),
    ]
    out = merge_stop_closures(segs)
    assert [s.raw_label for s in out] == ["tcl", "jh"]
    # tcl stays standalone → "t"; dcl merged into jh.
    assert out[0].start == 0.00 and out[0].end == 0.04 and out[0].ipa_label == "t"
    assert out[1].start == 0.04 and out[1].end == 0.15 and out[1].ipa_label == "d͡ʒ"


def test_merge_stop_closures_release_without_closure_unchanged():
    """A release with no preceding closure is left untouched."""
    segs = [
        _seg("iy", 0.00, 0.10, "i"),
        _seg("b", 0.10, 0.15, "b"),  # release, but prev is a vowel
    ]
    out = merge_stop_closures(segs)
    assert [s.raw_label for s in out] == ["iy", "b"]
    assert out[1].start == 0.10  # not extended


def test_merge_adjacent_silence_coalesces_only_consecutive_silence():
    segs = [
        _seg("h#", 0.00, 0.05, "_"),
        _seg("pau", 0.05, 0.10, "_"),
        _seg("iy", 0.10, 0.20, "i"),
        _seg("epi", 0.20, 0.25, "_"),
        _seg("h#", 0.25, 0.30, "_"),
    ]
    out = merge_adjacent_silence(segs)

    assert [(s.start, s.end, s.ipa_label) for s in out] == [
        (0.00, 0.10, "_"),
        (0.10, 0.20, "i"),
        (0.20, 0.30, "_"),
    ]
