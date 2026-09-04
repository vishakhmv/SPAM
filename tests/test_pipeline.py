"""Unit tests for all 16 pipeline components (Section 19).

Uses synthetic vectors so each equation and behavior can be manually and deterministically verified:
1. PanPhon ternary -> binary channel conversion
2. center-frame extraction
3. difference-of-means phonological vectors (Equation 1)
4. alpha and lambda calculation (Equation 3)
5. SPAM projection (Equation 2)
6. sigmoid canonical-vector matching (Equation 4)
7. delta1 adjacent-frame difference (Equation 5)
8. delta2 multi-scale difference (Equation 6)
9. delta3 multi-scale difference (Equation 7)
10. least-squares backward regressor (Equation 8)
11. beta1/beta2/beta3 backward contrasts (Equation 9)
12. mel delta signal (Equation 10)
13. seven-signal multiplicative ensemble (Equation 11)
14. silence suppression
15. closure/release collapse
16. boundary evaluation (strict mode, 20ms tolerance, R-value)
"""

from __future__ import annotations

import pytest
import numpy as np
import torch

from src.panphon_mapping import PanphonMapping
from src.timit_loader import PhoneSegment, parse_eval_segments
from src.phonological_vectors import VectorEstimator, PhonologicalVectors
from src.spam_representation import compute_spam
from src.recognition_head import RecognitionHead
from src.segmentation_head import (
    BackwardRegressor,
    SegmentationHead,
    cosine_distance_frames,
    cosine_similarity_frames,
)
from src.evaluation import Evaluator
from phone_metrics.timit import Seg, merge_stop_closures


# 1. PanPhon ternary -> binary channel conversion
def test_panphon_ternary_to_binary():
    mapping = PanphonMapping()
    # 'p' is voiceless: voi is -1
    fts_p = mapping.phone_to_ternary("p")
    assert fts_p is not None
    assert fts_p["voi"] == -1
    active_p = mapping.get_segment_active_channels("p", is_silence=False, is_closure=False, is_release=False)
    assert "voi-" in active_p
    assert "voi+" not in active_p
    assert "closure-" in active_p
    assert "release-" in active_p

    # 'b' is voiced: voi is 1
    fts_b = mapping.phone_to_ternary("b")
    assert fts_b is not None
    assert fts_b["voi"] == 1
    active_b = mapping.get_segment_active_channels("b", is_silence=False, is_closure=False, is_release=False)
    assert "voi+" in active_b
    assert "voi-" not in active_b


# 2. center-frame extraction
def test_center_frame_extraction():
    # Stride = 320 samples (20 ms at 16 kHz)
    seg = PhoneSegment(
        start_sample=3200,
        stop_sample=6400,
        raw_label="aa",
        ipa_label="ɑ",
        is_silence=False,
        is_closure=False,
        is_release=False,
    )
    assert seg.center_sample == 4800
    assert seg.center_frame(320) == 15
    assert np.isclose(seg.center_time, 4800 / 16000)


# 3. difference-of-means phonological vectors (Eq. 1)
# 4. alpha and lambda calculation (Eq. 3)
def test_phonological_vectors_and_alpha_lambda():
    estimator = VectorEstimator(["ch1", "ch2"], dim=2)
    # Positive for ch1: [4, 6], [6, 8] -> mean = [5, 7]
    estimator.add_segment(np.array([4.0, 6.0]), {"ch1"})
    estimator.add_segment(np.array([6.0, 8.0]), {"ch1"})
    # Complement for ch1: [1, 2], [3, 4] -> mean = [2, 3]
    estimator.add_segment(np.array([1.0, 2.0]), {"ch2"})
    estimator.add_segment(np.array([3.0, 4.0]), {"ch2"})

    pv = estimator.compute_vectors()
    # v = [5, 7] - [2, 3] = [3, 4]
    v_expected = np.array([3.0, 4.0], dtype=np.float32)
    assert np.allclose(pv.vectors[0], v_expected)

    # lambda = ||v||^2 = 3^2 + 4^2 = 25
    assert np.isclose(pv.lambdas[0], 25.0)

    # alpha = 0.5 * (mu_pos^T v + mu_comp^T v)
    # mu_pos^T v = 5*3 + 7*4 = 15 + 28 = 43
    # mu_comp^T v = 2*3 + 3*4 = 6 + 12 = 18
    # alpha = 0.5 * (43 + 18) = 30.5
    assert np.isclose(pv.alphas[0], 30.5)


# 5. SPAM projection (Eq. 2)
def test_spam_projection():
    # For a vector with mu_pos = [5, 7], mu_comp = [2, 3], v = [3, 4], alpha = 30.5, lambda = 25
    pv = PhonologicalVectors(
        channels=["ch1"],
        vectors=np.array([[3.0, 4.0]], dtype=np.float32),
        alphas=np.array([30.5], dtype=np.float32),
        lambdas=np.array([25.0], dtype=np.float32),
        mu_pos=np.array([[5.0, 7.0]], dtype=np.float32),
        mu_comp=np.array([[2.0, 3.0]], dtype=np.float32),
        pos_counts=np.array([2]),
        comp_counts=np.array([2]),
    )
    # Test at mu_pos: dot = 43 -> (43 - 30.5) * 4 / 25 = 12.5 * 4 / 25 = 2.0
    # Test at mu_comp: dot = 18 -> (18 - 30.5) * 4 / 25 = -12.5 * 4 / 25 = -2.0
    R = np.array([[5.0, 7.0], [2.0, 3.0]], dtype=np.float32)
    spam = compute_spam(R, pv, gamma=4.0)
    assert np.allclose(spam, [[2.0], [-2.0]])


# 6. sigmoid canonical-vector matching (Eq. 4)
def test_recognition_canonical_matching():
    canonical_inv = {
        "p": np.array([0.7, 0.3]),
        "b": np.array([0.2, 0.8]),
    }
    rec = RecognitionHead(canonical_inv)
    # Strongly activates channel 0 -> should match 'p'
    m_p = np.array([10.0, -10.0])
    pred, score = rec.predict_frame(m_p)
    assert pred == "p"
    # Strongly activates channel 1 -> should match 'b'
    m_b = np.array([-10.0, 10.0])
    pred, score = rec.predict_frame(m_b)
    assert pred == "b"


# 7. delta1 adjacent-frame difference (Eq. 5)
# 8. delta2 multi-scale difference (Eq. 6)
# 9. delta3 multi-scale difference (Eq. 7)
def test_multi_scale_differences():
    head = SegmentationHead()
    # 4 frames: frames 0 and 1 are [1, 0], frames 2 and 3 are [0, 1]
    spam = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    d1, d2, d3 = head.compute_multi_scale_differences(spam)
    # Boundary occurs between frame 1 and frame 2
    # At t=2: d1(2) = 1 - cos(m_1, m_2) = 1 - cos([1,0], [0,1]) = 1 - 0 = 1.0
    assert np.isclose(d1[2], 1.0)
    # At t=1: d1(1) = 1 - cos(m_0, m_1) = 1 - cos([1,0], [1,0]) = 0.0
    assert np.isclose(d1[1], 0.0)


# 10. least-squares backward regressor (Eq. 8)
# 11. beta1/beta2/beta3 backward contrasts (Eq. 9)
def test_backward_regressor_and_contrasts():
    reg = BackwardRegressor()
    # Pair: r_curr [2, 0] preceded by m_prev [1, 1]
    R_curr = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    M_prev = np.array([[1.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    reg.fit(R_curr, M_prev)
    pred = reg.predict(np.array([[2.0, 0.0]]))
    assert np.allclose(pred, [[1.0, 1.0]])

    head = SegmentationHead(backward_regressor=reg)
    R = np.random.randn(10, 2).astype(np.float32)
    spam = np.random.randn(10, 2).astype(np.float32)
    b1, b2, b3 = head.compute_backward_contrasts(R, spam)
    assert b1.shape == (10,)
    assert b2.shape == (10,)
    assert b3.shape == (10,)
    # Check bounds: beta values cannot exceed [-2, 2]
    assert np.all(b1 >= -2.0001) and np.all(b1 <= 2.0001)


# 12. mel delta signal (Eq. 10)
def test_mel_delta_signal():
    head = SegmentationHead()
    wav = np.sin(np.linspace(0, 100, 3200)).astype(np.float32)
    mel_sig = head.compute_mel_signal(wav, target_length=10)
    assert mel_sig.shape == (10,)
    assert np.all(mel_sig >= 0.0) and np.all(mel_sig <= 1.0)


# 13. seven-signal multiplicative ensemble (Eq. 11)
# 14. silence suppression
def test_ensemble_and_silence_suppression():
    reg = BackwardRegressor()
    reg.fit(np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32))
    head = SegmentationHead(backward_regressor=reg, silence_channel_idx=0, silence_threshold=0.5)

    R = np.ones((8, 4), dtype=np.float32)
    spam = np.ones((8, 4), dtype=np.float32)
    # Frame 4 has strong silence
    spam[4, 0] = 20.0
    # Other frames non-silent
    spam[:4, 0] = -20.0
    spam[5:, 0] = -20.0

    wav = np.zeros(8 * 320, dtype=np.float32)
    b, sigs = head.compute_ensemble_signal(wav, R, spam)
    assert b.shape == (8,)
    # Silent frame 4 must be suppressed to 0.0
    assert b[4] == 0.0


# 15. closure/release collapse
def test_closure_release_collapse():
    # Sequence: closure 'tcl' followed by release 't'
    segs = [
        Seg(start=0.0, end=0.05, raw_label="tcl", ipa_label="t"),
        Seg(start=0.05, end=0.10, raw_label="t", ipa_label="t"),
        Seg(start=0.10, end=0.20, raw_label="aa", ipa_label="ɑ"),
    ]
    merged = merge_stop_closures(segs)
    assert len(merged) == 2
    assert merged[0].raw_label == "t"
    assert merged[0].start == 0.0
    assert merged[0].end == 0.10
    assert merged[1].raw_label == "aa"


# 16. boundary evaluation (strict mode, 20ms tolerance, R-value)
def test_boundary_evaluation():
    evaluator = Evaluator(tolerance=0.02, mode="strict")
    # Reference boundaries at 0.1, 0.2, 0.3
    ref = [np.array([0.100, 0.200, 0.300])]
    # Predictions within 20 ms: 0.105 (hit), 0.198 (hit), 0.315 (hit)
    pred_hits = [np.array([0.105, 0.198, 0.315])]
    res_hits = evaluator.evaluate_segmentation(ref, pred_hits)
    assert np.isclose(res_hits.r_value, 1.0)
    assert res_hits.hits == 3

    # Predictions outside 20 ms tolerance: 0.150, 0.250, 0.350
    pred_misses = [np.array([0.150, 0.250, 0.350])]
    res_misses = evaluator.evaluate_segmentation(ref, pred_misses)
    assert res_misses.hits == 0
    assert res_misses.r_value < 1.0
