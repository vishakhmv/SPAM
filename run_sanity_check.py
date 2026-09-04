"""Sanity-check pipeline on a small subset of TIMIT utterances (Section 18).

Verifies tensor shapes, checks for NaNs/inf, verifies active channels,
runs end-to-end segmentation and recognition on a sample utterance,
and produces a debugging visualization.
"""

from __future__ import annotations

import os
import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import default_config
from src.timit_loader import load_timit_split, TimitUtterance
from src.panphon_mapping import PanphonMapping
from src.wavlm_extractor import WavlmExtractor
from src.phonological_vectors import VectorEstimator
from src.spam_representation import compute_spam
from src.recognition_head import RecognitionHead
from src.segmentation_head import BackwardRegressor, SegmentationHead
from src.evaluation import Evaluator


def run_sanity_check(num_train: int = 10, num_test: int = 2):
    print("=" * 60)
    print("STEP 14: SANITY CHECK PIPELINE")
    print("=" * 60)

    # 1. Load small subset of utterances
    print(f"\n1. Loading {num_train} train utterances and {num_test} test utterances...")
    train_utts = load_timit_split(default_config.timit_root, split="train", max_utterances=num_train)
    test_utts = load_timit_split(default_config.timit_root, split="test", max_utterances=num_test)
    print(f"Loaded {len(train_utts)} train, {len(test_utts)} test utterances.")

    # 2. Extract WavLM representations
    print(f"\n2. Initializing WavLM extractor ({default_config.wavlm_model}, layer {default_config.wavlm_layer})...")
    extractor = WavlmExtractor(
        model_name=default_config.wavlm_model,
        layer=default_config.wavlm_layer,
    )

    train_reps = []
    for i, utt in enumerate(train_utts):
        wav = utt.load_audio()
        rep = extractor.extract(wav)
        train_reps.append(rep)
        if i == 0:
            print(f"Sample utterance '{utt.utterance_id}': audio samples = {len(wav)}, WavLM shape = {rep.shape} [T, D]")
            assert rep.ndim == 2 and rep.shape[1] == default_config.wavlm_dim
            assert np.all(np.isfinite(rep)), "WavLM representations contain non-finite values!"

    test_wav = test_utts[0].load_audio()
    test_rep = extractor.extract(test_wav)
    print(f"Test utterance '{test_utts[0].utterance_id}': audio samples = {len(test_wav)}, WavLM shape = {test_rep.shape} [T, D]")

    # 3. PanPhon mapping and channel setup
    print("\n3. Setting up PanPhon channels...")
    pm = PanphonMapping()
    print(f"Total potential channels: {len(pm.all_channels)}")

    # 4. Phonological vector estimation
    print("\n4. Accumulating center-pooled frames for difference-of-means vectors...")
    estimator = VectorEstimator(channels=pm.all_channels, dim=default_config.wavlm_dim)

    # Collect adjacent segment pairs for backward regressor
    pair_R_curr = []
    pair_M_prev_indices = []  # We will construct m_{c(l)} after estimating SPAM

    for utt, rep in zip(train_utts, train_reps):
        T = rep.shape[0]
        for seg in utt.segments_training:
            c_frame = min(max(seg.center_frame(default_config.wavlm_stride_samples), 0), T - 1)
            active = pm.get_segment_active_channels(
                ipa_label=seg.ipa_label,
                is_silence=seg.is_silence,
                is_closure=seg.is_closure,
                is_release=seg.is_release,
            )
            r_c = rep[c_frame]
            estimator.add_segment(r_c, active)

    pv = estimator.compute_vectors()
    print(f"Phonological vectors shape: {pv.vectors.shape} [C, D]")
    print(f"Alphas shape: {pv.alphas.shape} [C]")
    print(f"Lambdas shape: {pv.lambdas.shape} [C]")
    print(f"Active channels retained: {len(pv.channels)} / {len(pm.all_channels)}")
    assert np.all(np.isfinite(pv.vectors)), "Phonological vectors contain NaN or Inf!"
    assert np.all(np.isfinite(pv.alphas)), "Alphas contain NaN or Inf!"
    assert np.all(np.isfinite(pv.lambdas)), "Lambdas contain NaN or Inf!"
    assert np.all(pv.lambdas > 0), "Lambdas must be strictly positive!"

    # Update active channels in mapping
    pm.active_channels = pv.channels
    pm.channel_to_idx = {ch: i for i, ch in enumerate(pv.channels)}

    # 5. Compute SPAM on test utterance
    print("\n5. Computing SPAM representation for test utterance (Equation 2)...")
    spam_test = compute_spam(test_rep, pv, gamma=default_config.gamma)
    print(f"SPAM matrix shape: {spam_test.shape} [T, C]")
    assert spam_test.shape == (test_rep.shape[0], pv.num_channels)
    assert np.all(np.isfinite(spam_test)), "SPAM values contain NaN or Inf!"

    # 6. Fit Backward Regressor (Equation 8) on adjacent training pairs
    print("\n6. Fitting closed-form backward regressor W_hat (Equation 8)...")
    reg_R_curr = []
    reg_M_prev = []

    for utt, rep in zip(train_utts, train_reps):
        spam_utt = compute_spam(rep, pv, gamma=default_config.gamma)
        T = rep.shape[0]
        segs = utt.segments_training
        for i in range(1, len(segs)):
            prev_seg = segs[i - 1]
            curr_seg = segs[i]
            c_prev = min(max(prev_seg.center_frame(default_config.wavlm_stride_samples), 0), T - 1)
            c_curr = min(max(curr_seg.center_frame(default_config.wavlm_stride_samples), 0), T - 1)
            reg_R_curr.append(rep[c_curr])
            reg_M_prev.append(spam_utt[c_prev])

    reg_R_curr = np.stack(reg_R_curr, axis=0)
    reg_M_prev = np.stack(reg_M_prev, axis=0)
    print(f"Training pairs for backward regressor: {reg_R_curr.shape[0]}")

    regressor = BackwardRegressor()
    regressor.fit(reg_R_curr, reg_M_prev)
    print(f"Backward regressor W_hat shape: {regressor.W.shape} [D, C]")
    assert np.all(np.isfinite(regressor.W)), "W_hat contains NaN or Inf!"

    # 7. Segmentation Head (Equations 5-11)
    print("\n7. Computing seven segmentation signals and ensemble b(t)...")
    silence_idx = pm.channel_to_idx.get("silence+")
    seg_head = SegmentationHead(
        backward_regressor=regressor,
        silence_channel_idx=silence_idx,
        prominence=default_config.prominence_threshold,
        min_distance=default_config.min_peak_distance_frames,
        silence_threshold=default_config.silence_threshold,
    )

    b, sigs = seg_head.compute_ensemble_signal(test_wav, test_rep, spam_test)
    print(f"Ensemble signal b(t) shape: {b.shape} [T]")
    assert b.shape == (test_rep.shape[0],), "b(t) length does not match S3M frames!"
    assert np.all(np.isfinite(b)), "b(t) contains NaN or Inf!"

    peaks, pred_boundary_times = seg_head.detect_boundaries(b)
    print(f"Detected {len(peaks)} boundary peaks at frames: {peaks.tolist()[:10]}...")
    print(f"Predicted boundary timestamps (s): {pred_boundary_times.tolist()[:10]}...")

    # 8. Recognition Head (Equation 4)
    print("\n8. Evaluating recognition head with canonical PanPhon vectors (Equation 4)...")
    canonical_inv = pm.get_canonical_inventory()
    print(f"Canonical inventory entries: {len(canonical_inv)}")
    rec_head = RecognitionHead(canonical_inv)

    pred_phones = rec_head.predict_utterance(
        spam_test,
        peaks.tolist(),
        silence_channel_idx=silence_idx,
        silence_threshold=default_config.silence_threshold,
    )
    print(f"Delimited segments: {len(pred_phones)}")
    print(f"Sample predicted phones: {pred_phones[:10]}")

    # 9. Evaluation
    print("\n9. Running evaluation on test utterance...")
    evaluator = Evaluator(
        tolerance=default_config.tolerance_seconds,
        mode=default_config.eval_mode,
    )

    ref_boundaries = [test_utts[0].eval_boundaries]
    pred_boundaries = [pred_boundary_times]
    seg_res = evaluator.evaluate_segmentation(ref_boundaries, pred_boundaries)
    print("\n--- SEGMENTATION RESULT (1 test utterance) ---")
    print(seg_res)

    ref_phones = [test_utts[0].eval_ipa_sequence]
    rec_res = evaluator.evaluate_recognition(ref_phones, [pred_phones])
    print("\n--- RECOGNITION RESULT (1 test utterance) ---")
    print(rec_res)

    # 10. Generate visualization plot
    print("\n10. Generating visualization plot for debugging (Section 18)...")
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    T = test_rep.shape[0]
    time_axis = np.arange(T) * 0.02
    audio_time = np.arange(len(test_wav)) / default_config.sample_rate

    # Subplot 1: Waveform
    axes[0].plot(audio_time, test_wav, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title(f"TIMIT Utterance: {test_utts[0].utterance_id}")
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: SPAM Heatmap
    im = axes[1].imshow(
        spam_test.T,
        aspect="auto",
        origin="lower",
        extent=[0, time_axis[-1], 0, pv.num_channels],
        cmap="coolwarm",
        vmin=-4,
        vmax=4,
    )
    axes[1].set_ylabel("Channels")
    axes[1].set_title("SPAM Activations M ∈ R^(T × |C|)")
    fig.colorbar(im, ax=axes[1], orientation="vertical", pad=0.01)

    # Subplot 3: Component Signals (delta1, beta1, delta_mel)
    axes[2].plot(time_axis, sigs["delta1"], label="δ1 (Adjacent SPAM diff)", color="orange", alpha=0.8)
    axes[2].plot(time_axis, sigs["beta1"], label="β1 (Backward contrast)", color="purple", alpha=0.8)
    axes[2].plot(time_axis, sigs["delta_mel"], label="δ_mel (Mel diff)", color="teal", alpha=0.8)
    axes[2].set_ylabel("Signal")
    axes[2].set_title("Component Segmentation Signals")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)

    # Subplot 4: Ensemble Signal b(t) with Boundaries
    axes[3].plot(time_axis, b, label="Ensemble b(t)", color="crimson", linewidth=1.5)
    
    # Plot reference boundaries in green solid lines
    for r_b in ref_boundaries[0]:
        axes[3].axvline(r_b, color="forestgreen", linestyle="-", alpha=0.7, label="Reference" if r_b == ref_boundaries[0][0] else "")
    # Plot predicted boundaries in red dashed lines
    for p_b in pred_boundary_times:
        axes[3].axvline(p_b, color="red", linestyle="--", alpha=0.7, label="Predicted" if p_b == pred_boundary_times[0] else "")

    axes[3].set_xlabel("Time (seconds)")
    axes[3].set_ylabel("Ensemble b(t)")
    axes[3].set_title("Ensemble b(t) with Ground Truth (green) vs Predicted (red dashed) Boundaries")
    axes[3].legend(loc="upper right")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = default_config.output_root / "sanity_check_visualization.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Visualization saved to: {plot_path}")
    print("\nSanity check completed successfully!")


if __name__ == "__main__":
    run_sanity_check()
