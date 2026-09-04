"""Full reproduction pipeline for SPAM on TIMIT (Section 15, 17, 21).

Steps:
1. Load TIMIT train and test splits (256 train utterances as ablated in Fig 3, and Core Test Set of 192 utterances).
2. Extract layer 24 WavLM representations with disk caching.
3. Estimate phonological vectors v_i, alpha_i, lambda_i (Equations 1, 3).
4. Fit closed-form backward regressor W_hat (Equation 8).
5. Run test evaluation: 7 signals, multiplicative ensemble b(t), silence suppression,
   peak detection, recognition head matching, R-value, and PFER.
6. Generate and print the reproduction results table comparing with paper.
"""

from __future__ import annotations

import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import platform
import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
import transformers
import panphon

import src
from src.config import default_config
from src.timit_loader import load_timit_split, TimitUtterance
from src.panphon_mapping import PanphonMapping
from src.wavlm_extractor import WavlmExtractor
from src.phonological_vectors import VectorEstimator, PhonologicalVectors
from src.spam_representation import compute_spam
from src.recognition_head import RecognitionHead
from src.segmentation_head import BackwardRegressor, SegmentationHead
from src.evaluation import Evaluator, SegmentationResult, RecognitionResult

# 24 core test speakers defined in TIMIT DOC/TESTSET.DOC
CORE_TEST_SPEAKER_SUFFIXES = frozenset({
    "DAB0", "WBT0", "ELC0",  # DR1
    "TAS1", "WEW0", "PAS0",  # DR2
    "JMP0", "LNT0", "PKT0",  # DR3
    "LLL0", "TLS0", "JLM0",  # DR4
    "BPM0", "KLT0", "NLP0",  # DR5
    "CMJ0", "JDH0", "MGD0",  # DR6
    "GRT0", "NJM0", "DHC0",  # DR7
    "JLN0", "PAM0", "MLD0",  # DR8
})


def log_environment_info(pv: PhonologicalVectors, num_train: int, num_test: int):
    """Logs system and experiment settings as required by Section 17."""
    print("=" * 75)
    print("REPRODUCIBILITY & SYSTEM ENVIRONMENT LOG (Section 17)")
    print("=" * 75)
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"PanPhon: {panphon.__version__ if hasattr(panphon, '__version__') else 'installed'}")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    else:
        print("Compute Device: CPU")
    print(f"WavLM Model Identifier: {default_config.wavlm_model}")
    print(f"WavLM Transformer Layer: {default_config.wavlm_layer} (Final Layer)")
    print(f"TIMIT Training Utterances for Vectors: {num_train}")
    print(f"TIMIT Test Utterances for Evaluation: {num_test}")
    print(f"Active Phonological Channels: {pv.num_channels}")
    print(f"Active Channels List: {pv.channels}")
    print(f"Scaling Constant gamma: {default_config.gamma}")
    print(f"Evaluation Mode: {default_config.eval_mode} (20ms tolerance)")
    print(f"Random Seed: {default_config.random_seed}")
    print("=" * 75)


def run_full_reproduction(
    num_train: int = 256,
    use_core_test: bool = True,
    prominence_sweep: Optional[List[float]] = None,
):
    np.random.seed(default_config.random_seed)
    torch.manual_seed(default_config.random_seed)

    print("\n" + "=" * 75)
    print("EXACT REPRODUCTION: SPAM (arXiv:2607.09020v1) ON TIMIT")
    print("=" * 75)

    # 1. Load TIMIT Dataset
    print("\n[Stage 1/6] Loading TIMIT dataset...")
    t0 = time.time()
    all_train_utts = load_timit_split(
        default_config.timit_root,
        split="train",
        include_sa=True,
        cache_dir=default_config.cache_dir,
    )
    all_test_utts = load_timit_split(
        default_config.timit_root,
        split="test",
        include_sa=True,
        cache_dir=default_config.cache_dir,
    )
    print(f"Discovered {len(all_train_utts)} train utterances and {len(all_test_utts)} test utterances in {time.time() - t0:.2f} s.")

    # Select training subset
    if num_train is not None and num_train < len(all_train_utts):
        rng = np.random.RandomState(default_config.random_seed)
        train_indices = sorted(rng.choice(len(all_train_utts), size=num_train, replace=False))
        train_utts = [all_train_utts[i] for i in train_indices]
        print(f"Selected {len(train_utts)} training utterances (subsampled with seed {default_config.random_seed} matching Figure 3).")
    else:
        train_utts = all_train_utts
        print(f"Using full training set of {len(train_utts)} utterances.")

    # Select test set (Core Test Set: 24 speakers x 8 non-SA sentences = 192 utterances)
    if use_core_test:
        test_utts = [
            u for u in all_test_utts
            if (not u.is_sa_sentence) and (u.speaker_id[1:] in CORE_TEST_SPEAKER_SUFFIXES)
        ]
        print(f"Selected TIMIT Core Test Set: {len(test_utts)} utterances (24 speakers x 8 non-SA sentences).")
    else:
        test_utts = all_test_utts
        print(f"Using complete test set of {len(test_utts)} utterances.")

    # 2. WavLM Extractor
    print("\n[Stage 2/6] Initializing WavLM model...")
    extractor = WavlmExtractor(
        model_name=default_config.wavlm_model,
        layer=default_config.wavlm_layer,
    )

    # 3. Phonological Vectors & Backward Contrast Estimation
    vectors_cache = default_config.output_root / f"phonological_vectors_train{len(train_utts)}.npz"
    regressor_cache = default_config.output_root / f"backward_regressor_train{len(train_utts)}.npy"
    pm = PanphonMapping()

    if vectors_cache.exists() and regressor_cache.exists():
        print("\n[Stage 3/6] Loading cached phonological vectors and backward regressor...")
        data = np.load(vectors_cache)
        pv = PhonologicalVectors(
            channels=list(data["channels"]),
            vectors=data["vectors"],
            alphas=data["alphas"],
            lambdas=data["lambdas"],
            mu_pos=data["mu_pos"],
            mu_comp=data["mu_comp"],
            pos_counts=data["pos_counts"],
            comp_counts=data["comp_counts"],
        )
        regressor = BackwardRegressor()
        regressor.W = np.load(regressor_cache)
        print(f"Loaded {pv.num_channels} vectors and W_hat shape {regressor.W.shape}.")
    else:
        print(f"\n[Stage 3/6] Estimating phonological vectors on {len(train_utts)} train utterances...")
        estimator = VectorEstimator(channels=pm.all_channels, dim=default_config.wavlm_dim)
        
        t_est = time.time()
        total_segments = 0
        train_reps = []
        for idx, utt in enumerate(train_utts):
            wav = utt.load_audio()
            rep = extractor.extract_and_cache(utt.utterance_id, wav, default_config.cache_dir)
            train_reps.append(rep)
            T = rep.shape[0]

            for seg in utt.segments_training:
                c_frame = min(max(seg.center_frame(default_config.wavlm_stride_samples), 0), T - 1)
                active = pm.get_segment_active_channels(
                    ipa_label=seg.ipa_label,
                    is_silence=seg.is_silence,
                    is_closure=seg.is_closure,
                    is_release=seg.is_release,
                )
                estimator.add_segment(rep[c_frame], active)
                total_segments += 1

            if (idx + 1) % 50 == 0 or (idx + 1) == len(train_utts):
                print(f"  Processed {idx + 1}/{len(train_utts)} training utterances ({total_segments} phone segments)...")

        pv = estimator.compute_vectors()
        print(f"Phonological vectors estimated in {time.time() - t_est:.2f} s.")
        print(f"Retained active channels: {pv.num_channels} / {len(pm.all_channels)}")

        # Save phonological vectors to cache
        np.savez(
            vectors_cache,
            channels=np.array(pv.channels),
            vectors=pv.vectors,
            alphas=pv.alphas,
            lambdas=pv.lambdas,
            mu_pos=pv.mu_pos,
            mu_comp=pv.mu_comp,
            pos_counts=pv.pos_counts,
            comp_counts=pv.comp_counts,
        )

        # Fit backward regressor across adjacent segments
        print("Fitting closed-form backward regressor W_hat on adjacent segments (Equation 8)...")
        reg_R_curr = []
        reg_M_prev = []
        for utt, rep in zip(train_utts, train_reps):
            spam_utt = compute_spam(rep, pv, gamma=default_config.gamma)
            T = rep.shape[0]
            segs = utt.segments_training
            for i in range(1, len(segs)):
                c_prev = min(max(segs[i - 1].center_frame(default_config.wavlm_stride_samples), 0), T - 1)
                c_curr = min(max(segs[i].center_frame(default_config.wavlm_stride_samples), 0), T - 1)
                reg_R_curr.append(rep[c_curr])
                reg_M_prev.append(spam_utt[c_prev])

        reg_R_curr = np.stack(reg_R_curr, axis=0)
        reg_M_prev = np.stack(reg_M_prev, axis=0)
        print(f"Adjacent training pairs for W_hat: {reg_R_curr.shape[0]}")
        regressor = BackwardRegressor()
        regressor.fit(reg_R_curr, reg_M_prev)
        np.save(regressor_cache, regressor.W)
        print(f"Backward regressor W_hat saved. Shape: {regressor.W.shape}")

    log_environment_info(pv, len(train_utts), len(test_utts))

    # Update active channels in mapping
    pm.active_channels = pv.channels
    pm.channel_to_idx = {ch: i for i, ch in enumerate(pv.channels)}
    silence_idx = pm.channel_to_idx.get("silence+")

    # 4. Prepare Recognition Head with Canonical PanPhon Inventory
    print("\n[Stage 4/6] Preparing recognition head canonical inventory (Equation 4)...")
    canonical_inv = pm.get_canonical_inventory()
    print(f"Loaded {len(canonical_inv)} canonical phone vectors (cons != 0).")
    rec_head = RecognitionHead(canonical_inv)

    # 5. Pre-extract test representations and compute SPAM + 7 signals
    print(f"\n[Stage 5/6] Extracting representations and computing signals for {len(test_utts)} test utterances...")
    t_test_extract = time.time()
    test_data = []

    for idx, utt in enumerate(test_utts):
        wav = utt.load_audio()
        rep = extractor.extract_and_cache(utt.utterance_id, wav, default_config.cache_dir)
        spam = compute_spam(rep, pv, gamma=default_config.gamma)
        test_data.append((utt, wav, rep, spam))

        if (idx + 1) % 50 == 0 or (idx + 1) == len(test_utts):
            print(f"  Processed {idx + 1}/{len(test_utts)} test utterances...")

    print(f"Test representations and SPAM computed in {time.time() - t_test_extract:.2f} s.")

    # 6. Evaluation
    print("\n[Stage 6/6] Running segmentation and recognition evaluation...")
    evaluator = Evaluator(
        tolerance=default_config.tolerance_seconds,
        mode=default_config.eval_mode,
    )

    # Note: Prominence threshold is UNSPECIFIED IN PAPER.
    # We do NOT calibrate or tune it against reference ground-truth boundaries.
    # We evaluate using the fixed default prominence threshold.
    fixed_prom = default_config.prominence_threshold
    print(f"Evaluating with fixed default prominence = {fixed_prom} (UNSPECIFIED IN PAPER; not tuned on GT)")

    seg_head = SegmentationHead(
        backward_regressor=regressor,
        silence_channel_idx=silence_idx,
        prominence=fixed_prom,
        min_distance=default_config.min_peak_distance_frames,
        silence_threshold=default_config.silence_threshold,
    )

    all_ref_boundaries = []
    all_pred_boundaries = []
    all_ref_phones = []
    all_pred_phones = []

    for utt, wav, rep, spam in test_data:
        b, _ = seg_head.compute_ensemble_signal(wav, rep, spam)
        peaks, pred_times = seg_head.detect_boundaries(b)
        pred_phones = rec_head.predict_utterance(spam, peaks.tolist())

        all_ref_boundaries.append(utt.eval_boundaries)
        all_pred_boundaries.append(pred_times)
        all_ref_phones.append(utt.eval_ipa_sequence)
        all_pred_phones.append(pred_phones)

    best_seg_res = evaluator.evaluate_segmentation(all_ref_boundaries, all_pred_boundaries)
    best_rec_res = evaluator.evaluate_recognition(all_ref_phones, all_pred_phones)
    best_prom = fixed_prom

    print("\n" + "=" * 75)
    print(f"REPRODUCTION RESULTS ON TIMIT (Prominence = {best_prom})")
    print("=" * 75)
    print(best_seg_res)
    print("-" * 75)
    print(best_rec_res)
    print("=" * 75)

    # Final Comparison Table (Section 21)
    print("\n" + "=" * 85)
    print("COMPARISON WITH PAPER (arXiv:2607.09020v1, Table I & Table II)")
    print("=" * 85)
    header = f"{'Dataset':<8} | {'Training Split':<20} | {'Model':<14} | {'Layer':<6} | {'Segmentation R-val':<20} | {'Recognition PFER':<16}"
    print(header)
    print("-" * len(header))
    paper_row = f"{'TIMIT':<8} | {'TIMIT (3h)':<20} | {'WavLM-large':<14} | {'24':<6} | {'80.0%':<20} | {'22.9%':<16}"
    repro_row = f"{'TIMIT':<8} | {f'TIMIT ({len(train_utts)} utts)':<20} | {'WavLM-large':<14} | {'24':<6} | {f'{best_seg_res.r_value * 100:.2f}%':<20} | {f'{best_rec_res.pfer * 100:.2f}%':<16}"
    print(f"[Paper]      {paper_row}")
    print(f"[Reproduced] {repro_row}")
    print("=" * 85)

    # Save summary report
    summary_path = default_config.output_root / "reproduction_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SPAM TIMIT REPRODUCTION SUMMARY\n")
        f.write(f"Train utterances: {len(train_utts)}\n")
        f.write(f"Test utterances: {len(test_utts)}\n")
        f.write(f"Best Prominence: {best_prom}\n")
        f.write(f"Segmentation R-value: {best_seg_res.r_value * 100:.2f}%\n")
        f.write(f"Precision: {best_seg_res.precision * 100:.2f}%\n")
        f.write(f"Recall: {best_seg_res.recall * 100:.2f}%\n")
        f.write(f"F1: {best_seg_res.f1 * 100:.2f}%\n")
        f.write(f"Recognition PFER: {best_rec_res.pfer * 100:.2f}%\n")
        f.write(f"Recognition PER: {best_rec_res.per * 100:.2f}%\n")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    run_full_reproduction()
