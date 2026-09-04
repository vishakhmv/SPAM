"""Evaluation module for phone segmentation (R-value) and phone recognition (PFER, PER).

Implements:
- Strict mode 20ms tolerance boundary evaluation (Precision, Recall, F1, Over-segmentation, R-value).
- Phone Feature Edit Rate (PFER) using PanPhon feature distance and Phone Error Rate (PER).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Sequence
import numpy as np
import panphon.distance
from phone_metrics.segmentation import PrecisionRecallMetric
from phone_metrics.recognition import phone_error_rates, _levenshtein, _expand_phones
from phone_metrics.timit import SILENCE


@dataclass
class SegmentationResult:
    """Detailed boundary segmentation metrics."""
    r_value: float
    precision: float
    recall: float
    f1: float
    over_segmentation: float
    num_reference_boundaries: int
    num_predicted_boundaries: int
    hits: int
    false_alarms: int
    misses: int

    def __str__(self) -> str:
        return (
            f"R-value: {self.r_value * 100:.2f}%\n"
            f"Precision: {self.precision * 100:.2f}%\n"
            f"Recall: {self.recall * 100:.2f}%\n"
            f"F1: {self.f1 * 100:.2f}%\n"
            f"Over-segmentation: {self.over_segmentation:.4f}\n"
            f"Hits: {self.hits} | False Alarms: {self.false_alarms} | Misses: {self.misses}\n"
            f"Reference boundaries: {self.num_reference_boundaries} | Predicted: {self.num_predicted_boundaries}"
        )


@dataclass
class RecognitionResult:
    """Detailed phone recognition metrics."""
    pfer: float
    per: float
    phone_total: int
    pfer_cost: float
    per_edits: int

    def __str__(self) -> str:
        return (
            f"PFER: {self.pfer * 100:.2f}%\n"
            f"PER: {self.per * 100:.2f}%\n"
            f"Total reference phones: {self.phone_total}\n"
            f"Feature distance cost: {self.pfer_cost:.2f} | Levenshtein edits: {self.per_edits}"
        )


class Evaluator:
    """Evaluates segmentation and recognition predictions on TIMIT."""

    def __init__(self, tolerance: float = 0.02, mode: str = "strict"):
        self.tolerance = tolerance
        self.mode = mode
        self.metric = PrecisionRecallMetric(tolerance=tolerance, mode=mode)
        self.panphon_dist = panphon.distance.Distance()

    def evaluate_segmentation(
        self,
        ref_boundaries_list: List[np.ndarray],
        pred_boundaries_list: List[np.ndarray],
    ) -> SegmentationResult:
        """Evaluates predicted boundaries against ground truth.
        
        Args:
            ref_boundaries_list: List of 1D float arrays with ground-truth boundary timestamps (seconds).
            pred_boundaries_list: List of 1D float arrays with predicted boundary timestamps (seconds).
        """
        metric = PrecisionRecallMetric(tolerance=self.tolerance, mode=self.mode)
        for ref, pred in zip(ref_boundaries_list, pred_boundaries_list):
            metric.update(ref, pred)

        res = metric.compute()
        n_gts = sum(len(r) for r in ref_boundaries_list)
        n_preds = sum(len(p) for p in pred_boundaries_list)
        
        # Calculate hits, misses, false alarms from precision & recall
        p_count = int(round(res["precision"] * n_preds))
        r_count = int(round(res["recall"] * n_gts))
        hits = r_count
        misses = n_gts - hits
        false_alarms = n_preds - p_count

        return SegmentationResult(
            r_value=float(res["rval"]),
            precision=float(res["precision"]),
            recall=float(res["recall"]),
            f1=float(res["f1"]),
            over_segmentation=float(res["over_seg"]),
            num_reference_boundaries=n_gts,
            num_predicted_boundaries=n_preds,
            hits=hits,
            false_alarms=false_alarms,
            misses=misses,
        )

    def evaluate_recognition(
        self,
        ref_phones_list: List[List[str]],
        pred_phones_list: List[List[str]],
    ) -> RecognitionResult:
        """Evaluates predicted phone sequences against ground-truth IPA sequences.
        
        Args:
            ref_phones_list: List of ground-truth IPA token lists.
            pred_phones_list: List of predicted IPA token lists.
        """
        total_pfer_cost = 0.0
        total_per_edits = 0
        total_phones = 0

        for ref, pred in zip(ref_phones_list, pred_phones_list):
            # Expand compound phones (diphthongs -> component vowels) per phone_metrics benchmark convention
            ref_expanded = _expand_phones(ref)
            pred_expanded = _expand_phones(pred)

            # Ignore silence in PFER and PER
            ref_ns = [tok for tok in ref_expanded if tok != SILENCE and tok is not None]
            pred_ns = [tok for tok in pred_expanded if tok != SILENCE and tok is not None]
            if not ref_ns:
                continue

            # Feature edit distance via PanPhon
            cost = float(self.panphon_dist.feature_edit_distance("".join(pred_ns), "".join(ref_ns)))
            edits = _levenshtein(pred_ns, ref_ns)

            total_pfer_cost += cost
            total_per_edits += edits
            total_phones += len(ref_ns)

        pfer = total_pfer_cost / total_phones if total_phones > 0 else 0.0
        per = total_per_edits / total_phones if total_phones > 0 else 0.0

        return RecognitionResult(
            pfer=pfer,
            per=per,
            phone_total=total_phones,
            pfer_cost=total_pfer_cost,
            per_edits=total_per_edits,
        )
