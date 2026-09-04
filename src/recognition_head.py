"""Training-free recognition head based on PanPhon canonical vector matching (Section III-C, Eq. 4).

Predicts phone identity at a segment's center frame without any learned parameters:
    v_hat = argmax_v sigma(m_{c(s)})^T p_v
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np


class RecognitionHead:
    """Parameter-free recognition head using nearest-neighbor dot product matching."""

    def __init__(
        self,
        canonical_inventory: Dict[str, np.ndarray],
        vocab_filter: Optional[List[str]] = None,
    ):
        """
        Args:
            canonical_inventory: Mapping from phone string to normalized canonical vector p_v in R^|C|.
            vocab_filter: Optional list of allowed phones (e.g. language inventory).
                          If None, uses the full PanPhon inventory.
        """
        if vocab_filter is not None:
            self.phones = [p for p in vocab_filter if p in canonical_inventory]
        else:
            self.phones = list(canonical_inventory.keys())

        if not self.phones:
            raise ValueError("No valid phones in canonical inventory matching vocab filter.")

        # Stack into matrix P of shape [V, C]
        self.P = np.stack([canonical_inventory[p] for p in self.phones], axis=0).astype(np.float32)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        """Element-wise numerically stable sigmoid."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def predict_frame(
        self,
        spam_vector: np.ndarray,
        silence_channel_idx: Optional[int] = None,
        silence_threshold: float = 0.5,
    ) -> Tuple[str, float]:
        """Predicts phone identity for a single frame SPAM vector m_t in R^|C|.
        
        Args:
            spam_vector: 1D array of shape [C].
            silence_channel_idx: Optional index of silence+ channel. If provided and
                                 activation exceeds silence_threshold, predicts '_' (silence).
            silence_threshold: Threshold on sigmoid(m_{t, silence+}) for silence prediction.
            
        Returns:
            predicted_phone: The phone string ('_' for silence, or PanPhon phone maximizing Eq. 4).
            score: The maximum dot-product score or silence probability.
        """
        sig_m = self.sigmoid(spam_vector)  # [C]

        # Handle silence per Section III-B and Figure 2 ("[0.00-0.14 _]")
        if silence_channel_idx is not None and 0 <= silence_channel_idx < len(sig_m):
            if sig_m[silence_channel_idx] >= silence_threshold:
                return "_", float(sig_m[silence_channel_idx])

        scores = self.P @ sig_m            # [V]
        best_idx = int(np.argmax(scores))
        return self.phones[best_idx], float(scores[best_idx])

    def predict_segment(
        self,
        spam: np.ndarray,
        start_frame: int,
        stop_frame: int,
        silence_channel_idx: Optional[int] = None,
        silence_threshold: float = 0.5,
    ) -> Tuple[str, float]:
        """Predicts phone identity for a segment s by reading SPAM at its temporal center c(s).
        
        Args:
            spam: 2D array of shape [T, C].
            start_frame: Start frame index of the segment.
            stop_frame: Stop frame index of the segment (exclusive).
            silence_channel_idx: Optional index of silence+ channel.
            silence_threshold: Threshold on sigmoid(m_{c(s), silence+}).
            
        Returns:
            predicted_phone: Best matching phone string ('_' for silence).
            score: Maximum score.
        """
        center_frame = (start_frame + stop_frame) // 2
        center_frame = min(max(center_frame, 0), spam.shape[0] - 1)
        return self.predict_frame(
            spam[center_frame],
            silence_channel_idx=silence_channel_idx,
            silence_threshold=silence_threshold,
        )

    def predict_utterance(
        self,
        spam: np.ndarray,
        boundary_frames: List[int],
        silence_channel_idx: Optional[int] = None,
        silence_threshold: float = 0.5,
    ) -> List[str]:
        """Predicts phone sequence for all segments delimited by boundary frames.
        
        Args:
            spam: 2D array of shape [T, C].
            boundary_frames: Sorted list of boundary frame indices.
            silence_channel_idx: Optional index of silence+ channel.
            silence_threshold: Threshold on sigmoid(m_{c(s), silence+}).
            
        Returns:
            predicted_phones: List of predicted phone strings for each interval.
        """
        T = spam.shape[0]
        # Ensure boundaries start at 0 and end at T
        bounds = sorted(list(set([0] + boundary_frames + [T])))
        
        predictions: List[str] = []
        for i in range(len(bounds) - 1):
            start = bounds[i]
            stop = bounds[i + 1]
            if stop <= start:
                continue
            phone, _ = self.predict_segment(
                spam,
                start,
                stop,
                silence_channel_idx=silence_channel_idx,
                silence_threshold=silence_threshold,
            )
            predictions.append(phone)

        return predictions
