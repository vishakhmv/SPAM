"""Phonological vector estimation via difference-of-means (Section III-A, Eq. 1)
and SPAM normalization parameters alpha_i and lambda_i (Section III-B, Eq. 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class PhonologicalVectors:
    """Stores estimated phonological vectors and normalization constants."""
    channels: List[str]
    vectors: np.ndarray  # Shape [C, D]
    alphas: np.ndarray   # Shape [C]
    lambdas: np.ndarray  # Shape [C]
    mu_pos: np.ndarray   # Shape [C, D]
    mu_comp: np.ndarray  # Shape [C, D]
    pos_counts: np.ndarray  # Shape [C]
    comp_counts: np.ndarray  # Shape [C]

    @property
    def num_channels(self) -> int:
        return len(self.channels)

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]


class VectorEstimator:
    """Accumulates center-pooled frame representations to estimate difference-of-means vectors."""

    def __init__(self, channels: List[str], dim: int = 1024):
        self.channels = list(channels)
        self.channel_to_idx = {ch: i for i, ch in enumerate(self.channels)}
        self.num_channels = len(channels)
        self.dim = dim

        # Running accumulators
        self.sum_pos = np.zeros((self.num_channels, dim), dtype=np.float64)
        self.count_pos = np.zeros(self.num_channels, dtype=np.int64)

        self.sum_total = np.zeros(dim, dtype=np.float64)
        self.count_total = 0

    def add_segment(self, rep_center: np.ndarray, active_channels: set[str]):
        """Adds a single segment's center-pooled representation to the accumulators.
        
        Args:
            rep_center: 1D numpy array of shape [D].
            active_channels: set of channel names that are active (1) for this segment.
        """
        rep = rep_center.astype(np.float64)
        self.sum_total += rep
        self.count_total += 1

        for ch in active_channels:
            if ch in self.channel_to_idx:
                idx = self.channel_to_idx[ch]
                self.sum_pos[idx] += rep
                self.count_pos[idx] += 1

    def compute_vectors(self) -> PhonologicalVectors:
        """Computes difference-of-means vectors v_i, alpha_i, and lambda_i.
        
        Discards channels with empty positive or complement sets.
        """
        retained_channels = []
        vectors_list = []
        alphas_list = []
        lambdas_list = []
        mu_pos_list = []
        mu_comp_list = []
        pos_counts_list = []
        comp_counts_list = []

        for i, ch in enumerate(self.channels):
            pos_cnt = self.count_pos[i]
            comp_cnt = self.count_total - pos_cnt

            if pos_cnt == 0 or comp_cnt == 0:
                print(f"Skipping channel '{ch}' with pos={pos_cnt}, comp={comp_cnt}")
                continue

            mu_i = self.sum_pos[i] / pos_cnt
            sum_comp = self.sum_total - self.sum_pos[i]
            mu_comp_i = sum_comp / comp_cnt

            # Equation (1): v_i = mu_i - mu_comp_i
            v_i = mu_i - mu_comp_i

            # Equation (3):
            # alpha_i = 1/2 * (mu_i^T v_i + mu_comp_i^T v_i)
            # lambda_i = mu_i^T v_i - mu_comp_i^T v_i
            alpha_i = 0.5 * (np.dot(mu_i, v_i) + np.dot(mu_comp_i, v_i))
            lambda_i = np.dot(mu_i, v_i) - np.dot(mu_comp_i, v_i)

            # Avoid division by zero if vectors are somehow identical
            if lambda_i <= 1e-12:
                print(f"Skipping degenerate channel '{ch}' with lambda={lambda_i}")
                continue

            retained_channels.append(ch)
            vectors_list.append(v_i.astype(np.float32))
            alphas_list.append(float(alpha_i))
            lambdas_list.append(float(lambda_i))
            mu_pos_list.append(mu_i.astype(np.float32))
            mu_comp_list.append(mu_comp_i.astype(np.float32))
            pos_counts_list.append(int(pos_cnt))
            comp_counts_list.append(int(comp_cnt))

        return PhonologicalVectors(
            channels=retained_channels,
            vectors=np.stack(vectors_list, axis=0),
            alphas=np.array(alphas_list, dtype=np.float32),
            lambdas=np.array(lambdas_list, dtype=np.float32),
            mu_pos=np.stack(mu_pos_list, axis=0),
            mu_comp=np.stack(mu_comp_list, axis=0),
            pos_counts=np.array(pos_counts_list, dtype=np.int64),
            comp_counts=np.array(comp_counts_list, dtype=np.int64),
        )
