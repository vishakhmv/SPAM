"""SPAM (S3M-based Phonological Activation Mapping) representation module.

Computes the SPAM matrix M in R^(T x |C|) from frame-level S3M representations R in R^(T x D)
and phonological vectors V in R^(|C| x D) using Equation (2):
    m_{t, i} = (gamma / lambda_i) * (r_t^T v_i - alpha_i)
with gamma = 4.
"""

from __future__ import annotations

import numpy as np
from src.phonological_vectors import PhonologicalVectors


def compute_spam(
    representations: np.ndarray,
    phonological_vectors: PhonologicalVectors,
    gamma: float = 4.0,
) -> np.ndarray:
    """Projects frame-wise representations onto phonological vectors with affine normalization.
    
    Args:
        representations: 2D numpy array of shape [T, D] containing WavLM frame representations.
        phonological_vectors: PhonologicalVectors instance containing V [C, D], alphas [C], lambdas [C].
        gamma: Scaling constant, fixed to 4.0 as specified in Section III-B, Footnote 1.
        
    Returns:
        spam: 2D numpy array of shape [T, C] representing the time-aligned phonological activation map.
    """
    R = representations.astype(np.float32)  # [T, D]
    V = phonological_vectors.vectors.astype(np.float32)  # [C, D]
    alphas = phonological_vectors.alphas.astype(np.float32)  # [C]
    lambdas = phonological_vectors.lambdas.astype(np.float32)  # [C]

    # Projections: [T, D] @ [D, C] -> [T, C]
    projections = R @ V.T

    # Per-channel affine normalization: gamma / lambda_i * (r_t^T v_i - alpha_i)
    scale = (gamma / lambdas).reshape(1, -1)  # [1, C]
    spam = scale * (projections - alphas.reshape(1, -1))  # [T, C]

    return spam
