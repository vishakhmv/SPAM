"""Segmentation head implementing all 7 signals, multiplicative ensemble,
silence suppression, and prominence-based peak detection (Section III-D, Equations 5-11).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import scipy.signal
import torchaudio
import torch

from src.phonological_vectors import PhonologicalVectors


def cosine_distance_frames(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes 1 - cos(u_t, v_t) along the time axis (axis 0).
    
    Args:
        u: array of shape [T, C]
        v: array of shape [T, C]
        
    Returns:
        distance: 1D array of shape [T]
    """
    u_norm = np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    cos_sim = np.sum((u / u_norm) * (v / v_norm), axis=-1)
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return 1.0 - cos_sim


def cosine_similarity_frames(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Computes cos(u_t, v_t) along the time axis (axis 0)."""
    u_norm = np.linalg.norm(u, axis=-1, keepdims=True) + 1e-12
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    cos_sim = np.sum((u / u_norm) * (v / v_norm), axis=-1)
    return np.clip(cos_sim, -1.0, 1.0)


class BackwardRegressor:
    """Closed-form ordinary least-squares regressor for backward contrast (Section III-D, Eq. 8).
    
    W_hat = argmin_W E_{l, s} || W^T r_{c(s)} - m_{c(l)} ||^2
    Solved directly with unregularized ordinary least squares:
        argmin_W || R_curr W - M_prev ||_F^2
    No ridge regularization or intercept is added per the exact paper specification.
    """

    def __init__(self):
        self.W: Optional[np.ndarray] = None  # Shape [D, C]

    def fit(self, R_curr: np.ndarray, M_prev: np.ndarray):
        """Fits W_hat in closed form using unregularized ordinary least squares (Equation 8).
        
        Args:
            R_curr: 2D array of shape [N_pairs, D] containing r_{c(s)} for phone s.
            M_prev: 2D array of shape [N_pairs, C] containing m_{c(l)} for immediately preceding phone l.
        """
        # Exact unregularized ordinary least squares solution
        self.W = np.linalg.lstsq(R_curr, M_prev, rcond=None)[0].astype(np.float32)

    def predict(self, R: np.ndarray) -> np.ndarray:
        """Predicts previous phone activations: m_hat_t = W^T r_t (Equation 8).
        
        In batch form: R @ W has shape [T, C].
        
        Args:
            R: 2D array of shape [T, D].
            
        Returns:
            m_hat: 2D array of shape [T, C].
        """
        if self.W is None:
            raise ValueError("BackwardRegressor has not been fitted.")
        return (R @ self.W).astype(np.float32)


class SegmentationHead:
    """Computes the seven segmentation signals, ensembling, and peak detection."""

    def __init__(
        self,
        backward_regressor: Optional[BackwardRegressor] = None,
        silence_channel_idx: Optional[int] = None,
        prominence: float = 1e-6,
        min_distance: int = 2,
        silence_threshold: float = 0.5,
        sample_rate: int = 16000,
        mel_bins: int = 40,
    ):
        self.backward_regressor = backward_regressor
        self.silence_channel_idx = silence_channel_idx
        self.prominence = prominence
        self.min_distance = min_distance
        self.silence_threshold = silence_threshold
        self.sample_rate = sample_rate
        self.mel_bins = mel_bins

        # Mel-spectrogram transform (10 ms hop = 160 samples, 25 ms win = 400 samples)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=mel_bins,
            power=2.0,
        )

    def compute_multi_scale_differences(
        self, spam: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes delta_1, delta_2, delta_3 (Equations 5, 6, 7).
        
        delta_1(t) = 1 - cos(m_{t-1}, m_t)
        delta_2(t) = 1 - cos(m_{t-1}, m_{t+1})
        delta_3(t) = 1 - cos(m_{t-2}, m_{t+1})
        """
        T = spam.shape[0]
        t_indices = np.arange(T)

        idx_tm2 = np.clip(t_indices - 2, 0, T - 1)
        idx_tm1 = np.clip(t_indices - 1, 0, T - 1)
        idx_t   = t_indices
        idx_tp1 = np.clip(t_indices + 1, 0, T - 1)

        m_tm2 = spam[idx_tm2]
        m_tm1 = spam[idx_tm1]
        m_t   = spam[idx_t]
        m_tp1 = spam[idx_tp1]

        delta1 = cosine_distance_frames(m_tm1, m_t)
        delta2 = cosine_distance_frames(m_tm1, m_tp1)
        delta3 = cosine_distance_frames(m_tm2, m_tp1)

        return delta1, delta2, delta3

    def compute_backward_contrasts(
        self, R: np.ndarray, spam: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes backward contrast signals beta_1, beta_2, beta_3 (Equation 9).
        
        beta_l(t) = cos(m_hat_{t+a}, m_{t+a-l}) - cos(m_hat_{t+a}, m_{t+a})
        where a = floor(l/2).
        For l=1, a=0: beta_1(t) = cos(m_hat_t, m_{t-1}) - cos(m_hat_t, m_t)
        For l=2, a=1: beta_2(t) = cos(m_hat_{t+1}, m_{t-1}) - cos(m_hat_{t+1}, m_{t+1})
        For l=3, a=1: beta_3(t) = cos(m_hat_{t+1}, m_{t-2}) - cos(m_hat_{t+1}, m_{t+1})
        """
        if self.backward_regressor is None:
            raise ValueError("Backward regressor is required for computing beta signals.")

        T = spam.shape[0]
        m_hat = self.backward_regressor.predict(R)  # [T, C]
        t_indices = np.arange(T)

        # l = 1, a = 0
        m_hat_t = m_hat[t_indices]
        m_tm1 = spam[np.clip(t_indices - 1, 0, T - 1)]
        m_t = spam[t_indices]
        beta1 = cosine_similarity_frames(m_hat_t, m_tm1) - cosine_similarity_frames(m_hat_t, m_t)

        # l = 2, a = 1
        m_hat_tp1 = m_hat[np.clip(t_indices + 1, 0, T - 1)]
        m_tp1 = spam[np.clip(t_indices + 1, 0, T - 1)]
        beta2 = cosine_similarity_frames(m_hat_tp1, m_tm1) - cosine_similarity_frames(m_hat_tp1, m_tp1)

        # l = 3, a = 1
        m_tm2 = spam[np.clip(t_indices - 2, 0, T - 1)]
        beta3 = cosine_similarity_frames(m_hat_tp1, m_tm2) - cosine_similarity_frames(m_hat_tp1, m_tp1)

        return beta1, beta2, beta3

    def compute_mel_signal(
        self, waveform: np.ndarray, target_length: int
    ) -> np.ndarray:
        """Computes mel spectrogram signal delta_mel (Equation 10).
        
        delta_mel(u) = 1 - cos(F_{u-2}, F_{u+1}) on 10ms grid.
        Min-max normalized and subsampled by taking every other frame.
        """
        wav_tensor = torch.from_numpy(waveform).float().unsqueeze(0)
        with torch.no_grad():
            mel_spec = self.mel_transform(wav_tensor)  # [1, 40, T_mel]
            log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))[0].transpose(0, 1).cpu().numpy()  # [T_mel, 40]

        T_mel = log_mel.shape[0]
        u_indices = np.arange(T_mel)
        idx_um2 = np.clip(u_indices - 2, 0, T_mel - 1)
        idx_up1 = np.clip(u_indices + 1, 0, T_mel - 1)

        F_um2 = log_mel[idx_um2]
        F_up1 = log_mel[idx_up1]

        delta_mel_raw = cosine_distance_frames(F_um2, F_up1)  # [T_mel]

        # Min-max normalize over utterance
        min_val = np.min(delta_mel_raw)
        max_val = np.max(delta_mel_raw)
        if max_val > min_val:
            delta_mel_norm = (delta_mel_raw - min_val) / (max_val - min_val)
        else:
            delta_mel_norm = np.zeros_like(delta_mel_raw)

        # Subsample onto S3M grid by taking every other frame (10ms -> 20ms)
        delta_mel_sub = delta_mel_norm[0::2]

        # Align length to target_length (S3M frames T)
        if len(delta_mel_sub) < target_length:
            delta_mel = np.pad(delta_mel_sub, (0, target_length - len(delta_mel_sub)), mode="edge")
        else:
            delta_mel = delta_mel_sub[:target_length]

        return delta_mel

    def compute_ensemble_signal(
        self,
        waveform: np.ndarray,
        R: np.ndarray,
        spam: np.ndarray,
    ) -> Tuple[np.ndarray, dict]:
        """Computes the multiplicative ensemble signal b(t) from all seven signals (Equation 11).
        
        b(t) = prod_k (b_k(t) - phi_k)
        with silence suppression applied.
        """
        T = spam.shape[0]

        # 1-3: multi-scale differences
        delta1, delta2, delta3 = self.compute_multi_scale_differences(spam)

        # 4-6: backward contrasts
        beta1, beta2, beta3 = self.compute_backward_contrasts(R, spam)

        # 7: mel spectrogram signal
        delta_mel = self.compute_mel_signal(waveform, target_length=T)

        # Theoretical minima phi_k:
        # For cosine distances (delta1, delta2, delta3, delta_mel): phi = 0
        # For cosine differences (beta1, beta2, beta3): phi = -2
        term_d1 = np.maximum(delta1 - 0.0, 0.0)
        term_d2 = np.maximum(delta2 - 0.0, 0.0)
        term_d3 = np.maximum(delta3 - 0.0, 0.0)
        term_b1 = np.maximum(beta1 - (-2.0), 0.0)  # beta1 + 2 >= 0
        term_b2 = np.maximum(beta2 - (-2.0), 0.0)  # beta2 + 2 >= 0
        term_b3 = np.maximum(beta3 - (-2.0), 0.0)  # beta3 + 2 >= 0
        term_mel = np.maximum(delta_mel - 0.0, 0.0)

        # Multiplicative ensemble (Equation 11)
        b = term_d1 * term_d2 * term_d3 * term_b1 * term_b2 * term_b3 * term_mel

        # Silence suppression (Section III-D):
        # "Additionally, using the silence channel, we suppress peaks falling inside silent spans."
        if self.silence_channel_idx is not None:
            silence_acts = spam[:, self.silence_channel_idx]
            # Sigmoid activation for silence
            silence_prob = 1.0 / (1.0 + np.exp(-np.clip(silence_acts, -50.0, 50.0)))
            is_silent = silence_prob >= self.silence_threshold
            b[is_silent] = 0.0

        signals_dict = {
            "delta1": delta1,
            "delta2": delta2,
            "delta3": delta3,
            "beta1": beta1,
            "beta2": beta2,
            "beta3": beta3,
            "delta_mel": delta_mel,
            "b_ensemble": b,
        }
        return b, signals_dict

    def detect_boundaries(
        self,
        b: np.ndarray,
        prominence: Optional[float] = None,
        min_distance: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Performs prominence-based peak detection on b(t).
        
        Returns:
            peak_frames: 1D array of integer frame indices.
            boundary_times: 1D array of timestamps in seconds (peak_frames * 0.02s).
        """
        prom = prominence if prominence is not None else self.prominence
        dist = min_distance if min_distance is not None else self.min_distance

        # Normalize b for scale-invariant prominence detection if desired, or use raw b
        # Find peaks
        peaks, _ = scipy.signal.find_peaks(b, prominence=prom, distance=dist)

        # A peak at frame t marks a boundary at the start of frame t (t * 20ms)
        boundary_times = peaks.astype(float) * 0.02

        return peaks, boundary_times
