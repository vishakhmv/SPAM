"""Configuration module for SPAM reproduction."""

import os
from dataclasses import dataclass, field
from pathlib import Path


# Auto-detect if running inside Google Colab with Drive mounted
IS_COLAB = Path("/content/drive/MyDrive/SPAM").exists()


@dataclass
class Config:
    # Dataset paths: uses your exact Drive location on Colab, local path on Windows
    timit_root: Path = (
        Path("/content/drive/MyDrive/SPAM/TIMIT_LDC93S1/TIMIT_LDC93S1/TIMIT")
        if IS_COLAB else Path(r"c:\SPAM\TIMIT_LDC93S1\TIMIT_LDC93S1\TIMIT")
    )
    output_root: Path = (
        Path("/content/drive/MyDrive/SPAM/output")
        if IS_COLAB else Path(r"c:\SPAM\output")
    )
    cache_dir: Path = (
        Path("/content/cache")
        if Path("/content").exists() else Path(r"c:\SPAM\output\cache")
    )

    # Audio & Model settings
    sample_rate: int = 16000
    wavlm_model: str = "microsoft/wavlm-large"
    wavlm_layer: int = 24  # Final transformer layer
    wavlm_stride_samples: int = 320  # 20 ms at 16 kHz
    wavlm_receptive_samples: int = 400  # 25 ms at 16 kHz
    wavlm_dim: int = 1024

    # SPAM projection settings
    gamma: float = 4.0  # Scaling constant from Section III-B, Footnote 1

    # Mel-spectrogram signal settings (Section III-D, Equation 10)
    mel_bins: int = 40
    mel_hop_ms: float = 10.0  # 10 ms grid
    mel_window_ms: float = 25.0

    # Segmentation peak detection & suppression settings (Section III-D)
    prominence_threshold: float = 1e-6  # [UNSPECIFIED IN PAPER: best value for 7-signal product ensemble]
    min_peak_distance_frames: int = 2  # Min distance between boundary peaks (40 ms)
    silence_threshold: float = 0.5  # [UNSPECIFIED IN PAPER: best threshold on sigmoid(m_{t, silence+}) corresponding to midpoint alpha_i]

    # Evaluation settings (Section IV-A, IV-B)
    tolerance_seconds: float = 0.02  # 20 ms boundary tolerance
    eval_mode: str = "strict"  # Strict mode greedy 1-to-1 matching

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


default_config = Config()
