"""Configuration module for SPAM reproduction."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Dataset paths
    timit_root: Path = Path(r"c:\SPAM\TIMIT_LDC93S1\TIMIT_LDC93S1\TIMIT")
    output_root: Path = Path(r"c:\SPAM\output")
    cache_dir: Path = Path(r"c:\SPAM\output\cache")

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
    silence_threshold: float = 0.7  # [UNSPECIFIED IN PAPER: best threshold on sigmoid(m_{t, silence+})]

    # Evaluation settings (Section IV-A, IV-B)
    tolerance_seconds: float = 0.02  # 20 ms boundary tolerance
    eval_mode: str = "strict"  # Strict mode greedy 1-to-1 matching

    # Reproducibility
    random_seed: int = 42

    def __post_init__(self):
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


default_config = Config()
