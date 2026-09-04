"""Pretrained WavLM representation extractor.

Extracts frame-level speech representations from the final transformer layer (layer 24)
of microsoft/wavlm-large without any fine-tuning or architecture modifications.
Supports disk caching of extracted representations (.npy).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union
import numpy as np
import torch
from transformers import WavLMModel

from src.config import Config, default_config


class WavlmExtractor:
    """Extracts frame-level representations from a frozen WavLM model."""

    def __init__(
        self,
        model_name: str = "microsoft/wavlm-large",
        layer: int = 24,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.layer = layer
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading {model_name} onto {self.device}...")
        self.model = WavLMModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        # Freeze all parameters explicitly
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract(self, waveform: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Extracts representations for a single 1D audio waveform at 16 kHz.
        
        Args:
            waveform: 1D numpy array or torch tensor of audio samples.
            
        Returns:
            representations: 2D numpy array of shape [T, 1024] from the specified layer.
        """
        if isinstance(waveform, np.ndarray):
            audio_tensor = torch.from_numpy(waveform).float()
        else:
            audio_tensor = waveform.float()

        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)  # [1, N]
        elif audio_tensor.ndim > 2:
            raise ValueError(f"Expected 1D or 2D audio tensor, got shape {audio_tensor.shape}")

        audio_tensor = audio_tensor.to(self.device)

        outputs = self.model(audio_tensor, output_hidden_states=True)

        # hidden_states is a tuple of length 25:
        # [0] is CNN feature projection, [1..24] are transformer layers
        if self.layer == 24 or self.layer == -1:
            # Final transformer layer
            rep = outputs.last_hidden_state[0]  # [T, 1024]
        else:
            rep = outputs.hidden_states[self.layer][0]  # [T, 1024]

        return rep.cpu().numpy().astype(np.float32)

    def extract_and_cache(
        self,
        utt_id: str,
        waveform: Union[np.ndarray, torch.Tensor],
        cache_dir: Path,
    ) -> np.ndarray:
        """Loads representation from disk cache if present, otherwise extracts and saves."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{utt_id}_wavlm_l{self.layer}.npy"

        if cache_file.exists():
            return np.load(cache_file)

        rep = self.extract(waveform)
        np.save(cache_file, rep)
        return rep
