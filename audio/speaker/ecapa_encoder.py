"""Minimal ECAPA waveform encoder used by the production speaker pipeline."""

from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal as scipy_signal


ECAPA_MODEL = Path(os.environ.get(
    "ECAPA_MODEL", "speechbrain/spkrec-ecapa-voxceleb"
))
SAMPLE_RATE = 16000


def _install_soundfile_torchaudio_shim() -> None:
    """Provide the small torchaudio surface imported by this SpeechBrain build."""

    fake = types.ModuleType("torchaudio")
    fake.__version__ = "2.3.0"
    fake.functional = types.ModuleType("torchaudio.functional")
    fake.functional.resample = lambda waveform, _source, _target: waveform
    fake.transforms = types.ModuleType("torchaudio.transforms")
    fake.load = lambda *args, **kwargs: (None, None)
    fake.save = lambda *args, **kwargs: None
    fake.get_audio_backend = lambda: "soundfile"
    fake.set_audio_backend = lambda *args, **kwargs: None
    fake.list_audio_backends = lambda: ["soundfile"]
    sys.modules["torchaudio"] = fake
    sys.modules["torchaudio.functional"] = fake.functional
    sys.modules["torchaudio.transforms"] = fake.transforms


class EcapaEncoder:
    def __init__(self, device: str = "cuda:0") -> None:
        _install_soundfile_torchaudio_shim()
        from speechbrain.inference.speaker import SpeakerRecognition

        self.model = SpeakerRecognition.from_hparams(
            source=str(ECAPA_MODEL),
            savedir="/tmp/voxpoly_ecapa",
            run_opts={"device": device},
        )
        self.device = device

    def encode(self, audio_path: str | Path) -> np.ndarray | None:
        waveform, sample_rate = sf.read(str(audio_path))
        waveform = waveform.astype(np.float32)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            waveform = scipy_signal.resample_poly(
                waveform, SAMPLE_RATE, sample_rate
            ).astype(np.float32)
        if len(waveform) == 0:
            return None

        # Repeat only the encoder input for very short acknowledgements.  The
        # source waveform and turn boundary remain unchanged.
        minimum_samples = int(SAMPLE_RATE * 0.5)
        if len(waveform) < minimum_samples:
            copies = int(math.ceil(minimum_samples / len(waveform)))
            waveform = np.tile(waveform, copies)[:minimum_samples]

        tensor = torch.from_numpy(waveform).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(tensor.unsqueeze(0))
        vector = embedding.squeeze().cpu().numpy()
        return vector / (np.linalg.norm(vector) + 1e-8)
