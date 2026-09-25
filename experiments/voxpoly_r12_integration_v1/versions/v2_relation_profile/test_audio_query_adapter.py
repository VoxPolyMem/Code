from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    from .audio_query_adapter import (
        AudioIdentityContractError,
        _profile_index,
        build_sidecar,
        match_embedding,
        resolve_query_audio,
        sha256_file,
    )
except ImportError:
    from audio_query_adapter import (
        AudioIdentityContractError,
        _profile_index,
        build_sidecar,
        match_embedding,
        resolve_query_audio,
        sha256_file,
    )


class AudioQueryAdapterTests(unittest.TestCase):
    def test_confidence_policy(self) -> None:
        common = {
            "acoustic_speaker_ids": ["A", "B"],
            "prototypes": [[1.0, 0.0], [0.0, 1.0]],
            "acoustic_to_stable": {"A": "SPK_A", "B": "SPK_B"},
            "stable_to_display": {"SPK_A": "Alice", "SPK_B": "Bob"},
        }
        high = match_embedding([1.0, 0.0], **common)
        self.assertEqual((high["stable_asker_ref"], high["confidence"]), ("SPK_A", "high"))
        low = match_embedding([1.0, 0.99], **common)
        self.assertEqual((low["stable_asker_ref"], low["confidence"]), ("SPK_A", "low"))
        unresolved = match_embedding([-1.0, -1.0], **common)
        self.assertEqual((unresolved["stable_asker_ref"], unresolved["confidence"]), (None, "unresolved"))

    def test_forbidden_offline_name_or_gold_is_rejected_before_encoding(self) -> None:
        with self.assertRaises(AudioIdentityContractError):
            build_sidecar({"items": [{"qa_id": "P1", "asker_name": "leak"}]})
        with self.assertRaises(AudioIdentityContractError):
            build_sidecar({"items": [{"qa_id": "P1", "gold_evidence_ids": ["T1"]}]})

    def test_registry_sha_mismatch_fails_closed(self) -> None:
        memory = {
            "profiles": [{
                "stable_speaker_id": "SPK_A",
                "speaker_name": "Alice",
                "acoustic_speaker_ids": ["A"],
                "registry_state_sha256": "wrong",
            }]
        }
        with self.assertRaises(AudioIdentityContractError):
            _profile_index(memory, "actual")

    @unittest.skipIf(np is None, "numpy unavailable")
    def test_waveform_path_name_cannot_override_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "registry.npz"
            np.savez(
                registry,
                speaker_ids=np.array(["A", "B"]),
                prototypes=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
            registry_sha = sha256_file(registry)
            memory = root / "memory.json"
            memory.write_text(json.dumps({"profiles": [
                {
                    "stable_speaker_id": "SPK_A",
                    "speaker_name": "Alice",
                    "acoustic_speaker_ids": ["A"],
                    "registry_state_sha256": registry_sha,
                },
                {
                    "stable_speaker_id": "SPK_B",
                    "speaker_name": "Bob",
                    "acoustic_speaker_ids": ["B"],
                    "registry_state_sha256": registry_sha,
                },
            ]}), encoding="utf-8")
            audio = root / "PERSONA_Bob.wav"
            audio.write_bytes(b"not-used-by-fake-encoder")
            encoder = root / "fake_encoder.py"
            encoder.write_text(
                "class EcapaEncoder:\n"
                "    def __init__(self, device): self.device = device\n"
                "    def encode(self, path): return [1.0, 0.0]\n",
                encoding="utf-8",
            )
            result = resolve_query_audio(
                qa_id="P1",
                case_id="CASE_X",
                audio_path=str(audio),
                memory_path=str(memory),
                registry_path=str(registry),
                encoder_module_path=str(encoder),
                device="cpu",
            )
            self.assertEqual(result.stable_asker_ref, "SPK_A")
            self.assertEqual(result.profile_display, "Alice")
            self.assertFalse(result.uses_filename_identity)


if __name__ == "__main__":
    unittest.main()
