from os import path
from pathlib import Path
import soundfile as sf
import numpy as np
import json
import torch
import safetensors.torch as st
import logging
from snac import SNAC
from stopwatch import Stopwatch

logger = logging.getLogger(__name__)

class SnacModel:
    def __init__(self, snac_path: str):
        self.snac_device = "cuda"
        config = json.load(open(path.join(snac_path, "config.json")))
        model = SNAC(**config)
        safetensors_path = next(Path(snac_path).glob("*.safetensors"))
        st.load_model(model, safetensors_path, device=self.snac_device)
        self.snac_model = torch.compile(
            model.to(self.snac_device),
            mode="max-autotune"
        )

    def load_audio(self, voice_path: str):
        voice_audio, sr = sf.read(voice_path, dtype='float32')
        max_seconds = 45
        max_len = max_seconds * sr
        if len(voice_audio) > max_len:
            raise ValueError(f"Audio too long: {len(voice_audio)} samples ({len(voice_audio) / sr:.2f} seconds), maximum is {max_seconds} seconds")
        if voice_audio.ndim > 1:
            voice_audio = voice_audio.mean(axis=1)
        target_sr = 24000
        if sr != target_sr:
            scale = target_sr / sr
            n_samples = int(len(voice_audio) * scale)
            indices = np.linspace(0, len(voice_audio)-1, n_samples)
            voice_audio = np.interp(indices, np.arange(len(voice_audio)), voice_audio)
        return self.tokenize_audio(voice_audio)

    def tokenize_audio(self, waveform: np.array) -> list[int]:
        with Stopwatch("SNAC tokenization"):
            waveform_tensor = torch.tensor(waveform, dtype=torch.float32, device=self.snac_device).unsqueeze(0).unsqueeze(0)
            with torch.inference_mode(), torch.no_grad():
                codes = self.snac_model.encode(waveform_tensor)

            c0, c1, c2 = codes
            L = c0.shape[1]
            all_codes = torch.empty(7 * L, dtype=torch.int64, device=self.snac_device)
            all_codes[0::7] = c0[0, :L] + 128266
            all_codes[1::7] = c1[0, :2*L][::2] + 128266 + 4096
            all_codes[2::7] = c2[0, :4*L][::4] + 128266 + 2 * 4096
            all_codes[3::7] = c2[0, :4*L][1::4] + 128266 + 3 * 4096
            all_codes[4::7] = c1[0, :2*L][1::2] + 128266 + 4 * 4096
            all_codes[5::7] = c2[0, :4*L][2::4] + 128266 + 5 * 4096
            all_codes[6::7] = c2[0, :4*L][3::4] + 128266 + 6 * 4096
            return all_codes.tolist()

    def convert_audio_tokens_to_speech(self, generated: list[int]) -> torch.Tensor:
        tokens = torch.tensor([generated], dtype=torch.int64)
        assert tokens.ndim == 2 and tokens.size(0) == 1, "Expected shape (1, T)"
        row = tokens[0]
        usable_len = (row.size(0) // 7) * 7
        if usable_len == 0:
            return None
        trimmed = row[:usable_len] - 128266
        try:
            result = self._redistribute_codes(trimmed.tolist())
            return result
        except:
            logger.exception(f"Error during decoding tokens: {row[:usable_len]}")
            return None

    def _redistribute_codes(self, code_list: list[int]) -> torch.Tensor:
        if len(code_list) < 7:
            raise ValueError("Insufficient token length: {}".format(len(code_list)))
        layer_1, layer_2, layer_3 = [], [], []
        for i in range(len(code_list) // 7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1] - 4096)
            layer_3.append(code_list[7*i+2] - 2*4096)
            layer_3.append(code_list[7*i+3] - 3*4096)
            layer_2.append(code_list[7*i+4] - 4*4096)
            layer_3.append(code_list[7*i+5] - 5*4096)
            layer_3.append(code_list[7*i+6] - 6*4096)
        codes = [
            torch.tensor(layer_1, dtype=torch.int64, device=self.snac_device).unsqueeze(0),
            torch.tensor(layer_2, dtype=torch.int64, device=self.snac_device).unsqueeze(0),
            torch.tensor(layer_3, dtype=torch.int64, device=self.snac_device).unsqueeze(0),
        ]

        if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
            raise ValueError(f"Invalid SNAC tokens received")

        with torch.inference_mode():
            return self.snac_model.decode(codes)

    def to_numpy_array(self, samples: torch.Tensor) -> np.ndarray:
        return samples.squeeze().cpu().numpy()
    
    def to_bytes(self, array: np.ndarray) -> bytes:
        return (array * 32767).astype(np.int16).tobytes()
