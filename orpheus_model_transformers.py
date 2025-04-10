import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Iterable
from collections import OrderedDict
from snac_model_transformers import SnacModel
from stopwatch import Stopwatch
import time

from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

logger = logging.getLogger(__name__)

class OrpheusModelTransformers:
    def __init__(
            self,
            model_name: str,
            audio_model: SnacModel,
            dtype=torch.bfloat16, device="cuda" if torch.cuda.is_available() else "cpu"):
        self._cache = OrderedDict()

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)
        self.model = torch.compile(model, mode="max-autotune")

        self.audio_model = audio_model

        logger.warning(f"Orpheus: Using transformers for model inference, this is only supported for testing new safetensors versions, it has limited features and performance.")

    def _format_prompt(self, finetune_voice, voice_path, voice_text, text):
        # Tokenize the actual prompt text (for model generation)
        if finetune_voice:
            text = f"{finetune_voice}: {text}"
        prompt_input_ids = self.tokenizer(text, return_tensors="pt")["input_ids"]

        start_tokens = torch.tensor([[128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)

        if voice_path:
            cache_key = (voice_path, voice_text)
            if cache_key in self._cache:
                logger.debug("Using cached input_ids")
                zeroprompt_input_ids = self._cache[cache_key]
            else:
                logger.debug("Tokenizing voice")
                st = time.monotonic()
                voice_audio_tokens = self.audio_model.load_audio(voice_path)
                voice_transcript_ids = self.tokenizer(voice_text, return_tensors="pt")["input_ids"]
                zeroprompt_input_ids = torch.cat([start_tokens, voice_transcript_ids, end_tokens, torch.tensor([voice_audio_tokens], dtype=torch.int64), final_tokens], dim=1)
                logger.debug(f"Tokenized voice ({zeroprompt_input_ids.size(1)} tokens) in {time.monotonic() - st:.2f}s")
                if len(self._cache) > 9:
                    self._cache.popitem(last=False)
                self._cache[cache_key] = zeroprompt_input_ids.cpu().detach().clone()

            # Merge everything
            final_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, prompt_input_ids, end_tokens], dim=1)
            attention_mask = torch.ones_like(final_input_ids, dtype=torch.int64, device=self.device)
            return final_input_ids.to(self.device), attention_mask
        else:
            final_input_ids = torch.cat([start_tokens, prompt_input_ids, end_tokens], dim=1)
            attention_mask = torch.ones_like(final_input_ids, dtype=torch.int64, device=self.device)
            return final_input_ids.to(self.device), attention_mask
    
    def generate_speech(
        self,
        prompt: str,
        finetune_voice: str,
        voice_path:str,
        voice_transcript:str,
        previous_text: str,
        reply_to: str,
        temperature: float,
        top_p: float,
        min_p: float,
        max_tokens: int,
        repetition_penalty: float,
        greedy_snac_tokens: int,
        use_continuation: bool,
    ) -> Iterable[bytes]:
        if greedy_snac_tokens > 0:
            logger.warning("greedy_snac_tokens is not supported with safetensors")
        if use_continuation:
            logger.warning("use_continuation is not supported with safetensors")

        torch.cuda.empty_cache()

        input_ids, attention_mask = self._format_prompt(finetune_voice, voice_path, voice_transcript, prompt)

        with Stopwatch("Generating speech") as st:
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    # top_k=40,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=1,
                    eos_token_id=128258,
                    pad_token_id=128263,
                )
            generated = generated_ids[0].tolist()
            generated = generated[len(input_ids[0]):]
            # Find index of eos
            if 128258 in generated:
                eos_index = generated.index(128258)
                if eos_index < len(generated) - 1:
                    st.append(f"waste: {len(generated) - eos_index}")
                generated = generated[:eos_index]
            st.append(f"{len(generated)} tokens")
        
        # check if generated is a multiple of 7
        if len(generated) % 7 != 0:
            logger.warning(f"Generated length is not a multiple of 7: {len(generated)}")
        min_token_id = 128266
        max_token_id = min_token_id + 8192
        bad_tokens = [t for t in generated if t < min_token_id or t > max_token_id]
        if len(bad_tokens) > 0:
            logger.warning(f"Generated {len(bad_tokens)} bad tokens: {bad_tokens}")

        with Stopwatch("Converting speech to audio") as sw:
            hat = self.audio_model.convert_audio_tokens_to_speech(generated)
            np_audio = self.audio_model.to_numpy_array(hat)
            bytes = self.audio_model.to_bytes(np_audio)
            sw.append(f"{len(bytes) / 1024:.2f} KB")
            yield bytes
