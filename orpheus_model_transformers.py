import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import OrderedDict
from snac_model_transformers import SnacModel
from stopwatch import Stopwatch
import time

from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

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
                logging.info("Using cached input_ids")
                zeroprompt_input_ids = self._cache[cache_key]
            else:
                logging.info("Tokenizing voice")
                st = time.monotonic()
                voice_audio_tokens = self.audio_model.load_audio(voice_path)
                voice_transcript_ids = self.tokenizer(voice_text, return_tensors="pt")["input_ids"]
                zeroprompt_input_ids = torch.cat([start_tokens, voice_transcript_ids, end_tokens, torch.tensor([voice_audio_tokens], dtype=torch.int64), final_tokens], dim=1)
                logging.info(f"Tokenized voice ({zeroprompt_input_ids.size(1)} tokens) in {time.monotonic() - st:.2f}s")
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
    ) -> bytes:
        if greedy_snac_tokens > 0:
            logging.warning("greedy_snac_tokens is not supported with safetensors")
        if use_continuation:
            logging.warning("use_continuation is not supported with safetensors")

        torch.cuda.empty_cache()

        input_ids, attention_mask = self._format_prompt(finetune_voice, voice_path, voice_transcript, prompt)

        ## Print first and last 2 tokens
        debug_tokens = []
        for token in input_ids[0, :4]:
            token_str = self.tokenizer.decode([token])
            debug_tokens.append(f'{token} "{token_str}"')
        debug_tokens.append("...")
        for token in input_ids[0, -4:]:
            token_str = self.tokenizer.decode([token])
            debug_tokens.append(f'{token} "{token_str}"')
        logging.info(f"Input: {', '.join(debug_tokens)}")

        st = time.monotonic()
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
                pad_token_id=128258,
            )
        generated = generated_ids[0].tolist()
        generated = generated[len(input_ids[0]):]
        logging.info(f"Generated {len(generated)} tokens in {time.monotonic() - st:.2f}s")

        debug_tokens = []
        for token in generated[:8]:
            token_str = self.tokenizer.decode([token])
            debug_tokens.append(f'{token} "{token_str}"')
        debug_tokens.append("...")
        for token in generated[-8:]:
            token_str = self.tokenizer.decode([token])
            debug_tokens.append(f'{token} "{token_str}"')

        logging.info(f"Tokens: {', '.join(debug_tokens)}")

        with Stopwatch("Converting speech to audio") as sw:
            audio = self.audio_model.convert_audio_tokens_to_speech(generated)
            bytes = self.audio_model.to_audio_bytes(audio)
            sw.append(f"{len(bytes) / 1024:.2f} KB")
            return bytes