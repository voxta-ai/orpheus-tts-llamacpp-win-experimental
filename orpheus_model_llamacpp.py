import os
import struct
import logging
import numpy as np
from typing import Iterable, Optional
from ctypes import c_ubyte, c_long, POINTER, cast, create_string_buffer
from snac_model_transformers import SnacModel
from stopwatch import Stopwatch
from cache_manager import StateCacheManager
import llama_cpp

llama_cpp.llama_backend_init(numa=True)

logger = logging.getLogger(__name__)

class OrpheusModelLlamaCpp:
    def __init__(self, model_path: str, audio_model: SnacModel):
        model_params = llama_cpp.llama_model_default_params()
        model_params.n_gpu_layers = 52
        model_params.n_threads = max(os.cpu_count() // 2, 1)
        self.model = llama_cpp.llama_load_model_from_file(model_path.encode(), model_params)
        self.vocab = llama_cpp.llama_model_get_vocab(self.model)
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = 4096
        ctx_params.n_batch = 256
        self.ctx = llama_cpp.llama_new_context_with_model(self.model, ctx_params)

        self.tokens_state_cache = StateCacheManager(4)
        self.tokens_finetune_voice_prefix_cache = StateCacheManager(4)
        self.previous_generation_cache = None

        self.audio_model = audio_model
        self.prompt_formatter = OrpheusPromptFormatterLlamaCpp(self, audio_model)
        
        self.n_past = 0

        self._sampler_chain_params = None
        self._continuation_sampler_chain = None
        self._greedy_sampler_chain = None
        self._sampler_chain = None

    def tokenize_text(self, text: str) -> list[int]:
        prompt = text.encode("utf-8")
        max_tokens = len(prompt) + 1
        embd_inp = (llama_cpp.llama_token * max_tokens)()
        n_tokens = llama_cpp.llama_tokenize(
            self.vocab,
            prompt,
            len(prompt),
            embd_inp,
            max_tokens,
            False,
            False
        )
        return list(embd_inp[:n_tokens])
    
    def detokenize_text(self, tokens: list[int], special = False) -> str:
        tokens_array = (c_long * len(tokens))(*tokens)
        text = create_string_buffer(4096)
        llama_cpp.llama_detokenize(
            self.vocab,
            tokens_array,
            len(tokens),
            text,
            4096,
            special,
            special
        )
        return text.value.decode("utf-8")

    def process_tokens(self, tokens: list[int], batch_size=256) -> None:
        total = len(tokens)
        cursor = 0

        batch = llama_cpp.llama_batch_init(batch_size, 0, 1)
        log_enabled = logger.isEnabledFor(logging.DEBUG)
        try:
            while cursor < total:
                chunk = tokens[cursor:cursor + batch_size]
                k = len(chunk)
                if k == 0:
                    break

                if log_enabled:
                    logger.debug(f"Processing chunk of {k} tokens (total context: {self.n_past + k})")
                batch.n_tokens = k
                is_last_chunk = (cursor + batch_size) >= total

                for i, token in enumerate(chunk):
                    logits = is_last_chunk and (i == k - 1)
                    batch.token[i] = token
                    batch.pos[i] = self.n_past + i
                    batch.seq_id[i][0] = 0
                    batch.n_seq_id[i] = 1
                    batch.logits[i] = logits

                ret = llama_cpp.llama_decode(self.ctx, batch)

                if ret != 0:
                    logger.error(f"llama_decode failed at n_past {self.n_past}: {chunk}")
                    return

                self.n_past += k
                cursor += k
        finally:
            llama_cpp.llama_batch_free(batch)

    def generate_tokens(
            self,
            max_tokens: int,
            stop_token_ids: list[int],
            temperature: float,
            top_p: float,
            min_p: float,
            repetition_penalty: float,
            greedy_snac_tokens: int = 0,
            force_continuation: int = 0
        ) -> Iterable[int]:
        greedy_snac_tokens = min(7, max(0, greedy_snac_tokens))

        with Stopwatch("llama_cpp create samplers"):
            sampler_chain = self._acquire_sampler_chain(temperature, top_p, min_p, repetition_penalty)
            greedy_chain = self._acquire_greedy_chain() if greedy_snac_tokens else None
            continuation_chain = self._acquire_continuation_chain(temperature, top_p, min_p, repetition_penalty) if force_continuation else None

        continuation_remaining = force_continuation
        snac_index = 0
        snac_greedy = 7 - greedy_snac_tokens
        first = True
        batch = llama_cpp.llama_batch_init(1, 0, 1)

        with Stopwatch("generate_tokens"), Stopwatch("llamacpp sample", auto_start=False) as st_sample, Stopwatch("llamacpp decode", auto_start=False) as st_decode:
            try:
                for _ in range(max_tokens):
                    chain = sampler_chain
                    if greedy_chain and snac_index >= snac_greedy:
                        chain = greedy_chain
                    elif continuation_remaining > 0:
                        chain = continuation_chain or chain

                    snac_index = (snac_index + 1) % 7
                    if continuation_remaining > 0:
                        continuation_remaining -= 1

                    st_sample.start()
                    token_id = llama_cpp.llama_sampler_sample(chain, self.ctx, -1)
                    st_sample.stop()

                    if token_id in stop_token_ids:
                        if first:
                            logger.warning(f"Stopping early due to stop token {token_id}")
                        break
                    first = False

                    yield token_id
                    
                    st_decode.start()
                    batch.n_tokens = 1
                    batch.token[0] = token_id
                    batch.pos[0] = self.n_past
                    batch.seq_id[0][0] = 0
                    batch.n_seq_id[0] = 1
                    batch.logits[0] = True
                    ret = llama_cpp.llama_decode(self.ctx, batch)
                    st_decode.stop()
                    
                    if ret != 0:
                        logger.error(f"llama_decode failed at n_past {self.n_past}: {token_id}")
                        return

                    self.n_past += 1
            finally:
                llama_cpp.llama_batch_free(batch)
    
    def _acquire_greedy_chain(self):
        if self._greedy_sampler_chain:
            return self._greedy_sampler_chain
        sparams = llama_cpp.llama_sampler_chain_default_params()
        chain = llama_cpp.llama_sampler_chain_init(sparams)
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_greedy())
        self._greedy_sampler_chain = chain
        return chain

    def _acquire_continuation_chain(self, temperature: float, top_p: float, min_p: float, repetition_penalty: float):
        if self._validate_cached_chain(temperature, top_p, min_p, repetition_penalty):
            return self._continuation_sampler_chain
        sparams = llama_cpp.llama_sampler_chain_default_params()
        chain = llama_cpp.llama_sampler_chain_init(sparams)
        bias_arr = (llama_cpp.llama_logit_bias * 1)()
        bias_arr[0].token = 128258
        bias_arr[0].bias = -1e9
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_logit_bias(self.vocab, 1, bias_arr))
        self._configure_sampler_chain_common(chain, temperature, top_p, min_p, repetition_penalty)
        return chain

    def _acquire_sampler_chain(self, temperature: float, top_p: float, min_p: float, repetition_penalty: float):
        if self._validate_cached_chain(temperature, top_p, min_p, repetition_penalty):
            return self._sampler_chain
        sparams = llama_cpp.llama_sampler_chain_default_params()
        chain = llama_cpp.llama_sampler_chain_init(sparams)
        self._configure_sampler_chain_common(chain, temperature, top_p, min_p, repetition_penalty)
        return chain
    
    def _configure_sampler_chain_common(self, chain, temperature: float, top_p: float, min_p: float, repetition_penalty: float):
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_k(400))
        if repetition_penalty != 1:
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_penalties(64, repetition_penalty, 0.0, 0.0))
        if top_p < 1.0:
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_top_p(top_p, 1))
        if min_p > 0.0:
            llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_min_p(min_p, 1))
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_temp(temperature))
        seed = struct.unpack("I", os.urandom(4))[0]
        llama_cpp.llama_sampler_chain_add(chain, llama_cpp.llama_sampler_init_dist(seed))

    def _validate_cached_chain(self, temperature: float, top_p: float, min_p: float, repetition_penalty: float) -> bool:
        cache_params = (temperature, top_p, min_p, repetition_penalty)
        if not self._continuation_sampler_chain:
            return False

        if self._sampler_chain_params == cache_params:
            return True

        if self._continuation_sampler_chain:
            llama_cpp.llama_sampler_free(self._continuation_sampler_chain)
        if self._sampler_chain:
            llama_cpp.llama_sampler_free(self._sampler_chain)
        return False

    def _debug_logits(self, idx: int) -> None:
        def softmax(x: np.ndarray) -> np.ndarray:
            x = x - np.max(x)
            exp_x = np.exp(x)
            return exp_x / exp_x.sum()

        vocab_size = llama_cpp.llama_n_vocab(self.vocab)
        logits_ptr = llama_cpp.llama_get_logits_ith(self.ctx, idx)
        logits = np.ctypeslib.as_array(logits_ptr, shape=(vocab_size,)).copy()
        probs = softmax(logits)
        top_n = 10
        top_indices = np.argsort(probs)[-top_n:][::-1]
        for token_id in top_indices:
            prob = probs[token_id]
            logger.debug(f"{token_id}: {prob:.4%}")

    def save_state(self) -> tuple[bytes, int]:
        state_size = llama_cpp.llama_get_state_size(self.ctx)
        buf = create_string_buffer(state_size)
        llama_cpp.llama_copy_state_data(self.ctx, cast(buf, POINTER(c_ubyte)))
        return buf, self.n_past

    def restore_state(self, buf: bytes, n_past: int) -> None:
        llama_cpp.llama_set_state_data(self.ctx, cast(buf, POINTER(c_ubyte)))
        self.n_past = n_past
    
    def clear_kv_cache(self) -> None:
        llama_cpp.llama_kv_cache_clear(self.ctx)
        self.n_past = 0

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
        streaming: bool = True
    ) -> Iterable[bytes]:
        with Stopwatch("Full generation") as fullSw:
            cache_key = (finetune_voice, voice_path, voice_transcript) if voice_path else None

            voice_text_ids, voice_audio_ids = None, None
            previous_text_ids, previous_audio_ids = None, None
            request_text_ids = None

            with Stopwatch("- Tokenize prompt") as sw:
                prefixed_prompt = f"{finetune_voice}: {prompt}" if finetune_voice else f"...: {prompt}"
                request_text_ids = self.tokenize_text(prefixed_prompt)
                sw.append(f"{len(request_text_ids)} tokens")

            if voice_path:
                voice_cached_ids = self.tokens_state_cache.get(cache_key)
                if voice_cached_ids is None:
                    with Stopwatch("- Loading voice audio") as sw:
                        prefixed_voice_transcript = f"{finetune_voice}: {voice_transcript}" if finetune_voice else f" {voice_transcript}"
                        voice_text_ids = self.tokenize_text(prefixed_voice_transcript)
                        voice_audio_ids = self.audio_model.load_audio(voice_path)
                        self.tokens_state_cache.add(cache_key, (voice_text_ids, voice_audio_ids))
                        sw.append(f"{len(voice_audio_ids)} voice tokens (cache miss)")
                        sw.append(f"{len(voice_text_ids)} transcript tokens (cache miss)")
                else:
                    voice_text_ids, voice_audio_ids = voice_cached_ids
            
            if self.previous_generation_cache is not None:
                previous_text_ids, previous_audio_ids = self.previous_generation_cache
                if self.previous_generation_cache != voice_path:
                    self.previous_generation_cache = None
                    previous_text_ids, previous_audio_ids = None, None
            
            with Stopwatch("- Processing prompt") as sw:
                prompt_ids = self.prompt_formatter.format_prompt(
                    voice_text_ids=voice_text_ids,
                    voice_audio_ids=voice_audio_ids,
                    previous_text_ids=previous_text_ids,
                    previous_audio_ids=previous_audio_ids,
                    request_text_ids=request_text_ids,
                    use_continuation=use_continuation,
                )
                # self.prompt_formatter.print_tokens(prompt_ids)
                self.clear_kv_cache()
                self.process_tokens(prompt_ids)
                sw.append(f"{len(prompt_ids)} tokens")

            totalBytes = 0
            ttfbStopwatch = Stopwatch("Time to first byte")
            ttfbStopwatch.__enter__()
            with Stopwatch("- Generating speech") as sw, Stopwatch("  - Orpheus tokens", auto_start=False) as tkSw, Stopwatch("  - SNAC decode", auto_start=False) as swSnac, Stopwatch("  - Audio processing", auto_start=False) as swBytes:
                SAMPLES_PER_FRAME = 2048
                SNAC_TOKENS_PER_FRAME = 7
                OVERLAP_FRAMES = 2
                FADE_FRAMES = 2
                OVERLAP_TOKENS = OVERLAP_FRAMES * SNAC_TOKENS_PER_FRAME
                OVERLAP_SAMPLES = OVERLAP_FRAMES * SAMPLES_PER_FRAME
                CHUNK_SIZE = SNAC_TOKENS_PER_FRAME * 5 # 100ms
                MIN_FIRST_BUFFER_FRAMES = FADE_FRAMES + OVERLAP_FRAMES
                
                token_buffer = []
                generated = []
                pending_overlap_tokens = []
                pending_overlap_audio = None

                tkSw.start()
                for token in self.generate_tokens(
                        max_tokens=max_tokens,
                        stop_token_ids=self.prompt_formatter.STOP, 
                        temperature=temperature,
                        top_p=top_p,
                        min_p=min_p,
                        repetition_penalty=repetition_penalty,
                        greedy_snac_tokens=greedy_snac_tokens,
                        force_continuation=0 if not use_continuation else 7 * 8 # 8 SNAC samples
                    ):
                    tkSw.stop()

                    if streaming:
                        token_buffer.append(token)

                        # Whenever we have enough for one "chunk + overlap"
                        if len(token_buffer) == CHUNK_SIZE:
                            # Prepend any overlap tokens from previous chunk so we can regenerate overlap frames
                            chunk_tokens = pending_overlap_tokens + token_buffer[:CHUNK_SIZE]
                            del token_buffer[:CHUNK_SIZE]  # remove consumed tokens

                            # Convert tokens to audio
                            swSnac.start()
                            hat = self.audio_model.convert_audio_tokens_to_speech(chunk_tokens)
                            swSnac.stop()

                            if hat is None:
                                continue

                            swBytes.start()
                            np_audio = self.audio_model.to_numpy_array(hat)
                            
                            if totalBytes == 0:
                                np_audio_frames_count = np_audio.shape[0] / SAMPLES_PER_FRAME
                                start_frame = self.find_start_frame_index(np_audio, samples_per_frame=SAMPLES_PER_FRAME)
                                if start_frame > 0:
                                    swBytes.append(f"(trimmed {start_frame} / {np_audio_frames_count} empty audio frames from start)")
                                    if(np_audio_frames_count - start_frame) < MIN_FIRST_BUFFER_FRAMES:
                                        start_frame = max(0, np_audio_frames_count - MIN_FIRST_BUFFER_FRAMES)
                                    np_audio = np_audio[start_frame * SAMPLES_PER_FRAME:]
                                    np_audio = self.apply_fade_in(np_audio, fade_frames=FADE_FRAMES, samples_per_frame=SAMPLES_PER_FRAME)
                                else:
                                    swBytes.append(f"(No empty audio frames found in start: {start_frame})")

                            # The actual chunk minus the final overlap (so we "keep aside" only the last frames)
                            main_chunk = np_audio[:-OVERLAP_SAMPLES] if len(np_audio) > OVERLAP_SAMPLES else np.array([], dtype=np.float32)

                            # Crossfade with pending overlap audio if it exists
                            if pending_overlap_audio is not None:
                                # Crossfade the chunk's overlap with the old overlap
                                main_chunk = self.crossfade(pending_overlap_audio, main_chunk)

                            # Yield main chunk (no overlap portion at the end)
                            if len(main_chunk) > 0:
                                bytes = self.audio_model.to_bytes(main_chunk)
                                yield bytes
                                totalBytes += len(bytes)
                                if ttfbStopwatch:
                                    ttfbStopwatch.__exit__(None, None, None)
                                    ttfbStopwatch = None

                            # Save the last overlap frames for the next chunk's crossfade
                            pending_overlap_audio = np_audio[-OVERLAP_SAMPLES:] if len(np_audio) >= OVERLAP_SAMPLES else np_audio

                            # Also keep last overlap tokens to regenerate the start of the next chunk
                            pending_overlap_tokens = chunk_tokens[-OVERLAP_TOKENS:] if len(chunk_tokens) >= OVERLAP_TOKENS else chunk_tokens

                            generated += chunk_tokens
                            swBytes.stop()
                    else:
                        generated.append(token)
                
                    tkSw.start()
                tkSw.stop()

                if streaming:
                    # Final flush if leftover tokens
                    if len(token_buffer) >= SNAC_TOKENS_PER_FRAME:
                        # Prepend overlap tokens for final generation
                        chunk_tokens = pending_overlap_tokens + token_buffer
                        swSnac.start()
                        hat = self.audio_model.convert_audio_tokens_to_speech(chunk_tokens)
                        swSnac.stop()

                        if hat is not None:
                            swBytes.start()
                            np_audio = self.audio_model.to_numpy_array(hat)

                            # Crossfade final chunk if needed
                            if pending_overlap_audio is not None:
                                main_chunk = self.crossfade(pending_overlap_audio, np_audio)
                            else:
                                main_chunk = np_audio

                            bytes = self.audio_model.to_bytes(main_chunk)
                            yield bytes
                            totalBytes += len(bytes)
                            if ttfbStopwatch:
                                ttfbStopwatch.__exit__(None, None, None)
                                ttfbStopwatch = None
                            generated += chunk_tokens
                            swBytes.stop()

            sw.append(f"{len(generated)} tokens")
            self.previous_generation_cache = (request_text_ids, generated)
        
            if not streaming:
                with Stopwatch("- SNAC decoding"):
                    hat = self.audio_model.convert_audio_tokens_to_speech(generated)

                if hat is not None:
                    with Stopwatch("- Converting speech to audio") as sw:
                        bytes = self.audio_model.to_audio_bytes(hat)
                        sw.append(f"{len(bytes) / 1024:.2f} KB")
                        yield bytes
            
            generation_duration = fullSw.duration()
            generated_audio_duration = (totalBytes / (24000 * 2))
            real_time_ratio = generated_audio_duration / generation_duration
            fullSw.append(f"({generated_audio_duration:.2f}s audio generated, {real_time_ratio:.2f}x realtime speed)")

    def crossfade(self, prev: np.ndarray, current: np.ndarray) -> np.ndarray:
        fade_len = min(len(prev), len(current))
        fade = np.linspace(0, 1, fade_len)
        cross = prev[-fade_len:] * (1 - fade) + current[:fade_len] * fade
        return np.concatenate([cross, current[fade_len:]], axis=0)

    def apply_fade_in(
        self,
        samples: np.ndarray,
        fade_frames: int,
        samples_per_frame: int = 2048
    ) -> np.ndarray:
        fade_samples = fade_frames * samples_per_frame
        fade_samples = min(fade_samples, len(samples))
        fade = np.linspace(0, 1, fade_samples, dtype=np.float32)
        samples[:fade_samples] *= fade
        return samples
        
    def find_start_frame_index(
        self,
        samples: np.ndarray,
        threshold: float = 0.01,
        window_frames: int = 1,
        lookahead_frames: int = 5,
        samples_per_frame: int = 2048
    ) -> int:
        total_frames = len(samples) // samples_per_frame
        for frame_idx in range(0, total_frames - window_frames):
            start = frame_idx * samples_per_frame
            end = start + window_frames * samples_per_frame
            window = samples[start:end]

            if np.max(np.abs(window)) > threshold:
                lookahead_start = start
                lookahead_end = start + lookahead_frames * samples_per_frame
                lookahead = samples[lookahead_start:lookahead_end]
                if np.max(np.abs(lookahead)) > threshold:
                    return max(frame_idx - lookahead_frames, 0)
        return 0
    
    def __del__(self):
        if self._sampler_chain:
            llama_cpp.llama_sampler_free(self._sampler_chain)
        if self._continuation_sampler_chain:
            llama_cpp.llama_sampler_free(self._continuation_sampler_chain)
        if self._greedy_sampler_chain:
            llama_cpp.llama_sampler_free(self._greedy_sampler_chain)
        llama_cpp.llama_free(self.ctx)
        llama_cpp.llama_free_model(self.model)

class OrpheusPromptFormatterLlamaCpp:
    def __init__(self, tts_model: OrpheusModelLlamaCpp, audio_model: SnacModel):
        self.tts_model = tts_model
        self.audio_model = audio_model

        custom_token_3_start_of_human_and_text = 128259 # SOH
        begin_of_text = 128000 # SOT
        self.START = [custom_token_3_start_of_human_and_text, begin_of_text]
        eot_id_end_of_text = 128009 # EOT
        custom_token_4_end_of_human = 128260 # EOH
        custom_token_5_start_of_ai = 128261 # SOA
        custom_token_1_start_of_speech = 128257 # SOS
        self.END = [eot_id_end_of_text, custom_token_4_end_of_human, custom_token_5_start_of_ai, custom_token_1_start_of_speech]
        custom_token_2_end_of_speech = 128258 # EOS
        custom_token_6_end_of_ai = 128262 # EOA
        self.END_OF_SPEECH = custom_token_2_end_of_speech
        self.FINAL = [custom_token_2_end_of_speech, custom_token_6_end_of_ai]

        self.SPACE = 220 # Ġ
        special_stop_token = 49158 # " rez", this was in their python example, not sure why
        self.STOP = [self.END_OF_SPEECH, special_stop_token]

        self.SEPARATOR = [2564]

    def format_prompt(self,
        voice_text_ids: Optional[list[int]],
        voice_audio_ids: Optional[list[int]],
        previous_text_ids: Optional[list[int]],
        previous_audio_ids: Optional[list[int]],
        request_text_ids: list[int],
        use_continuation: bool
    ):
        if use_continuation:
            if voice_audio_ids and previous_audio_ids:
                return self.START + voice_text_ids + self.SEPARATOR + previous_text_ids + self.SEPARATOR + request_text_ids + self.END + voice_text_ids + previous_audio_ids
            elif voice_audio_ids:
                return self.START + voice_text_ids + self.SEPARATOR + request_text_ids + self.END + voice_audio_ids
            elif previous_audio_ids:
                return self.START + previous_text_ids + self.SEPARATOR + request_text_ids + self.END + previous_audio_ids
            else:
                return self.START + request_text_ids + self.END
        else:
            if voice_audio_ids and previous_audio_ids and len(previous_audio_ids) > 40:
                return self.START + voice_text_ids + self.END + voice_audio_ids + self.FINAL + self.START + previous_text_ids + self.END + previous_audio_ids + self.FINAL + self.START + request_text_ids + self.END
            elif voice_audio_ids:
                return self.START + voice_text_ids + self.END + voice_audio_ids + self.FINAL + self.START + request_text_ids + self.END
            elif previous_audio_ids and len(previous_audio_ids) > 40:
                return self.START + previous_text_ids + self.END + previous_audio_ids + self.FINAL + self.START + request_text_ids + self.END
            else:
                return self.START + request_text_ids + self.END

    def print_tokens(self, ids: list[int]) -> list:
        """Prints the tokens in a human-readable format. For debugging purposes."""
        result = []
        snac_range = (128263, 156938)
        in_snac = False
        for t in ids:
            if snac_range[0] <= t <= snac_range[1]:
                if not in_snac:
                    result.append("SNAC")
                    in_snac = True
            else:
                str = self.tts_model.detokenize_text([t], special=True)
                result.append(f'{t}: "{str}"')
                in_snac = False
        logger.debug(f"Prompt: {', '.join(result)}")
