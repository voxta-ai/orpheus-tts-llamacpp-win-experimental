import os
import io
import statistics
import argparse
import time
import logging
import wave
import winsound
from orpheus_engine import create_orpheus_engine

logging.getLogger().setLevel(logging.INFO)

class OrpheusTest:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-path", required=True)
        parser.add_argument("--snac-path", required=True)
        parser.add_argument("--output-path", required=True)
        parser.add_argument("--warmup-runs", default=1, type=int)
        parser.add_argument("--runs", default=5, type=int)
        parser.add_argument("--play-test", default=False, action="store_true")
        parser.add_argument("--text", default="Hello <laugh>, my name is uhm, Orpheus. I'm saying something a bit long so we reach the total amount of tokens <cough> and we're sure to be cut off.")
        parser.add_argument("--max_tokens", default=294, type=int)
        parser.add_argument("--voice", default="tara")
        parser.add_argument("--voice-wav", default=None, type=str)
        parser.add_argument("--voice-transcript", default=None, type=str)
        args = parser.parse_args()

        if args.max_tokens % 7 != 0:
            raise ValueError("max_tokens must be a multiple of 7")

        self.engine = create_orpheus_engine(
            model_path=args.model_path,
            snac_path=args.snac_path,
        )

        self.output_path = args.output_path
        self.warmup_runs = args.warmup_runs
        self.runs = args.runs
        self.play_test = args.play_test

        self.text = args.text
        self.max_tokens = args.max_tokens
        self.voice = args.voice
        self.voice_wav = args.voice_wav
        self.voice_transcript = args.voice_transcript

        self.model_filename = os.path.basename(args.model_path)

        os.makedirs(self.output_path, exist_ok=True)

    def run(self):
        id = 1

        for i in range(self.warmup_runs):
            chunks = self.run_one()
            if i == 0:
                self.save_audio(chunks, 0)

        timings = []    
        for i in range(self.runs):
            id += 1
            st = time.monotonic()
            chunks = self.run_one()
            timings.append(time.monotonic() - st)
            self.save_audio(chunks, i)
        
        # Calculate statistics
        mean_time = statistics.mean(timings)
        median_time = statistics.median(timings)
        min_time = min(timings)
        max_time = max(timings)
        logging.info(f"Mean time: {mean_time:.3f} seconds")
        logging.info(f"Median time: {median_time:.3f} seconds")
        logging.info(f"Min time: {min_time:.3f} seconds")
        logging.info(f"Max time: {max_time:.3f} seconds")
    
    def run_one(self):
        return self.engine.generate_speech(
            finetune_voice=self.voice,
            greedy_snac_tokens=3,
            max_tokens=self.max_tokens,
            prompt=self.text,
            voice_transcript=self.voice_transcript,
            voice_path=self.voice_wav,
            use_continuation=False,
            min_p=0,
            top_p=0.9,
            temperature=0.8,
            repetition_penalty=1.2,
            previous_text=None,
            reply_to=None,
        )
        
    def save_audio(self, chunks, i):
        tmp_file = os.path.join(self.output_path, self.model_filename + "-" + str(i) + ".wav")
        try:
            os.remove(tmp_file)
        except FileNotFoundError:
            pass

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)

            for audio_chunk in chunks:
                wf.writeframes(audio_chunk)

        with open(tmp_file, "wb") as f:
            f.write(wav_buffer.getvalue())
        if self.play_test:
            winsound.PlaySound(tmp_file, winsound.SND_FILENAME)  

if __name__ == "__main__":
    OrpheusTest().run()