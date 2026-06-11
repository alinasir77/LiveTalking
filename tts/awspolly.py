import boto3
import time
import numpy as np
import soundfile as sf
from io import BytesIO

from utils.logger import logger

from registry import register
from .base_tts import BaseTTS, State


@register("tts", "awspolly")
class AwsPollyTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.opt = opt
        self.parent = parent

        self.sample_rate = 16000
        self.chunk_size = parent.chunk if parent else 320

        self.voice_id = getattr(opt, "POLLY_VOICE_ID", "Joanna")
        self.engine = "neural"

        self.client = boto3.client(
            "polly",
            region_name=getattr(opt, "region", "us-east-1")
        )

    def put_msg_txt(self, msg, datainfo=None):
        """
        Convert text → speech → push audio frames into avatar pipeline
        """
        try:
            start = time.perf_counter()

            response = self.client.synthesize_speech(
                Text=msg,
                OutputFormat="pcm",   # IMPORTANT: raw PCM for real-time avatar
                VoiceId=self.voice_id,
                Engine=self.engine,
                SampleRate=str(self.sample_rate)
            )

            audio_stream = response["AudioStream"].read()

            audio_array = np.frombuffer(audio_stream, dtype=np.int16).astype(np.float32) / 32768.0

            # chunk and push into avatar pipeline
            idx = 0
            total = len(audio_array)

            while idx < total:
                chunk = audio_array[idx:idx + self.chunk_size]

                if hasattr(self.parent, "asr") and self.parent.asr:
                    self.parent.asr.put_audio_frame(chunk, datainfo or {})

                idx += self.chunk_size

            end = time.perf_counter()
            logger.info(f"[PollyTTS] Generated speech in {end - start:.3f}s")

        except Exception as e:
            logger.exception(f"Polly TTS error: {e}")

    def stop_tts(self):
        pass

    def flush_talk(self):
        pass