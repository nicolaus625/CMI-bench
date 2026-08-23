from __future__ import annotations

import base64
import os
import time

from .base import ModelAdapter, register_adapter
from .common import clipped_audio_file, normalize_generation_text


API_AUDIO_SAMPLE_RATE = int(os.environ.get("CMI_API_AUDIO_SAMPLE_RATE", "16000"))
API_MAX_RETRIES = int(os.environ.get("CMI_API_MAX_RETRIES", "3"))
API_RETRY_SLEEP = float(os.environ.get("CMI_API_RETRY_SLEEP", "2"))


def _read_b64(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def _retry(call):
    last_exc = None
    for attempt in range(API_MAX_RETRIES):
        try:
            return call()
        except Exception as exc:
            last_exc = exc
            if attempt + 1 >= API_MAX_RETRIES:
                break
            time.sleep(API_RETRY_SLEEP * (2**attempt))
    raise last_exc


@register_adapter
class OpenAIGptAudioAdapter(ModelAdapter):
    model_key = "gpt_audio"
    aliases = ("openai_gpt_audio", "gpt-audio")
    default_model_subdir = os.environ.get("OPENAI_AUDIO_MODEL", "gpt-audio-1.5")
    is_api_model = True

    def load(self):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = self.model_path
        self.audio_format = os.environ.get("OPENAI_AUDIO_INPUT_FORMAT", "wav")
        self.modalities = [
            item.strip()
            for item in os.environ.get("OPENAI_AUDIO_MODALITIES", "").split(",")
            if item.strip()
        ]
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        with clipped_audio_file(
            audio_path,
            API_AUDIO_SAMPLE_RATE,
            start,
            end,
            f"{self.model_key}_",
        ) as temp_audio:
            encoded_audio = _read_b64(temp_audio)

        request = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": encoded_audio,
                                "format": self.audio_format,
                            },
                        },
                    ],
                }
            ],
        }
        if self.modalities:
            request["modalities"] = self.modalities
        if "audio" in self.modalities:
            request["audio"] = {
                "voice": os.environ.get("OPENAI_AUDIO_VOICE", "alloy"),
                "format": os.environ.get("OPENAI_AUDIO_OUTPUT_FORMAT", "wav"),
            }

        response = _retry(lambda: self.client.chat.completions.create(**request))
        message = response.choices[0].message
        content = message.content or ""
        if isinstance(content, list):
            content = "\n".join(
                part.get("text", "") if isinstance(part, dict) else getattr(part, "text", "")
                for part in content
            )
        return normalize_generation_text(content)


class GeminiAudioAdapter(ModelAdapter):
    is_api_model = True

    def load(self):
        from google import genai

        self.client = genai.Client()
        self.model = self.model_path
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        from google.genai import types

        with clipped_audio_file(
            audio_path,
            API_AUDIO_SAMPLE_RATE,
            start,
            end,
            f"{self.model_key}_",
        ) as temp_audio:
            with open(temp_audio, "rb") as handle:
                audio_bytes = handle.read()

        response = _retry(
            lambda: self.client.models.generate_content(
                model=self.model,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav"),
                ],
            )
        )
        return normalize_generation_text(response.text or "")


@register_adapter
class Gemini31ProAdapter(GeminiAudioAdapter):
    model_key = "gemini_3_1_pro"
    aliases = ("gemini31pro", "gemini-3.1-pro", "gemini_3.1_pro")
    default_model_subdir = os.environ.get("GEMINI_3_1_PRO_MODEL", "gemini-3.1-pro-preview")


@register_adapter
class Gemini35FlashAdapter(GeminiAudioAdapter):
    model_key = "gemini_3_5_flash"
    aliases = ("gemini35flash", "gemini-3.5-flash", "gemini_3.5_flash")
    default_model_subdir = os.environ.get("GEMINI_3_5_FLASH_MODEL", "gemini-3.5-flash")
