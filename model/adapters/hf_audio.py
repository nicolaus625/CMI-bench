from __future__ import annotations

import json
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path

from .base import ModelAdapter, register_adapter
from .common import HF_PATH, clipped_audio_file, load_audio, normalize_generation_text


def _decode_new_tokens(processor, outputs, input_ids):
    if hasattr(outputs, "sequences"):
        sequences = outputs.sequences
    else:
        sequences = outputs
    new_tokens = sequences[:, input_ids.shape[1] :]
    decoded = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return normalize_generation_text(decoded[0])


def _extract_tagged_answer(text: str) -> str:
    import re

    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return normalize_generation_text(text)


def _move_batch_to_model(inputs, model):
    import torch

    moved = {}
    model_dtype = getattr(model, "dtype", None)
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
            continue
        value = value.to(model.device)
        if model_dtype is not None and torch.is_floating_point(value):
            value = value.to(model_dtype)
        moved[key] = value
    return moved


def _fake_cairosvg_if_missing():
    if "cairosvg" in sys.modules:
        return
    module = types.ModuleType("cairosvg")

    def _svg2png(*args, **kwargs):
        raise RuntimeError("SVG image conversion is unavailable in this environment.")

    module.svg2png = _svg2png
    sys.modules["cairosvg"] = module


def _relax_baichuan_clip_validation():
    try:
        from transformers.models.clip.configuration_clip import CLIPVisionConfig
    except Exception:
        return

    CLIPVisionConfig.validate_architecture = lambda self: None
    CLIPVisionConfig.validate = lambda self: None


@contextmanager
def _prepend_sys_path(path: str):
    if path in sys.path:
        yield
        return
    sys.path.insert(0, path)
    try:
        yield
    finally:
        try:
            sys.path.remove(path)
        except ValueError:
            pass


def _af2_audio_windows(audio_path: str, start: float, end: float, clap_config: dict):
    import math
    import torch

    sr = 16000
    waveform = load_audio(
        audio_path,
        target_sr=sr,
        is_mono=True,
        is_normalize=True,
        pad=False,
        start=start,
        end=end,
    ).squeeze(0)
    if waveform.numel() == 0:
        waveform = torch.zeros(1, dtype=torch.float32)

    window_length = int(float(clap_config["window_length"]) * sr)
    window_overlap = int(float(clap_config["window_overlap"]) * sr)
    max_num_window = int(clap_config["max_num_window"])
    hop = window_length - window_overlap
    total_samples = waveform.shape[-1]

    if total_samples <= window_length:
        num_windows = 1
        full_length = window_length
    elif total_samples >= max_num_window * window_length - (max_num_window - 1) * window_overlap:
        num_windows = max_num_window
        full_length = max_num_window * window_length - (max_num_window - 1) * window_overlap
    else:
        num_windows = 1 + int(math.ceil((total_samples - window_length) / float(hop)))
        full_length = num_windows * window_length - (num_windows - 1) * window_overlap

    if full_length > total_samples:
        waveform = torch.nn.functional.pad(waveform, (0, full_length - total_samples))
    else:
        waveform = waveform[:full_length]

    audio_clips = []
    for idx in range(num_windows):
        clip_start = idx * hop
        audio_clips.append(waveform[clip_start : clip_start + window_length].unsqueeze(0))

    audio_clips = torch.cat(audio_clips, dim=0)[:max_num_window]
    audio_embed_mask = torch.ones(audio_clips.shape[0], dtype=torch.float32)
    return audio_clips, audio_embed_mask


@register_adapter
class Qwen25OmniAdapter(ModelAdapter):
    model_key = "qwen2"
    aliases = ("qwen2_5_omni",)
    default_model_subdir = "Qwen2.5-Omni-7B"

    def load(self):
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            self.model_path,
            use_fast=False,
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        with clipped_audio_file(audio_path, self.sample_rate, start, end, f"{self.model_key}_") as temp_audio:
            conversation = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant."}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": temp_audio},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False,
            )
            audio = load_audio(
                temp_audio,
                target_sr=self.sample_rate,
                is_mono=True,
                is_normalize=False,
                pad=False,
            ).squeeze(0).numpy()
            inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
            inputs = _move_batch_to_model(inputs, self.model)
            outputs = self.model.generate(
                **inputs,
                return_audio=False,
                thinker_max_new_tokens=1024,
            )
            return _decode_new_tokens(self.processor, outputs, inputs["input_ids"])


@register_adapter
class AudSemThinkerAdapter(ModelAdapter):
    model_key = "audsemthinker"
    aliases = ("audsem", "audsem_thinker")
    default_model_subdir = "../audsemthinker"
    default_processor_path = str(HF_PATH / "Qwen2.5-Omni-7B")

    def load(self):
        from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

        processor_path = os.environ.get("AUDSEM_PROCESSOR_PATH", self.default_processor_path)
        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
        )
        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        audio = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            is_mono=True,
            is_normalize=False,
            pad=False,
            start=start,
            end=end,
        ).squeeze(0).numpy()
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are Qwen, a virtual human developed by the Qwen Team, "
                            "Alibaba Group, capable of perceiving auditory and visual inputs, "
                            "as well as generating text and speech."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = _move_batch_to_model(inputs, self.model)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return _extract_tagged_answer(decoded[0])


@register_adapter
class Qwen3OmniAdapter(ModelAdapter):
    model_key = "qwen3_omni"
    aliases = ("qwen3",)
    default_model_subdir = "Qwen3-Omni-30B-A3B"

    def load(self):
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.model_path)
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        if hasattr(self.model, "disable_talker"):
            self.model.disable_talker()
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        audio = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            is_mono=True,
            is_normalize=False,
            pad=False,
            start=start,
            end=end,
        ).squeeze(0).numpy()
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "local_audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, audio=[audio], return_tensors="pt", padding=True)
        inputs = _move_batch_to_model(inputs, self.model)
        outputs = self.model.generate(
            **inputs,
            return_audio=False,
            use_audio_in_video=False,
        )
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        return _decode_new_tokens(self.processor, outputs, inputs["input_ids"])


@register_adapter
class AudioReasonerAdapter(ModelAdapter):
    model_key = "audio_reasoner"
    aliases = ("audioreasoner",)
    default_model_subdir = "Audio_Reasoner-7B"

    def load(self):
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.sample_rate = self.processor.feature_extractor.sampling_rate
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        audio = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            is_mono=True,
            is_normalize=False,
            pad=False,
            start=start,
            end=end,
        ).squeeze(0).numpy()
        conversation = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful audio reasoning assistant. Answer with the final answer only.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": "local_audio.wav"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.processor(text=text, audios=[audio], return_tensors="pt", padding=True)
        inputs = _move_batch_to_model(inputs, self.model)
        outputs = self.model.generate(**inputs, max_new_tokens=512)
        return _decode_new_tokens(self.processor, outputs, inputs["input_ids"])


@register_adapter
class MossAudio8BAdapter(ModelAdapter):
    model_key = "moss_audio_8b"
    aliases = ("moss_audio", "moss8b")
    default_model_subdir = "MOSS-Audio-8B-Instruct"

    def load(self):
        from src.modeling_moss_audio import MossAudioModel
        from src.processing_moss_audio import MossAudioProcessor

        self.model = MossAudioModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            dtype=self.torch_dtype,
            device_map=self.device_map,
        ).eval()
        self.processor = MossAudioProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            enable_time_marker=True,
        )
        self.sample_rate = self.processor.config.mel_sr
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        import torch

        audio = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            is_mono=True,
            is_normalize=False,
            pad=False,
            start=start,
            end=end,
        ).squeeze(0).numpy()
        inputs = self.processor(text=prompt, audios=[audio], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        if inputs.get("audio_data") is not None:
            inputs["audio_data"] = inputs["audio_data"].to(self.model.dtype)
        inputs["audio_input_mask"] = inputs["input_ids"] == self.processor.audio_token_id

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                use_cache=True,
            )

        new_tokens = outputs[0, inputs["input_ids"].shape[1] :]
        return normalize_generation_text(
            self.processor.decode(new_tokens, skip_special_tokens=True)
        )


@register_adapter
class AudioFlamingo3Adapter(ModelAdapter):
    model_key = "audio_flamingo3"
    aliases = ("af3",)
    default_model_subdir = "Audio_Flamingo_3-7B"

    def load(self):
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.sample_rate = 16000
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        with clipped_audio_file(audio_path, self.sample_rate, start, end, f"{self.model_key}_") as temp_audio:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": temp_audio},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = _move_batch_to_model(inputs, self.model)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            return _decode_new_tokens(self.processor, outputs, inputs["input_ids"])


@register_adapter
class MusicFlamingoAdapter(ModelAdapter):
    model_key = "music_flamingo"
    aliases = ("mf",)
    default_model_subdir = "Music_Flamingo-7B"

    def load(self):
        from transformers import AutoProcessor, MusicFlamingoForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.sample_rate = 16000
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        with clipped_audio_file(audio_path, self.sample_rate, start, end, f"{self.model_key}_") as temp_audio:
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": temp_audio},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            inputs = _move_batch_to_model(inputs, self.model)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            return _decode_new_tokens(self.processor, outputs, inputs["input_ids"])


@register_adapter
class BaichuanOmniAdapter(ModelAdapter):
    model_key = "baichuan_omni"
    aliases = ("baichuan",)
    default_model_subdir = "Baichuan-Omni-11B"

    def load(self):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        _fake_cairosvg_if_missing()
        _relax_baichuan_clip_validation()
        if self.model_path not in sys.path:
            sys.path.insert(0, self.model_path)
        from processor_omni import OmniMMProcessor

        self.config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.processor = OmniMMProcessor(self.tokenizer, self.config, training=False, relative_path=None)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            trust_remote_code=True,
        ).eval()
        self.sample_rate = self.config.audio_config.sampling_rate
        return self

    def _format_prompt(self, prompt: str, audio_path: str) -> str:
        audio_info = json.dumps({"path": audio_path}, ensure_ascii=False)
        return (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{self.processor.audio_start_tag}{audio_info}{self.processor.audio_end_tag}\n"
            f"{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        with clipped_audio_file(audio_path, self.sample_rate, start, end, f"{self.model_key}_") as temp_audio:
            processed = self.processor([self._format_prompt(prompt, temp_audio)], parallel=1)
            inputs = {
                key: value
                for key, value in processed.items()
                if value is not None and key in {
                    "input_ids",
                    "attention_mask",
                    "audios",
                    "encoder_length",
                    "bridge_length",
                    "images",
                    "images_grid",
                    "patch_nums",
                    "videos",
                    "videos_grid",
                    "videos_patch_nums",
                }
            }
            inputs = _move_batch_to_model(inputs, self.model)
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            new_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            return normalize_generation_text(decoded[0])


@register_adapter
class AudioFlamingo2Adapter(ModelAdapter):
    model_key = "audio_flamingo2"
    aliases = ("af2",)
    default_model_subdir = "Audio_Flamingo_2-7B"
    source_dir = str(Path.home() / "audio-flamingo" / "inference_HF_pretrained")
    qwen_paths_by_hidden_size = {
        1536: str(HF_PATH / "Qwen2.5-1.5B"),
        2048: str(HF_PATH / "Qwen2.5-3B"),
    }

    def _checkpoint_lang_hidden_size(self, safe_ckpt_dir: Path) -> int:
        from safetensors import safe_open

        lang_embed_key = "lang_encoder.model.embed_tokens.weight"
        for chunk_path in sorted(safe_ckpt_dir.glob("*.safetensors")):
            with safe_open(str(chunk_path), framework="pt", device="cpu") as handle:
                if lang_embed_key in handle.keys():
                    return int(handle.get_slice(lang_embed_key).get_shape()[1])

        raise KeyError(f"Could not find {lang_embed_key!r} in Audio Flamingo 2 checkpoint chunks")

    def _resolve_qwen_path(self, safe_ckpt_dir: Path) -> Path:
        override = os.environ.get("AUDIO_FLAMINGO2_QWEN_PATH")
        if override:
            return Path(override)

        hidden_size = self._checkpoint_lang_hidden_size(safe_ckpt_dir)
        qwen_path = self.qwen_paths_by_hidden_size.get(hidden_size)
        if qwen_path is None:
            raise ValueError(
                "Unsupported Audio Flamingo 2 language hidden size "
                f"{hidden_size}. Set AUDIO_FLAMINGO2_QWEN_PATH to the matching Qwen checkpoint."
            )
        return Path(qwen_path)

    def load(self):
        import torch
        import yaml
        from safetensors.torch import load_file

        if not torch.cuda.is_available():
            raise RuntimeError("Audio Flamingo 2 requires a CUDA GPU. Run this adapter inside a GPU job.")

        source_dir = Path(self.source_dir)
        model_dir = Path(self.model_path)
        config_path = source_dir / "configs" / "inference.yaml"
        safe_ckpt_dir = model_dir / "safe_ckpt"
        metadata_path = safe_ckpt_dir / "metadata.json"
        clap_ckpt = model_dir / "clap_ckpt" / "epoch_16.pt"
        qwen_path = self._resolve_qwen_path(safe_ckpt_dir)

        for required_path in (source_dir, config_path, metadata_path, clap_ckpt, qwen_path):
            if not required_path.exists():
                raise FileNotFoundError(f"Audio Flamingo 2 required path not found: {required_path}")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        with open(config_path, "r", encoding="utf-8") as handle:
            config = yaml.load(handle, Loader=yaml.FullLoader)

        self.clap_config = config["clap_config"]
        self.clap_config["checkpoint"] = str(clap_ckpt)
        self.model_config = config["model_config"]
        self.model_config["cache_dir"] = str(model_dir / ".cache")
        self.model_config["lang_encoder_path"] = str(qwen_path)
        self.model_config["tokenizer_path"] = str(qwen_path)

        train_config = config["train_config"]
        train_config["offline"] = True
        train_config["gradient_checkpointing"] = False
        train_config["freeze_lm_embeddings"] = False

        with _prepend_sys_path(str(source_dir)):
            from src.factory import create_model_and_transforms
            from utils import Dict2Class, get_autocast, get_cast_dtype

            args = Dict2Class(train_config)
            self.model, self.tokenizer = create_model_and_transforms(
                **self.model_config,
                clap_config=self.clap_config,
                use_local_files=args.offline,
                gradient_checkpointing=args.gradient_checkpointing,
                freeze_lm_embeddings=args.freeze_lm_embeddings,
            )

        self.model = self.model.to(0).eval()

        with open(metadata_path, "r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        state_dict = {}
        for chunk_name in metadata:
            chunk_path = safe_ckpt_dir / f"{chunk_name}.safetensors"
            if not chunk_path.exists():
                raise FileNotFoundError(f"Audio Flamingo 2 checkpoint chunk not found: {chunk_path}")
            state_dict.update(load_file(str(chunk_path)))

        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        print(
            "Audio Flamingo 2 load report: "
            f"{len(missing_keys)} missing keys, {len(unexpected_keys)} unexpected keys"
        )
        precision = train_config.get("precision", "amp_bf16")
        self.autocast = get_autocast(precision, cache_enabled=True)
        self.cast_dtype = get_cast_dtype(precision)
        self.sample_rate = 16000
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        import torch

        audio_clips, audio_embed_mask = _af2_audio_windows(audio_path, start, end, self.clap_config)
        device = next(self.model.parameters()).device
        if self.cast_dtype is not None:
            audio_clips = audio_clips.to(dtype=self.cast_dtype)
            audio_embed_mask = audio_embed_mask.to(dtype=self.cast_dtype)
        audio_clips = audio_clips.to(device, non_blocking=True)
        audio_embed_mask = audio_embed_mask.to(device, non_blocking=True)

        text_prompt = str(prompt).lower().strip()
        sample = f"<audio>{text_prompt}{self.tokenizer.sep_token}"
        text = self.tokenizer(
            sample,
            max_length=512,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        input_ids = text["input_ids"].to(device, non_blocking=True)
        attention_mask = text["attention_mask"].to(device, non_blocking=True)

        with torch.no_grad(), self.autocast():
            output = self.model.generate(
                audio_x=audio_clips.unsqueeze(0),
                audio_x_mask=audio_embed_mask.unsqueeze(0),
                lang_x=input_ids,
                attention_mask=attention_mask,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                top_k=30,
                top_p=0.95,
                num_return_sequences=1,
            )[0]

        decoded = self.tokenizer.decode(output)
        decoded = decoded.split(self.tokenizer.sep_token)[-1]
        decoded = decoded.replace(self.tokenizer.eos_token or "", "")
        decoded = decoded.replace(self.tokenizer.pad_token or "", "")
        decoded = decoded.replace("<|endofchunk|>", "")
        return normalize_generation_text(decoded)


@register_adapter
class MellowAdapter(ModelAdapter):
    model_key = "mellow"
    aliases = ("mellow_v0",)
    default_model_subdir = "../mellow"

    def _device_index(self):
        import torch

        if not torch.cuda.is_available():
            return "cpu", False
        if isinstance(self.device_map, str) and self.device_map.startswith("cuda:"):
            return int(self.device_map.split(":", 1)[1]), True
        if isinstance(self.device_map, str) and self.device_map.isdigit():
            return int(self.device_map), True
        return 0, True

    def _local_hf_download(self, repo_id, filename, *args, **kwargs):
        local_path = self.model_dir / filename
        if not local_path.exists():
            raise FileNotFoundError(f"Mellow required file not found: {local_path}")
        return str(local_path)

    def load(self):
        import random

        import numpy as np
        import torch

        self.model_dir = Path(self.model_path).resolve()
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Mellow model directory not found: {self.model_dir}")

        for required_name in ("v0.ckpt", "v0_s.ckpt", "config.json", "mellow/config/v0.yaml"):
            required_path = self.model_dir / required_name
            if not required_path.exists():
                raise FileNotFoundError(f"Mellow required file not found: {required_path}")

        hub_cache = str(self.model_dir / ".cache" / "huggingface")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["HF_HOME"] = hub_cache
        os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
        os.environ["TRANSFORMERS_CACHE"] = hub_cache
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        device, use_cuda = self._device_index()
        with _prepend_sys_path(str(self.model_dir)):
            import mellow.wrapper as mellow_wrapper

            # Mellow's wrapper is written for HF Hub checkpoints. This repo already
            # contains the checkpoint files, so keep inference fully local.
            mellow_wrapper.hf_hub_download = self._local_hf_download
            decoder_path = Path(os.environ.get("MELLOW_DECODER_PATH", HF_PATH / "SmolLM2"))
            if not decoder_path.exists():
                decoder_snapshots = sorted(
                    (self.model_dir / ".cache" / "huggingface" / "models--HuggingFaceTB--SmolLM2-135M" / "snapshots").glob("*")
                )
                decoder_path = decoder_snapshots[-1] if decoder_snapshots else decoder_path
            if not decoder_path.exists():
                raise FileNotFoundError(
                    "Mellow requires the HuggingFaceTB/SmolLM2-135M decoder. "
                    f"Set MELLOW_DECODER_PATH or place it at {HF_PATH / 'SmolLM2'}."
                )
            decoder_path = str(decoder_path)
            original_read_config = mellow_wrapper.MellowWrapper.read_config_as_args

            def _read_config_with_local_decoder(wrapper_self, config_path):
                args = original_read_config(wrapper_self, config_path)
                args.model["decoder"]["text_decoder"] = decoder_path
                args.data["tokenizer_type"] = decoder_path
                return args

            mellow_wrapper.MellowWrapper.read_config_as_args = _read_config_with_local_decoder
            from mellow import MellowWrapper

            self.wrapper = MellowWrapper(
                config="v0",
                model=os.environ.get("MELLOW_MODEL_VERSION", "v0"),
                device=device,
                use_cuda=use_cuda,
            )
            if not hasattr(self.wrapper.tokenizer, "encode_plus"):
                tokenizer = self.wrapper.tokenizer

                def _encode_plus_compat(
                    text=None,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=None,
                    pad_to_max_length=False,
                    return_tensors=None,
                    **kwargs,
                ):
                    return tokenizer(
                        text,
                        add_special_tokens=add_special_tokens,
                        truncation=truncation,
                        max_length=max_length,
                        padding="max_length" if pad_to_max_length else False,
                        return_tensors=return_tensors,
                        **kwargs,
                    )

                self.wrapper.tokenizer.encode_plus = _encode_plus_compat
            self.wrapper.model = self.wrapper.model.float()

        self.sample_rate = int(self.wrapper.args.data["sampling_rate"])
        self.segment_seconds = float(self.wrapper.args.data["segment_seconds"])
        self.max_len = max(2, int(os.environ.get("MELLOW_MAX_LEN", "300")))
        self.top_p = float(os.environ.get("MELLOW_TOP_P", "0.8"))
        self.temperature = float(os.environ.get("MELLOW_TEMPERATURE", "1.0"))
        return self

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        clip_end = end
        if end == -1 or end - start > self.segment_seconds:
            clip_end = start + self.segment_seconds

        with clipped_audio_file(audio_path, self.sample_rate, start, clip_end, f"{self.model_key}_") as temp_audio:
            outputs = self.wrapper.generate(
                examples=[[temp_audio, temp_audio, prompt]],
                max_len=self.max_len,
                top_p=self.top_p,
                temperature=self.temperature,
            )

        if isinstance(outputs, list):
            return normalize_generation_text(str(outputs[0] if outputs else ""))
        return normalize_generation_text(str(outputs))


@register_adapter
class OpenOmniAdapter(ModelAdapter):
    model_key = "openomni"
    aliases = ("open_omni",)
    default_model_subdir = "OpenOmni/qwen2"
    source_dir = str(Path.home() / "OpenOmni")

    def load(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("OpenOmni requires a CUDA GPU. Run this adapter inside a GPU job.")

        source_dir = Path(os.environ.get("OPENOMNI_SOURCE_DIR", self.source_dir)).resolve()
        model_path = Path(self.model_path).resolve()
        if (model_path / "qwen2" / "config.json").exists():
            model_path = model_path / "qwen2"

        if not source_dir.exists():
            raise FileNotFoundError(f"OpenOmni source directory not found: {source_dir}")
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(f"OpenOmni model config not found under: {model_path}")

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        with _prepend_sys_path(str(source_dir)):
            import builtins
            from openomni.constants import (
                DEFAULT_SPEECH_TOKEN,
                IMAGE_TOKEN_INDEX,
                SPEECH_TOKEN_INDEX,
            )

            # OpenOmni's mm_utils.py evaluates these names as default arguments
            # without importing them. Provide them via builtins instead of
            # patching the external source checkout.
            builtins.DEFAULT_SPEECH_TOKEN = DEFAULT_SPEECH_TOKEN
            builtins.SPEECH_TOKEN_INDEX = SPEECH_TOKEN_INDEX
            import transformers.generation.utils as generation_utils
            from transformers.utils import is_hqq_available, is_torchdynamo_compiling

            if not hasattr(generation_utils, "is_torchdynamo_compiling"):
                generation_utils.is_torchdynamo_compiling = is_torchdynamo_compiling
            if not hasattr(generation_utils, "NEED_SETUP_CACHE_CLASSES_MAPPING"):
                generation_utils.NEED_SETUP_CACHE_CLASSES_MAPPING = {}
            if not hasattr(generation_utils, "QUANT_BACKEND_CLASSES_MAPPING"):
                generation_utils.QUANT_BACKEND_CLASSES_MAPPING = {}
            if not hasattr(generation_utils, "is_hqq_available"):
                generation_utils.is_hqq_available = is_hqq_available
            if not hasattr(generation_utils, "is_quanto_available"):
                generation_utils.is_quanto_available = lambda: False
            if not hasattr(generation_utils, "QuantizedCacheConfig"):
                class QuantizedCacheConfig:
                    backend = "quanto"

                generation_utils.QuantizedCacheConfig = QuantizedCacheConfig
            from openomni.model import LlavaHerQwen2ForCausalLM
            from openomni.model.speech_encoder.builder import build_speech_encoder
            from openomni.utils import disable_torch_init
            from transformers import AutoConfig, AutoTokenizer

            disable_torch_init()
            config = AutoConfig.from_pretrained(str(model_path))
            if hasattr(config, "mm_vision_tower"):
                delattr(config, "mm_vision_tower")
            if hasattr(config, "vision_tower"):
                delattr(config, "vision_tower")
            if hasattr(config, "speech_generator_type"):
                delattr(config, "speech_generator_type")
            default_speech_encoder = str(HF_PATH / "whisper" / "large-v3.pt")
            if not os.path.exists(default_speech_encoder):
                default_speech_encoder = "large-v3"
            config.speech_encoder = os.environ.get("OPENOMNI_SPEECH_ENCODER", default_speech_encoder)
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=False)
            self.model = LlavaHerQwen2ForCausalLM.from_pretrained(
                str(model_path),
                config=config,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                device_map=self.device_map,
            )
            self.model.get_model().speech_encoder = build_speech_encoder(self.model.config)
            self.model.get_model().speech_encoder.to(device="cuda", dtype=torch.float16)

        self.IMAGE_TOKEN_INDEX = IMAGE_TOKEN_INDEX
        self.SPEECH_TOKEN_INDEX = SPEECH_TOKEN_INDEX
        added_tokens = self.tokenizer.add_tokens(["<image>", "<speech>"], special_tokens=True)
        if added_tokens:
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        )
        self.image_token_index = self.tokenizer.convert_tokens_to_ids("<image>")
        self.speech_token_index = self.tokenizer.convert_tokens_to_ids("<speech>")
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.sample_rate = 16000
        self.mel_size = int(os.environ.get("OPENOMNI_MEL_SIZE", "128"))
        self.max_new_tokens = int(os.environ.get("OPENOMNI_MAX_NEW_TOKENS", "512"))
        self.system_message = (
            "You are a helpful language, vision and speech assistant. "
            "Answer the user's music-audio instruction directly and concisely."
        )
        return self

    def _encode_prompt(self, prompt: str):
        import torch

        input_ids = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": f"<speech>\n{prompt}"},
            ],
            add_generation_prompt=True,
        )
        input_ids = [
            self.IMAGE_TOKEN_INDEX
            if token_id == self.image_token_index
            else self.SPEECH_TOKEN_INDEX
            if token_id == self.speech_token_index
            else token_id
            for token_id in input_ids
        ]
        return torch.tensor([input_ids], dtype=torch.long, device="cuda")

    def _speech_tensor(self, audio_path: str):
        import torch
        import whisper

        speech = whisper.load_audio(audio_path)
        speech = whisper.pad_or_trim(speech)
        speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        speech_lengths = torch.LongTensor([speech.shape[0]]).to(device="cuda", non_blocking=True)
        speech = speech.to(dtype=torch.float16, device="cuda", non_blocking=True).unsqueeze(0)
        return speech, speech_lengths

    def _speech_inputs_embeds(self, input_ids, speech, speech_lengths):
        import torch

        if input_ids.shape[0] != 1:
            raise ValueError("OpenOmniAdapter currently supports batch size 1")
        speech_positions = torch.where(input_ids[0] == self.SPEECH_TOKEN_INDEX)[0]
        if speech_positions.numel() != 1:
            raise ValueError("OpenOmniAdapter expects exactly one <speech> token in the prompt")

        speech_feature = self.model.encode_speech(speech, speech_lengths)[0].to(self.device)
        token_ids = input_ids[0].to(self.device)
        speech_pos = int(speech_positions[0].item())
        before = self.model.get_model().embed_tokens(token_ids[:speech_pos])
        after = self.model.get_model().embed_tokens(token_ids[speech_pos + 1 :])
        inputs_embeds = torch.cat([before, speech_feature, after], dim=0).unsqueeze(0)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        return inputs_embeds, attention_mask

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        import torch

        with clipped_audio_file(audio_path, self.sample_rate, start, end, f"{self.model_key}_") as temp_audio:
            input_ids = self._encode_prompt(prompt)
            speech, speech_lengths = self._speech_tensor(temp_audio)
            inputs_embeds, attention_mask = self._speech_inputs_embeds(input_ids, speech, speech_lengths)

            with torch.inference_mode():
                outputs = super(self.model.__class__, self.model).generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=self.max_new_tokens,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        output_ids = outputs[0] if isinstance(outputs, tuple) else outputs
        decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        return normalize_generation_text(decoded)
