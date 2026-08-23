from __future__ import annotations

from typing import Dict, Type


MODEL_ADAPTERS: Dict[str, Type["ModelAdapter"]] = {}


class ModelAdapter:
    model_key = ""
    aliases = ()
    default_model_subdir = ""
    is_api_model = False

    def __init__(self, model_path: str, device_map: str = "auto", torch_dtype: str = "auto"):
        self.model_path = model_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.model = None
        self.processor = None

    def load(self):
        raise NotImplementedError

    def predict(self, prompt: str, audio_path: str, start: float, end: float) -> str:
        raise NotImplementedError


def get_adapter_class(model_name: str) -> Type[ModelAdapter]:
    try:
        return MODEL_ADAPTERS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_ADAPTERS))
        raise KeyError(f"Unsupported model '{model_name}'. Available: {available}") from exc


def register_adapter(cls: Type[ModelAdapter]) -> Type[ModelAdapter]:
    if not cls.model_key:
        raise ValueError(f"{cls.__name__} must define model_key")
    names = (cls.model_key, *cls.aliases)
    for name in names:
        MODEL_ADAPTERS[name] = cls
    return cls
