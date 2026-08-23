from .base import MODEL_ADAPTERS, ModelAdapter, get_adapter_class

# Import adapter modules so they self-register.
from . import api_audio  # noqa: F401
from . import hf_audio  # noqa: F401

__all__ = ["MODEL_ADAPTERS", "ModelAdapter", "get_adapter_class"]
