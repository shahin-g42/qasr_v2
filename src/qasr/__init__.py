__version__ = "0.2.0"

from contextlib import suppress

from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForMultimodalLM, AutoProcessor

from .configuration import CohereEncoderConfig, QASRConfig
from .eagle import EagleConfig, EagleHead, EagleSpeculativeDecoder
from .feature_extraction import QASRFeatureExtractor
from .modeling import QASRForConditionalGeneration, QASRModel, QASRMultiModalProjector
from .processing import QASRProcessor

# Register QASR classes with the Auto* factories so that
# ``from_pretrained`` on a QASR checkpoint resolves the correct sub-processor
# without relying on the deprecated ``feature_extractor_class`` attribute
# (which is removed in transformers >= 5.14).  Registrations are idempotent
# so importing this package twice is safe.
with suppress(TypeError, ValueError):
    AutoConfig.register("qasr", QASRConfig, exist_ok=True)
with suppress(TypeError, ValueError):
    AutoFeatureExtractor.register(QASRConfig, QASRFeatureExtractor, exist_ok=True)
with suppress(TypeError, ValueError):
    AutoProcessor.register(QASRConfig, QASRProcessor, exist_ok=True)
with suppress(TypeError, ValueError):
    AutoModelForMultimodalLM.register(QASRConfig, QASRForConditionalGeneration, exist_ok=True)

__all__ = [
    "CohereEncoderConfig",
    "EagleConfig",
    "EagleHead",
    "EagleSpeculativeDecoder",
    "QASRConfig",
    "QASRFeatureExtractor",
    "QASRForConditionalGeneration",
    "QASRModel",
    "QASRMultiModalProjector",
    "QASRProcessor",
]
