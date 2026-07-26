__version__ = "0.2.0"

from transformers import AutoConfig, AutoFeatureExtractor, AutoModelForMultimodalLM, AutoProcessor

from .configuration import CohereEncoderConfig, QASRConfig
from .feature_extraction import QASRFeatureExtractor
from .modeling import QASRForConditionalGeneration, QASRModel, QASRMultiModalProjector
from .processing import QASRProcessor
from .eagle import EagleConfig, EagleHead, EagleSpeculativeDecoder


# Register QASR classes with the Auto* factories so that
# ``from_pretrained`` on a QASR checkpoint resolves the correct sub-processor
# without relying on the deprecated ``feature_extractor_class`` attribute
# (which is removed in transformers >= 5.14).  Registrations are idempotent
# so importing this package twice is safe.
try:
    AutoConfig.register("qasr", QASRConfig, exist_ok=True)
except (TypeError, ValueError):
    pass
try:
    AutoFeatureExtractor.register(QASRConfig, QASRFeatureExtractor, exist_ok=True)
except (TypeError, ValueError):
    pass
try:
    AutoProcessor.register(QASRConfig, QASRProcessor, exist_ok=True)
except (TypeError, ValueError):
    pass
try:
    AutoModelForMultimodalLM.register(QASRConfig, QASRForConditionalGeneration, exist_ok=True)
except (TypeError, ValueError):
    pass

__all__ = [
    "CohereEncoderConfig",
    "QASRConfig",
    "QASRFeatureExtractor",
    "QASRForConditionalGeneration",
    "QASRModel",
    "QASRMultiModalProjector",
    "QASRProcessor",
    "EagleConfig",
    "EagleHead",
    "EagleSpeculativeDecoder",
]
