from transformers import AutoConfig, AutoModelForCausalLM

from .configuration import AUDConfig
from .modeling import AUDForConditionalGeneration

AutoConfig.register("aud", AUDConfig)
AutoModelForCausalLM.register(AUDConfig, AUDForConditionalGeneration)

__all__ = ["AUDConfig", "AUDForConditionalGeneration", "AUDProcessor"]


def __getattr__(name):
    if name == "AUDProcessor":
        from .processing import AUDProcessor

        return AUDProcessor
    raise AttributeError(name)
