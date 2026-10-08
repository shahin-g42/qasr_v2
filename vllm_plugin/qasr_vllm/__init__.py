"""vLLM plugin that serves QASR checkpoints.

QASR welds a Cohere/Parakeet Conformer encoder onto a Qwen3 decoder under the
custom ``model_type: "qasr"``, which stock vLLM cannot load. This plugin
registers the config with transformers' ``AutoConfig`` and the architecture
with vLLM's ``ModelRegistry``. vLLM calls :func:`register` in every process
(API server, engine core, workers) through the ``vllm.general_plugins`` entry
point, so it must be idempotent and cheap: the model module is registered
lazily by its import path.
"""

from __future__ import annotations


def register() -> None:
    from transformers import AutoConfig
    from vllm import ModelRegistry

    from .config import QASRVllmConfig

    AutoConfig.register("qasr", QASRVllmConfig, exist_ok=True)
    if "QASRForConditionalGeneration" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "QASRForConditionalGeneration",
            "qasr_vllm.model:QASRForConditionalGeneration",
        )
