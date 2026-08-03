"""Distributed multilingual transcript processing pipeline.

Cleans, ITN-normalizes, and punctuates ASR manifests using LLM-powered
processing via vLLM. Arabic gets specialized treatment (dialect
preservation, critical diacritics restoration, Arabic ITN); all other
languages (en, zh, hi, ml, ...) share the generic language-aware agents.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
