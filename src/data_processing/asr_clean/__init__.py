"""ASR-assisted transcript cleaning: re-transcribe every clip, then LLM-correct.

For every record of every training/eval manifest:

1. our deployed QASR model (vLLM, ``scripts/serve_asr_vllm.sh``) produces a
   fresh transcript of the audio (``asr_text``);
2. the corrector LLM (``localhost:8010``, served as ``corrector``) reads the
   existing label (``org_text``) and ``asr_text`` side by side and writes the
   final transcript (``text``): the words both sources support, with
   punctuation, ITN, dialect/accent preservation and (Arabic) diacritics;
3. the record is written to a uniform manifest with exactly
   ``audio_filepath, duration, text, org_text, asr_text``.

Stages (``python -m data_processing.asr_clean <cmd>``):

- ``plan``     once: byte-range chunks over every manifest, exact
               ``audio_filepath`` dedup (eval files first), frozen plan.
- ``run``      many processes on each LLM node: claim chunks dynamically,
               ASR -> LLM -> buffered, fsynced, resumable writes.
- ``status``   progress, throughput, reject reasons, ETA.
- ``assemble`` concatenate finished chunk parts into one manifest per source.

Text-only cleaning (``exhaustive``) never heard the audio; this pipeline is the
first one where every decision has acoustic evidence behind it.
"""
