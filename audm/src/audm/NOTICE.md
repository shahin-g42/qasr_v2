# AUD attribution and changes

AUD combines pretrained components; renaming public classes and tensor prefixes
does not change their origin or licensing.

- AudPro preserves the local QASR Cohere/Parakeet Conformer, Qwen text model,
  learned bridge, optional pooling and optional CTC scaffold.
- AudTurbo imports the complete Qwen3-Omni Captioner audio tower, including its
  pretrained output projections, and the local trained QASR text decoder.
- AudFlash defaults to the Gemma 4 audio tower and frontend, a compatible trained
  GQASR bridge, and the trained GQASR decoder. Default temporal stride is 1;
  re-alignment training is required. Gemma source terms continue to apply.
- Legacy AudFlash imports the Moonshine Streaming Medium encoder and the local trained
  GQASR text decoder. Its new adapter replaces the original Moonshine decoder's
  interface. Moonshine's text decoder and positional adapter are not imported.
- Transformer primitives are provided by Hugging Face Transformers and their
  original authors. AUD instance subclasses retain these implementations.
- The Cohere frontend, chat processing and length helpers preserve code from
  this repository's qasr package. Training reuses QASR data, sampling,
  augmentation, code-switching, loss and evaluation modules.
- AUD changes Moonshine's causal convolution **padding mask** to retain only
  valid output time positions. This prevents padded batch tails from adding
  audio tokens or changing look-ahead attention. Convolution weights and
  valid-frame convolution mathematics are unchanged.
- Turbo and legacy Moonshine Flash add newly initialized adapters. Their exported checkpoints
  are starting points for alignment training, not claims of pretrained ASR quality.
- Legacy Moonshine Flash's training spectral augmentation masks waveform STFT bins and
  reconstructs audio. It is an AUD adaptation, not a claim about Moonshine's
  original training recipe or native streaming runtime.

Exports record immutable Hub revisions, source file hashes and tensor mappings
in provenance.json. Available upstream LICENSE/NOTICE files and model cards are
copied into licenses/. Source terms continue to apply; AUD does not relicense
pretrained weights. Local source checkpoints must retain their own notices.
