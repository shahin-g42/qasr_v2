"""Tiny CPU training smoke and version stamp for the separate training environment."""
import json
import sys
import tempfile
from pathlib import Path


def runtime_versions():
    import torch
    import transformers
    return {"torch": str(torch.__version__), "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda}


def check_training_profile():
    path = Path(sys.prefix) / 'gqasr-training-profile.json'
    versions = runtime_versions()
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get('passed') is not True or saved.get('runtime') != versions:
            raise RuntimeError('Training environment changed since validation; rerun python -m gqasr.env_check --write-profile')
    elif versions['torch'].split('+')[0] != '2.11.0' or versions['transformers'] != '5.19.0':
        raise RuntimeError('Unvalidated environment; use setup_training_env2.sh for the new training stack')
    return versions


def validate():
    import numpy as np
    import soundfile as sf
    import torch
    import wandb
    from .modeling import GQASRAcousticConfig, GQASRConfig, GQASRDecoderConfig, GQASRForSpeechRecognition
    from .processing import GQASRFeatureExtractor
    from .optimization import build_optimizer
    from .data import Collator
    from .tracking import Tracker
    from .augmentation import build_training_transforms
    torch.set_num_threads(1)
    torch.manual_seed(7)
    audio = GQASRAcousticConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        output_proj_dims=48, subsampling_conv_channels=[128, 4], attention_chunk_size=4,
        attention_context_left=5)
    decoder = GQASRDecoderConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=512, tie_word_embeddings=True)
    model = GQASRForSpeechRecognition(GQASRConfig(audio.to_dict(), decoder.to_dict(), [4], [5], 1, 16))
    model.configure_training(2, 2, True, True, 0.1, train_acoustic_frontend=False)
    wave = (np.sin(np.arange(4800) / 10) * .1).astype('float32')
    features = GQASRFeatureExtractor()(
        [wave], sampling_rate=16000, padding=True, truncation=False,
        max_length=None, return_tensors='pt',
    )
    batch = dict(input_features=features['input_features'],
                 input_features_mask=features['input_features_mask'].bool(),
                 targets=torch.tensor([[4, 3, 1]]), target_mask=torch.ones(1, 3, dtype=torch.bool))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
    model.train()
    loss = model(**batch)['loss']
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    if not torch.isfinite(loss) or any(g is None or not torch.isfinite(g).all() for g in gradients):
        raise RuntimeError('Nonfinite loss/gradient or disconnected trainable parameters')
    opt.step()
    with tempfile.TemporaryDirectory(prefix='gqasr-env-check-') as tmp:
        sf.write(Path(tmp) / 'audio.wav', wave, 16000)
        sf.read(Path(tmp) / 'audio.wav')
        model.save_pretrained(Path(tmp) / 'model')
        restored = GQASRForSpeechRecognition.from_pretrained(Path(tmp) / 'model')
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    return {"passed": True, "runtime": runtime_versions(), "python": sys.version,
            "kind": "tiny CPU frontend/backward/AdamW/save-reload check; not a CUDA or cluster health test",
            "loss": float(loss.detach())}


def main():
    result = validate()
    if '--write-profile' in sys.argv:
        if sys.prefix == sys.base_prefix:
            raise RuntimeError('Refusing to stamp a non-virtual environment')
        path = Path(sys.prefix) / 'gqasr-training-profile.json'
        path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
