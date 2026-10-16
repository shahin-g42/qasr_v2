"""Standard PyTorch gradient clipping with failure diagnostics."""

import json

import torch


@torch.no_grad()
def clip_gradients(model, max_norm, runtime, step):
    parameters = [(name, p) for name, p in model.named_parameters() if p.grad is not None]
    if not parameters:
        raise RuntimeError("GQASR backward produced no gradients")
    # Keep only the finiteness flags for diagnostics: clip_grad_norm_ can turn
    # initially finite gradients into NaNs when the total norm is nonfinite.
    flags = torch.stack([torch.isfinite(p.grad).all() for _, p in parameters])
    norm = torch.nn.utils.clip_grad_norm_([p for _, p in parameters], max_norm)
    if not runtime.all_finite(norm):
        bad = [name for (name, _), ok in zip(parameters, flags.tolist(), strict=True) if not ok]
        print(json.dumps({
            "event": "nonfinite_gradients", "rank": runtime.rank,
            "optimizer_step_attempt": step + 1, "parameters": bad,
            "local_gradients_finite_before_clipping": bool(flags.all()),
            "local_norm_finite": bool(torch.isfinite(norm)),
        }), flush=True)
        raise FloatingPointError(
            f"Nonfinite GQASR gradients at update {step + 1}; "
            "see nonfinite_gradients events. Optimizer was not updated."
        )
    return norm
