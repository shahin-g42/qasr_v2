"""Opt-in CUDA loss/gradient equivalence check before selecting the Liger backend."""

import json

import torch
from torch.nn import functional as F

from .losses import chunked_causal_loss


def main():
    if not torch.cuda.is_available():
        raise SystemExit("This check requires CUDA and the optional audm[fused] extra")
    torch.manual_seed(1729)
    reports = []
    for dtype in (torch.float32, torch.bfloat16):
        for smoothing in (0.0, 0.1):
            head = torch.nn.Linear(1024, 151936, bias=False, device="cuda", dtype=dtype)
            torch.nn.init.normal_(head.weight, std=0.02)
            reference = torch.nn.Linear(1024, 151936, bias=False, device="cuda", dtype=dtype)
            reference.weight.data.copy_(head.weight)
            ids = torch.randint(0, 151936, (2, 33), device="cuda")
            labels = ids.clone()
            labels[:, :16] = -100
            losses = []
            for module, backend in ((reference, "chunked"), (head, "liger")):
                # Exercise input/output tying and a denominator spanning accumulation.
                states = F.embedding(ids, module.weight)
                loss = chunked_causal_loss(
                    states,
                    module,
                    labels,
                    epsilon=smoothing,
                    num_items_in_batch=68,
                    backend=backend,
                )
                loss.backward()
                losses.append(loss.detach())
            torch.testing.assert_close(losses[0], losses[1], rtol=0.003, atol=0.003)
            torch.testing.assert_close(
                reference.weight.grad,
                head.weight.grad,
                rtol=0.03 if dtype == torch.bfloat16 else 0.003,
                atol=5e-5 if dtype == torch.bfloat16 else 2e-6,
            )
            reports.append(
                dict(
                    dtype=str(dtype),
                    smoothing=smoothing,
                    loss_difference=float(abs(losses[0] - losses[1])),
                    maximum_gradient_difference=float(
                        (reference.weight.grad - head.weight.grad).abs().max()
                    ),
                )
            )
            del head, reference, states, loss
    print(json.dumps(dict(status="passed", checks=reports), indent=2))


if __name__ == "__main__":
    main()
