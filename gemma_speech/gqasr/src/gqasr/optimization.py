"""Identical optimizer grouping for training and the cluster memory/backward check."""

import torch


def build_optimizer(model, cfg):
    parameters = {"bridge": [], "acoustic": [], "decoder": [], "embedding": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = (
            "bridge" if name.startswith("gqasr_bridge.") else
            "acoustic" if name.startswith("gqasr_acoustic.") else
            "embedding" if "embed_tokens" in name or "lm_head" in name else
            "decoder"
        )
        parameters[group].append(parameter)
    groups = []
    for group, params in parameters.items():
        if params:
            lr = cfg["learning_rate"] if group == "bridge" else cfg[group + "_learning_rate"]
            groups.append({"params": params, "lr": lr, "base_lr": lr, "name": group})
    return torch.optim.AdamW(groups, weight_decay=cfg["weight_decay"])


def training_scope(model, optimizer):
    audio = model.gqasr_acoustic.core
    return {
        "acoustic_blocks_total": len(audio.layers),
        "trainable_acoustic_block_indices": [
            i for i, layer in enumerate(audio.layers) if any(p.requires_grad for p in layer.parameters())
        ],
        "acoustic_frontend_trainable": any(
            p.requires_grad for p in audio.subsample_conv_projection.parameters()
        ),
        "acoustic_output_projection_trainable": any(p.requires_grad for p in audio.output_proj.parameters()),
        "optimizer_groups": {
            g["name"]: {"parameters": sum(p.numel() for p in g["params"]), "base_lr": g["base_lr"]}
            for g in optimizer.param_groups
        },
    }
