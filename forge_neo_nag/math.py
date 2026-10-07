"""Attention-space L1 NAG, following ComfyUI-Krea2-NAG (MIT).

See licenses/ComfyUI-Krea2-NAG-MIT.txt and docs/IMPLEMENTATION.md.
All normalization is FP32; no model weights are converted or replaced.
"""
from __future__ import annotations

import torch


def guide_attention(positive: torch.Tensor, negative: torch.Tensor, *,
                    phi: float, tau: float, alpha: float) -> torch.Tensor:
    if positive.shape != negative.shape:
        raise ValueError("Positive and negative image attention must have identical shapes.")
    if positive.device != negative.device or positive.dtype != negative.dtype:
        raise ValueError("Attention devices and dtypes must agree.")
    if not positive.is_floating_point() or positive.ndim < 2:
        raise ValueError("Attention must be a floating point tensor with a feature axis.")
    if phi == 0 or alpha == 0:
        return positive

    # Do not mutate the inputs: callers still need the positive text attention.
    p = positive.float()
    n = negative.float()
    extrapolated = p + phi * (p - n)
    eps = torch.finfo(torch.float32).eps
    norm_p = p.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    norm_e = extrapolated.abs().sum(dim=-1, keepdim=True).clamp_min(eps)
    # Equivalent to min(ratio, tau) / ratio, without the extra division.
    shrink = (tau * norm_p / norm_e).clamp_max(1.0)
    refined = alpha * (extrapolated * shrink) + (1.0 - alpha) * p
    return refined.to(positive.dtype)
