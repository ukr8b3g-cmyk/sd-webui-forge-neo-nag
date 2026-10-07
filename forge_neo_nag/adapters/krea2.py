"""Forge-native Krea2 attention NAG.

Adapted from iljung1106/ComfyUI-Krea2-NAG (MIT). The host's existing
Linear/LoRA/quantization and attention functions remain in use. Image Q/K/V
are projected once per block, then shared by the two attention contexts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

from ..config import NAGConfig, NAGError
from ..math import guide_attention


@dataclass(frozen=True)
class Krea2Ops:
    attention: Callable
    rope: Callable
    timestep_embedding: Callable
    pad: Callable


def _heads(tensor: torch.Tensor, heads: int) -> torch.Tensor:
    b, length, features = tensor.shape
    return tensor.reshape(b, length, heads, features // heads).transpose(1, 2)


def _project(attn, x, freqs, ops: Krea2Ops):
    q = _heads(attn.wq(x), attn.heads)
    k = _heads(attn.wk(x), attn.kvheads)
    v = _heads(attn.wv(x), attn.kvheads)
    gate = attn.gate(x)
    q, k = attn.qknorm(q, k)
    if freqs is not None:
        q, k = ops.rope(q, k, freqs)
    if attn.heads % attn.kvheads:
        raise NAGError("Krea2 attention heads must be divisible by KV heads.")
    if attn.kvheads != attn.heads:
        repeats = attn.heads // attn.kvheads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
    return q, k, v, gate


def nag_block(block, positive_text, negative_text, image, tvec,
              positive_freqs, negative_text_freqs, config: NAGConfig,
              options: dict, ops: Krea2Ops):
    """Update one shared image state and two independent text states."""
    pre_scale, pre_shift, pre_gate, post_scale, post_shift, post_gate = block.mod(tvec)
    pos_len, neg_len = positive_text.shape[1], negative_text.shape[1]
    positive = torch.cat((positive_text, image), dim=1)
    pos_pre = torch.addcmul(pre_shift, 1 + pre_scale, block.prenorm(positive))
    neg_pre = torch.addcmul(pre_shift, 1 + pre_scale, block.prenorm(negative_text))

    pq, pk, pv, pg = _project(block.attn, pos_pre, positive_freqs, ops)
    nq_text, nk_text, nv_text, ng = _project(block.attn, neg_pre, negative_text_freqs, ops)
    # Both branches use exactly these same, already-normalized/rotated image
    # projections. Negative text lengths need not equal positive text lengths.
    nq = torch.cat((nq_text, pq[:, :, pos_len:]), dim=2)
    nk = torch.cat((nk_text, pk[:, :, pos_len:]), dim=2)
    nv = torch.cat((nv_text, pv[:, :, pos_len:]), dim=2)

    positive_raw = ops.attention(pq, pk, pv, block.attn.heads, mask=None,
                                 skip_reshape=True, transformer_options=options)
    negative_raw = ops.attention(nq, nk, nv, block.attn.heads, mask=None,
                                 skip_reshape=True, transformer_options=options)
    image_attention = guide_attention(positive_raw[:, pos_len:], negative_raw[:, neg_len:],
                                       phi=config.phi, tau=config.tau, alpha=config.alpha)
    positive_raw = torch.cat((positive_raw[:, :pos_len], image_attention), dim=1)
    positive = torch.addcmul(positive, pre_gate, block.attn.wo(positive_raw * torch.sigmoid(pg)))
    negative_text = torch.addcmul(
        negative_text, pre_gate,
        block.attn.wo(negative_raw[:, :neg_len] * torch.sigmoid(ng)),
    )
    positive = torch.addcmul(positive, post_gate, block.mlp(
        torch.addcmul(post_shift, 1 + post_scale, block.postnorm(positive))))
    negative_text = torch.addcmul(negative_text, post_gate, block.mlp(
        torch.addcmul(post_shift, 1 + post_scale, block.postnorm(negative_text))))
    return positive[:, :pos_len], negative_text, positive[:, pos_len:]


def _patchify(x: torch.Tensor, patch: int) -> torch.Tensor:
    b, c, h, w = x.shape
    return x.reshape(b, c, h // patch, patch, w // patch, patch).permute(
        0, 2, 4, 1, 3, 5).reshape(b, (h // patch) * (w // patch), c * patch * patch)


def _unpatchify(x: torch.Tensor, height: int, width: int, patch: int, channels: int) -> torch.Tensor:
    b = x.shape[0]
    return x.reshape(b, height, width, channels, patch, patch).permute(
        0, 3, 1, 4, 2, 5).reshape(b, channels, height * patch, width * patch)


class Krea2Adapter:
    """One adapter per sampling run; caches must not outlive that run."""
    name = "Krea2"

    def __init__(self, model, negative_context: torch.Tensor, config: NAGConfig, ops: Krea2Ops):
        self.model = model
        self.config = config
        self.ops = ops
        self.negative_context = negative_context
        self._negative_text = None
        self._cache_key = None
        self.text_fusion_calls = 0

    def clear(self):
        self._negative_text = None
        self._cache_key = None
        self.negative_context = None

    def _initial_negative(self, like: torch.Tensor, options: dict) -> torch.Tensor:
        if self.negative_context is None:
            raise NAGError("NAG sampling cache has already been released.")
        key = (like.device, like.dtype)
        if self._cache_key != key:
            # Forge TextFusion uses in-place operations; never pass the master
            # conditioning tensor or the cached initial state for mutation.
            ctx = self.negative_context.to(device=like.device, dtype=like.dtype).clone()
            self._negative_text = self.model.txtmlp(
                self.model.txtfusion(ctx, mask=None, transformer_options=options))
            self._cache_key = key
            self.text_fusion_calls += 1
        return self._negative_text.expand(like.shape[0], -1, -1).clone()

    @torch.inference_mode()
    def __call__(self, x, timesteps, context, attention_mask=None,
                 transformer_options=None, control=None, **kwargs):
        if control is not None or attention_mask is not None or any(v is not None for v in kwargs.values()):
            raise NAGError("Krea2 NAG V1 does not accept ControlNet, reference, or extra model conditions.")
        model, ops = self.model, self.ops
        options = dict(transformer_options or {})
        if any(branch != 0 for branch in options.get("cond_or_uncond", [0])):
            raise NAGError("NAG requires the positive-only CFG=1 branch.")
        temporal = x.ndim == 5
        if temporal:
            if x.shape[2] != 1:
                raise NAGError("Krea2 NAG V1 supports still images only, not multi-frame latents.")
            x = x[:, :, 0]
        if x.ndim != 4 or x.shape[1] != model.channels:
            raise NAGError("Unexpected Krea2 latent shape.")
        b, _, original_h, original_w = x.shape
        if context.ndim != 4 or context.shape[0] != b or tuple(context.shape[2:]) != (model.txtlayers, model.txtdim):
            raise NAGError("Krea2 conditioning must have shape [batch, tokens, text layers, text dimension].")
        padded = ops.pad(x, (model.patch, model.patch))
        grid_h, grid_w = padded.shape[-2] // model.patch, padded.shape[-1] // model.patch
        image = model.first(_patchify(padded, model.patch))
        time = model.tmlp(ops.timestep_embedding(timesteps, model.tdim).unsqueeze(1).to(image.dtype))
        tvec = model.tproj(time)
        positive_text = model.txtmlp(model.txtfusion(context.clone(), mask=None, transformer_options=options))
        negative_text = self._initial_negative(context, options)
        pos_len = positive_text.shape[1]
        image_len = image.shape[1]

        # Position indices for image tokens do not depend on the text length.
        ids = torch.zeros(b, pos_len + image_len, 3, device=x.device, dtype=torch.float32)
        ids[:, pos_len:, 1] = torch.arange(grid_h, device=x.device).repeat_interleave(grid_w)
        ids[:, pos_len:, 2] = torch.arange(grid_w, device=x.device).repeat(grid_h)
        positive_freqs = model.pe_embedder(ids)
        negative_freqs = model.pe_embedder(torch.zeros(
            b, negative_text.shape[1], 3, device=x.device, dtype=torch.float32))
        options.update(total_blocks=len(model.blocks), block_type="single", img_slice=[pos_len, pos_len + image_len])
        for index, block in enumerate(model.blocks):
            options["block_index"] = index
            positive_text, negative_text, image = nag_block(
                block, positive_text, negative_text, image, tvec,
                positive_freqs, negative_freqs, self.config, options, ops)

        # Preserve the host's last-layer evaluation layout and crop semantics.
        final = model.last(torch.cat((positive_text, image), dim=1), time)
        result = _unpatchify(final[:, pos_len:pos_len + image_len], grid_h, grid_w,
                            model.patch, model.channels)[:, :, :original_h, :original_w]
        return result.unsqueeze(2) if temporal else result
