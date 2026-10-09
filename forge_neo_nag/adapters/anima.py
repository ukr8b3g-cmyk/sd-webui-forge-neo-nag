"""Anima: request-local views reuse native model and Block.forward verbatim.

Only cross_attn is delegated to NAG. The shared nn.Module tree, self-attention,
modulation, FP16/FP32 residual policy, final layer and position logic are native.
Host contract: Forge Neo 558e26c, backend/nn/anima.py.
"""
from __future__ import annotations

import math

import torch

from ..config import NAGError
from .base import (CrossAttentionAdapter, check_unmodified, context_tensor,
                   validate_literal_text)

MAX_TOKENS = 2048


def _valid_context(context, batch, width):
    # Forge prompt batching can retain the encoder's singleton batch axis.
    return (isinstance(context, torch.Tensor) and context.ndim in (3, 4)
            and (context.ndim == 3 or context.shape[1] == 1)
            and context.shape[0] == batch and context.shape[-1] == width)


def validate_model(model, layout_types):
    block_type, attention_type = layout_types
    for attr in ("blocks", "patch_spatial", "patch_temporal", "in_channels", "out_channels",
                 "prepare_embedded_sequence", "t_embedder", "t_embedding_norm", "final_layer", "unpatchify"):
        if not hasattr(model, attr):
            raise NAGError(f"Anima model is missing '{attr}'.")
    if not model.blocks or model.patch_spatial < 1 or model.patch_temporal != 1:
        raise NAGError("Anima NAG requires the native still-image block layout.")
    widths = set()
    for block in model.blocks:
        if type(block) is not block_type:
            raise NAGError("Anima NAG requires native Block modules.")
        check_unmodified(block, "Anima block")
        attn = block.cross_attn
        if type(attn) is not attention_type or attn.is_SelfAttn:
            raise NAGError("Anima NAG requires separate native Cross-Attention.")
        check_unmodified(attn, "Anima Cross-Attention", ("forward", "compute_qkv", "compute_attention", "torch_attention_op"))
        if (attn.k_proj.in_features != attn.v_proj.in_features
                or attn.q_proj.out_features != attn.n_heads * attn.head_dim
                or attn.k_proj.out_features != attn.q_proj.out_features
                or attn.v_proj.out_features != attn.q_proj.out_features):
            raise NAGError("Unsupported Anima Cross-Attention projection layout.")
        widths.add(attn.k_proj.in_features)
    if len(widths) != 1:
        raise NAGError("Anima Cross-Attention context widths disagree.")
    return widths.pop()


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    engine = p.sd_model.text_processing_engine_qwen
    tokens = engine.tokenize(config.negative)
    if (not isinstance(tokens, (tuple, list)) or len(tokens) != 2
            or any(not isinstance(t, (tuple, list)) or len(t) > MAX_TOKENS for t in tokens)):
        raise NAGError("Anima NAG requires Qwen and T5 token lists, each at most 2048 tokens; not truncated.")
    prompt = bindings.conditioning_type([config.negative], is_negative_prompt=True,
                                       width=p.width, height=p.height)
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1:
        raise NAGError("Anima encoder must return one preprocessed conditioning tensor.")
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    width = validate_model(model, bindings.layout_types)
    # Keep the native minimum-512 padding, and all native preprocessing.
    return context_tensor(encoded[0], width=width, name="Anima", max_tokens=MAX_TOKENS)


class _View:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)


class _CrossAttention:
    def __init__(self, owner, index, original):
        self.owner, self.index, self.original = owner, index, original

    def __call__(self, x, context=None, rope_emb=None, transformer_options=None):
        owner, attn = self.owner, self.original
        options = transformer_options or {}
        if not _valid_context(context, x.shape[0], owner.context_width):
            raise NAGError("Unexpected Anima positive context batch.")
        q, k, v = attn.compute_qkv(x, context, rope_emb=rope_emb)
        sq = owner.selected(q)
        ctx = owner.negative_for(sq)
        b, length = ctx.shape[:2]
        kn = attn.k_proj(ctx).reshape(b, length, attn.n_heads, attn.head_dim)
        vn = attn.v_proj(ctx).reshape(b, length, attn.n_heads, attn.head_dim)
        kn, vn = attn.k_norm(kn), attn.v_norm(vn)
        positive = attn.torch_attention_op(q, k, v, transformer_options=options)
        negative = attn.torch_attention_op(sq, kn, vn, transformer_options=options)
        guided = owner.mix(positive, negative, self.index)
        return attn.output_dropout(attn.output_proj(guided))


class _BlockView(_View):
    def __init__(self, original, cross):
        super().__init__(original)
        self.cross_attn = cross

    def __call__(self, *args, **kwargs):
        return type(self._original).forward(self, *args, **kwargs)


class _AnimaView(_View):
    def __init__(self, original, blocks):
        super().__init__(original)
        self.blocks = blocks

    def __call__(self, *args, **kwargs):
        return type(self._original).forward(self, *args, **kwargs)


class AnimaAdapter(CrossAttentionAdapter):
    name = "Anima"
    implementation = "anima-cross-attention-v1"

    def __init__(self, model, negative_context, config, ops, layout_types):
        super().__init__(model, negative_context, config)
        self.context_width = validate_model(model, layout_types)
        self.targets = tuple(range(len(model.blocks)))
        self.view = _AnimaView(model, tuple(
            _BlockView(block, _CrossAttention(self, i, block.cross_attn))
            for i, block in enumerate(model.blocks)))

    def __call__(self, x, timesteps, context, transformer_options=None, control=None, **kwargs):
        if control is not None or any(v is not None for v in kwargs.values()):
            raise NAGError("Anima NAG does not accept reference, control or extra model conditions.")
        if x.ndim != 5 or x.shape[2] != 1 or x.shape[1] != self.model.in_channels:
            raise NAGError("Anima NAG expects native single-frame five-dimensional latents.")
        if not _valid_context(context, x.shape[0], self.context_width):
            raise NAGError("Unexpected Anima conditioning shape.")
        return self.view(x, timesteps, context, transformer_options=transformer_options or {})

    def memory_reserve(self, x, dtype):
        if x.ndim != 5 or x.shape[2] != 1 or x.shape[1] != self.model.in_channels:
            raise NAGError("Anima NAG supports native single-frame image latents only.")
        element = torch.empty((), dtype=dtype).element_size()
        features = max(b.cross_attn.q_proj.out_features for b in self.model.blocks)
        image_tokens = math.ceil(x.shape[-2] / self.model.patch_spatial) * math.ceil(x.shape[-1] / self.model.patch_spatial)
        return int(2 * x.shape[0] * features * (image_tokens * (6 * element + 20)
                    + self.negative_context.shape[1] * 2 * element)
                   + self.negative_context.numel() * max(element, self.negative_context.element_size()))
