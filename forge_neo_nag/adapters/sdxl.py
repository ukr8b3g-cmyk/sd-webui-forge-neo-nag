"""SDXL/Illustrious: native attn2 replacement, before the host's single to_out.

Host contract: Haoming02/sd-webui-forge-classic, neo 558e26c,
backend/nn/unet.py and backend/patcher/base.py. No model forward is copied.
"""
from __future__ import annotations

import math

import torch

from ..config import NAGError
from .base import (CrossAttentionAdapter, check_unmodified, context_tensor,
                   validate_literal_text)

MAX_CHUNKS = 4
MAX_TOKENS = 77 * MAX_CHUNKS


def enumerate_attn2(model, layout_types):
    spatial_type, block_type, attention_type = layout_types
    if getattr(model, "in_channels", None) != 4 or getattr(model, "out_channels", None) != 4:
        raise NAGError("SDXL NAG requires the standard four-channel image UNet, not inpainting/video.")
    targets = {}
    for group in ("input", "middle", "output"):
        seq = getattr(model, group + "_blocks", None) if group != "middle" else [getattr(model, "middle_block", None)]
        if seq is None or not len(seq):
            raise NAGError(f"SDXL UNet has no supported {group} block layout.")
        for number, layers in enumerate(seq):
            if layers is None:
                raise NAGError("SDXL UNet has no middle block.")
            for layer in layers:
                if not isinstance(layer, spatial_type):
                    continue
                if type(layer) is not spatial_type:
                    raise NAGError("Unsupported SpatialTransformer replacement.")
                check_unmodified(layer, "SDXL SpatialTransformer")
                for local_index, block in enumerate(layer.transformer_blocks):
                    if type(block) is not block_type or getattr(block, "disable_self_attn", False):
                        raise NAGError("SDXL NAG requires native blocks with separate Self-Attention.")
                    check_unmodified(block, "SDXL transformer block", ("forward", "_forward"))
                    attn = block.attn2
                    if type(attn) is not attention_type:
                        raise NAGError("SDXL NAG requires native attn2.")
                    check_unmodified(attn, "SDXL attn2")
                    if (attn.to_k.in_features != attn.to_v.in_features
                            or attn.to_q.out_features != attn.to_k.out_features
                            or attn.to_q.out_features != attn.to_v.out_features
                            or attn.heads <= 0
                            or attn.to_q.out_features != attn.heads * attn.dim_head):
                        raise NAGError("Unsupported SDXL attn2 projection layout.")
                    # The setter calls this argument transformer_index, but the
                    # native lookup uses SpatialTransformer's LOCAL block_index.
                    key = (group, number, local_index)
                    if key in targets:
                        raise NAGError("SDXL attn2 registration keys collide; refusing to overwrite a layer.")
                    targets[key] = attn
    if not targets or len({id(x) for x in targets.values()}) != len(targets):
        raise NAGError("SDXL NAG found no unique native attn2 targets.")
    discovered = {id(m.attn2) for m in model.modules() if isinstance(m, block_type) and m.attn2 is not None}
    if discovered != {id(x) for x in targets.values()}:
        raise NAGError("Not every SDXL attn2 can be addressed by the host's replacement API.")
    if len({a.to_k.in_features for a in targets.values()}) != 1:
        raise NAGError("SDXL attn2 context widths disagree.")
    return targets


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    for name in ("text_processing_engine_l", "text_processing_engine_g"):
        engine = getattr(p.sd_model, name, None)
        if engine is None or not callable(getattr(engine, "tokenize_line", None)):
            raise NAGError("SDXL NAG requires both native CLIP text-processing engines.")
        chunks, _ = engine.tokenize_line(config.negative)
        if not chunks or len(chunks) > MAX_CHUNKS or any(len(c.tokens) != 77 for c in chunks):
            raise NAGError("SDXL NAG negative exceeds four native CLIP chunks (308 tokens), or has an unsupported chunk layout; not truncated.")
        if any(c.fixes for c in chunks):
            raise NAGError("Textual Inversion in the NAG negative is not supported. / NAG欄のTextual Inversionは非対応です。")
    prompt = bindings.conditioning_type([config.negative], is_negative_prompt=True,
                                       width=p.width, height=p.height)
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, dict) or "crossattn" not in encoded:
        raise NAGError("SDXL encoder did not return a crossattn dictionary.")
    targets = enumerate_attn2(p.sd_model.forge_objects.unet.model.diffusion_model, bindings.layout_types)
    width = next(iter(targets.values())).to_k.in_features
    result = context_tensor(encoded["crossattn"], width=width, name="SDXL", max_tokens=MAX_TOKENS)
    if result.shape[1] % 77:
        raise NAGError("SDXL NAG conditioning is not a native 77-position CLIP chunk sequence.")
    # Deliberately discard negative vector. The regular CFG call supplies its
    # own positive/unconditional pooled embeddings, crop and size conditions.
    return result


class _AttentionPatch:
    def __init__(self, owner, key, attention):
        self.owner, self.key, self.attention = owner, key, attention

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def __call__(self, q, k, v, extra_options):
        owner, attn = self.owner, self.attention
        if extra_options.get("n_heads") != attn.heads or extra_options.get("dim_head") != attn.dim_head:
            raise NAGError("SDXL attn2 head layout changed during sampling.")
        if any(extra_options.get(key) is not None for key in ("attention_mask", "attn_mask", "mask")):
            raise NAGError("SDXL NAG does not support additional attention masks.")
        selected_q = owner.selected(q)
        ctx = owner.negative_for(selected_q)
        kn, vn = attn.to_k(ctx), attn.to_v(ctx)
        positive = owner.attention(q, k, v, attn.heads)
        negative = owner.attention(selected_q, kn, vn, attn.heads)
        # The host executes to_out exactly once AFTER this callback returns.
        return owner.mix(positive, negative, self.key)


class SDXLAdapter(CrossAttentionAdapter):
    name = "SDXL"
    implementation = "sdxl-attn2-v1"

    def __init__(self, model, negative_context, config, ops, layout_types):
        super().__init__(model, negative_context, config)
        self.attention = ops.attention
        self.layers = enumerate_attn2(model, layout_types)
        self.targets = tuple(self.layers)
        self.patches = {key: _AttentionPatch(self, key, attn) for key, attn in self.layers.items()}

    def install(self, patcher):
        for (group, number, local_index), patch in self.patches.items():
            patcher.set_model_attn2_replace(patch, group, number, local_index)

    def validate_owned(self, options):
        replace = options.get("patches_replace", {})
        if not isinstance(replace, dict) or set(replace) != {"attn2"}:
            raise NAGError("SDXL NAG owned attention registrations are missing or another replacement is active.")
        actual = replace["attn2"]
        if (not isinstance(actual, dict) or set(actual) != set(self.patches)
                or any(actual[key] is not patch for key, patch in self.patches.items())):
            raise NAGError("SDXL NAG attention callbacks changed during sampling.")

    def without_patches(self, options):
        self.validate_owned(options)
        # All replacements have just been proven to be ours. Never mutate the
        # clone's dictionary or an options dictionary belonging to the host.
        result = dict(options)
        result.pop("patches_replace")
        return result

    def memory_reserve(self, x, dtype):
        if x.ndim != 4 or x.shape[1] != 4:
            raise NAGError("SDXL NAG supports four-dimensional image latents only.")
        element = torch.empty((), dtype=dtype).element_size()
        features = max(a.to_q.out_features for a in self.layers.values())
        image_tokens = math.prod(x.shape[-2:])  # conservative highest-resolution bound
        # Account for a native batched cond/uncond call as well as FP32 mixing.
        return int(2 * x.shape[0] * features * (image_tokens * (6 * element + 20)
                    + self.negative_context.shape[1] * 2 * element)
                   + self.negative_context.numel() * max(element, self.negative_context.element_size()))
