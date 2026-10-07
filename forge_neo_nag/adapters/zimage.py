"""Z-Image Turbo adapter for Forge Neo NextDiT.

Request-local only: the shared NextDiT tree is never monkey-patched. Native text/image
refiners, native Linear/LoRA/quantized layers and the selected Forge attention backend
are reused. The negative branch keeps its own text state while sharing the positive
image Q/K/V and image RoPE at every main JointTransformerBlock.

Reviewed Forge Neo contract: Haoming02/sd-webui-forge-classic neo
831d242d4cf45a2ba1a64c2087ef0a17934a9daf, backend/nn/lumina.py,
backend/diffusion_engine/zimage.py and backend/text_processing/z_image_engine.py.
"""
from __future__ import annotations

import contextlib
import math

import torch

from ..config import NAGConfig, NAGError
from ..math import guide_attention
from .base import check_unmodified, context_tensor, validate_literal_text

MAX_TOKENS = 2048


def validate_model(model, layout_types):
    block_type, attention_type = layout_types
    required = (
        "context_refiner", "noise_refiner", "layers", "cap_embedder", "x_embedder",
        "t_embedder", "rope_embedder", "final_layer", "unpatchify", "patchify_and_embed",
        "patch_size", "in_channels", "out_channels", "time_scale", "dim", "n_heads",
        "axes_dims", "axes_lens", "pad_tokens_multiple",
    )
    if any(not hasattr(model, name) for name in required):
        raise NAGError("Z-Image NAG requires the native NextDiT layout.")
    # The ZImage engine type is checked by the host; here we validate the native
    # NextDiT contract rather than hard-coding one checkpoint width/layer count.
    # Current official Turbo is dim=3840, 30 heads/layers, axes 32/48/48.
    if (model.patch_size != 2 or model.in_channels != 16 or model.out_channels != 16
            or model.dim < 1 or model.n_heads < 1 or model.dim % model.n_heads
            or len(model.axes_dims) != 3 or sum(model.axes_dims) != model.dim // model.n_heads
            or len(model.axes_lens) != 3 or float(model.time_scale) != 1000.0
            or len(model.context_refiner) != 2 or len(model.noise_refiner) != 2
            or not model.layers):
        raise NAGError("Unsupported Z-Image Turbo NextDiT configuration.")
    if getattr(model, "clip_text_pooled_proj", None) is not None:
        raise NAGError("Z-Image NAG does not support pooled-text NextDiT variants.")
    if getattr(model, "config", {}).get("nunchaku", False):
        raise NAGError("Nunchaku Z-Image is not supported by this adapter.")
    if not hasattr(model.cap_embedder, "__getitem__") or len(model.cap_embedder) < 2:
        raise NAGError("Unexpected Z-Image cap_embedder layout.")
    width = model.cap_embedder[1].in_features
    if width < 1:
        raise NAGError("Unexpected Z-Image text-conditioning width.")
    groups = (model.context_refiner, model.noise_refiner, model.layers)
    for group in groups:
        for block in group:
            if type(block) is not block_type:
                raise NAGError("Z-Image NAG requires native JointTransformerBlock modules.")
            check_unmodified(block, "Z-Image JointTransformerBlock")
            attn = block.attention
            if type(attn) is not attention_type:
                raise NAGError("Z-Image NAG requires native JointAttention modules.")
            check_unmodified(attn, "Z-Image JointAttention")
            if (attn.n_local_heads != model.n_heads or attn.n_local_kv_heads != model.n_heads
                    or attn.n_rep != 1 or not attn.qk_norm
                    or attn.head_dim != model.dim // model.n_heads):
                raise NAGError("Unsupported Z-Image JointAttention head layout.")
    return width


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    engine = p.sd_model.text_processing_engine_qwen
    tokens = engine.tokenize(config.negative)
    if not isinstance(tokens, (tuple, list)) or len(tokens) > MAX_TOKENS:
        raise NAGError(
            f"Z-Image NAG negative exceeds {MAX_TOKENS} Qwen tokens including the template; "
            "it was not truncated."
        )
    prompt = bindings.conditioning_type(
        [config.negative], is_negative_prompt=True, width=p.width, height=p.height,
        distilled_cfg_scale=getattr(p, "distilled_cfg_scale", 1.0),
    )
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1:
        raise NAGError("Z-Image encoder must return one conditioning tensor.")
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    width = validate_model(model, bindings.layout_types)
    value = encoded[0].unsqueeze(0) if encoded[0].ndim == 2 else encoded[0]
    return context_tensor(value, width=width, name="Z-Image", max_tokens=MAX_TOKENS)


def _split_qkv(attn, value):
    q, k, v = torch.split(
        attn.qkv(value),
        [
            attn.n_local_heads * attn.head_dim,
            attn.n_local_kv_heads * attn.head_dim,
            attn.n_local_kv_heads * attn.head_dim,
        ],
        dim=-1,
    )
    b, length = value.shape[:2]
    return (
        q.view(b, length, attn.n_local_heads, attn.head_dim),
        k.view(b, length, attn.n_local_kv_heads, attn.head_dim),
        v.view(b, length, attn.n_local_kv_heads, attn.head_dim),
    )


def _apply_qk_rope(attn, q, k, freqs, ops):
    if not attn.qk_norm or attn.n_local_heads != attn.n_local_kv_heads:
        raise NAGError("Z-Image NAG requires native equal-head RMSNorm RoPE attention.")
    q_scale, _, q_stream = ops.weights_manual_cast(attn.q_norm, q)
    k_scale, _, k_stream = ops.weights_manual_cast(attn.k_norm, k)
    eps = attn.q_norm.eps if attn.q_norm.eps is not None else torch.finfo(torch.float32).eps
    q_ctx = ops.main_stream_worker(q_scale, None, q_stream)
    k_ctx = ops.main_stream_worker(k_scale, None, k_stream)
    with q_ctx, k_ctx:
        q, k = ops.ck.rms_rope(q, k, freqs, q_scale, k_scale, eps)
    return q, k


class ZImageAdapter:
    """Positive-only Z-Image adapter; Forge retains native CFG ownership."""
    name = "Z-Image Turbo"
    implementation = "zimage-joint-attention-hybrid-rope-v1"

    def __init__(self, model, negative_context: torch.Tensor, config: NAGConfig, ops, layout_types):
        self.model = model
        self.negative_context = negative_context
        self.config = config
        self.ops = ops
        self.context_width = validate_model(model, layout_types)
        self.targets = tuple(range(len(model.layers)))
        self.attention_calls = 0
        self._seen = {}
        self.last_positive_image_pe = None
        self.last_negative_image_pe = None

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def _negative_for(self, context):
        if self.negative_context is None:
            raise NAGError("Z-Image NAG sampling cache has already been released.")
        return self.negative_context.to(device=context.device, dtype=context.dtype).expand(
            context.shape[0], -1, -1
        ).clone()

    def _negative_refined(self, context, dtype, options):
        negative = self._negative_for(context)
        negative = self.model.cap_embedder(negative)
        if self.model.pad_tokens_multiple is not None:
            pad = (-negative.shape[1]) % self.model.pad_tokens_multiple
            if pad:
                token = self.model.cap_pad_token.to(
                    device=negative.device, dtype=negative.dtype, copy=True
                ).unsqueeze(0).repeat(negative.shape[0], pad, 1)
                negative = torch.cat((negative, token), dim=1)
        ids = torch.zeros(
            negative.shape[0], negative.shape[1], 3,
            dtype=torch.float32, device=negative.device,
        )
        ids[:, :, 0] = torch.arange(
            negative.shape[1], dtype=torch.float32, device=negative.device
        ) + 1.0
        freqs = self.model.rope_embedder(ids).movedim(1, 2).to(dtype)
        for layer in self.model.context_refiner:
            negative = layer(negative, None, freqs, transformer_options=options)
        return negative, freqs

    def _joint_block(self, index, block, positive, negative, positive_freqs,
                     negative_text_freqs, adaln_input, options, pos_text_len):
        if block.modulation is not True:
            raise NAGError("Z-Image main JointTransformerBlock modulation changed.")
        scale_msa, gate_msa, scale_mlp, gate_mlp = block.adaLN_modulation(adaln_input).chunk(4, dim=1)

        pos_pre = self.ops.modulate(block.attention_norm1(positive), scale_msa)
        neg_pre = self.ops.modulate(block.attention_norm1(negative), scale_msa)
        pq, pk, pv = _split_qkv(block.attention, pos_pre)
        nq, nk, nv = _split_qkv(block.attention, neg_pre)
        pq, pk = _apply_qk_rope(block.attention, pq, pk, positive_freqs, self.ops)
        nq, nk = _apply_qk_rope(block.attention, nq, nk, negative_text_freqs, self.ops)

        neg_len = negative.shape[1]
        iq, ik, iv = pq[:, pos_text_len:], pk[:, pos_text_len:], pv[:, pos_text_len:]
        # Hybrid RoPE invariant: negative text has its own positions; image Q/K/V,
        # including already-applied image RoPE, is exactly the positive branch's.
        neg_q = torch.cat((nq, iq), dim=1)
        neg_k = torch.cat((nk, ik), dim=1)
        neg_v = torch.cat((nv, iv), dim=1)
        self.last_positive_image_pe = positive_freqs[:, pos_text_len:].detach()
        self.last_negative_image_pe = positive_freqs[:, pos_text_len:].detach()

        pos_raw = self.ops.attention(
            pq.movedim(1, 2), pk.movedim(1, 2), pv.movedim(1, 2),
            block.attention.n_local_heads, None, skip_reshape=True,
            transformer_options=options,
        )
        neg_raw = self.ops.attention(
            neg_q.movedim(1, 2), neg_k.movedim(1, 2), neg_v.movedim(1, 2),
            block.attention.n_local_heads, None, skip_reshape=True,
            transformer_options=options,
        )
        guided_img = guide_attention(
            pos_raw[:, pos_text_len:], neg_raw[:, neg_len:],
            phi=self.config.phi, tau=self.config.tau, alpha=self.config.alpha,
        )

        pos_attn = block.attention.out(torch.cat((pos_raw[:, :pos_text_len], guided_img), dim=1))
        neg_attn = block.attention.out(neg_raw[:, :neg_len])
        positive = positive + gate_msa.unsqueeze(1).tanh() * block.attention_norm2(
            self.ops.clamp_fp16(pos_attn)
        )
        negative = negative + gate_msa.unsqueeze(1).tanh() * block.attention_norm2(
            self.ops.clamp_fp16(neg_attn)
        )
        positive = positive + gate_mlp.unsqueeze(1).tanh() * block.ffn_norm2(
            self.ops.clamp_fp16(
                block.feed_forward(self.ops.modulate(block.ffn_norm1(positive), scale_mlp))
            )
        )
        negative = negative + gate_mlp.unsqueeze(1).tanh() * block.ffn_norm2(
            self.ops.clamp_fp16(
                block.feed_forward(self.ops.modulate(block.ffn_norm1(negative), scale_mlp))
            )
        )
        self.attention_calls += 1
        self._seen[index] = self._seen.get(index, 0) + 1
        return positive, negative

    @torch.inference_mode()
    def __call__(self, x, timesteps, context, num_tokens=None, attention_mask=None,
                 transformer_options=None, control=None, **kwargs):
        if control is not None:
            raise NAGError("Z-Image NAG does not support ControlNet.")
        if attention_mask is not None:
            raise NAGError("Z-Image NAG does not support explicit attention masks.")
        if any(value is not None for value in kwargs.values()):
            raise NAGError("Z-Image NAG received unsupported extra model conditions.")
        if (x.ndim != 4 or context.ndim != 3 or context.shape[0] != x.shape[0]
                or context.shape[-1] != self.context_width):
            raise NAGError("Unexpected Z-Image image/context shape.")
        options = dict(transformer_options or {})
        for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "attention_override"):
            if options.get(key):
                raise NAGError("Z-Image NAG cannot combine with existing transformer patches.")

        original_h, original_w = x.shape[-2:]
        x = self.ops.pad_to_patch_size(x, (self.model.patch_size, self.model.patch_size))
        t = 1.0 - timesteps
        adaln_input = self.model.t_embedder(t * self.model.time_scale, dtype=x.dtype)
        positive_cap = self.model.cap_embedder(context)
        positive, mask, img_size, cap_size, positive_freqs = self.model.patchify_and_embed(
            x, positive_cap, None, adaln_input, num_tokens, transformer_options=options
        )
        if mask is not None or len(set(cap_size)) != 1:
            raise NAGError("Unexpected Z-Image native sequence layout.")
        pos_text_len = cap_size[0]
        if pos_text_len < 1 or positive.shape[1] <= pos_text_len:
            raise NAGError("Z-Image native image token span is empty.")
        negative, negative_text_freqs = self._negative_refined(context, x.dtype, options)
        hybrid_freqs = torch.cat((negative_text_freqs, positive_freqs[:, pos_text_len:]), dim=1)
        if hybrid_freqs.shape[1] != negative.shape[1] + positive.shape[1] - pos_text_len:
            raise NAGError("Z-Image hybrid RoPE sequence length mismatch.")

        self._seen = {}
        completed = False
        try:
            for index, block in enumerate(self.model.layers):
                positive, negative = self._joint_block(
                    index, block, positive, negative, positive_freqs, negative_text_freqs,
                    adaln_input, options, pos_text_len,
                )
            positive = self.model.final_layer(positive, adaln_input)
            result = self.model.unpatchify(
                positive, img_size, cap_size, return_tensor=True
            )[:, :, :original_h, :original_w]
            completed = True
            return -result
        finally:
            if completed and (set(self._seen) != set(self.targets)
                              or any(n != 1 for n in self._seen.values())):
                raise NAGError("Z-Image NAG did not execute every native main block exactly once.")
            self._seen = {}

    def memory_reserve(self, x, dtype):
        if x.ndim != 4:
            raise NAGError("Z-Image NAG supports four-dimensional image latents only.")
        element = torch.empty((), dtype=dtype).element_size()
        h = math.ceil(x.shape[-2] / self.model.patch_size)
        w = math.ceil(x.shape[-1] / self.model.patch_size)
        image_tokens = h * w
        if self.model.pad_tokens_multiple is not None:
            image_tokens += (-image_tokens) % self.model.pad_tokens_multiple
        text_tokens = self.negative_context.shape[1]
        if self.model.pad_tokens_multiple is not None:
            text_tokens += (-text_tokens) % self.model.pad_tokens_multiple
        features = self.model.dim
        return int(
            x.shape[0] * features * (
                image_tokens * (8 * element + 16) + text_tokens * (10 * element + 16)
            ) + self.negative_context.numel() * max(element, self.negative_context.element_size())
        )

    def clear(self):
        self.negative_context = None
        self._seen = {}
        self.last_positive_image_pe = None
        self.last_negative_image_pe = None
