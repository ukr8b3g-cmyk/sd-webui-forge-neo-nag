"""Ernie Image adapter for Forge Neo joint self-attention.

Request-local only. Native ErnieImageModel.forward remains the outer execution path
through a read-only view. Each block shares positive image Q/K/V (including image
RoPE) with the negative branch, advances an independent negative text state, and
applies NAG only to image attention rows.

Reviewed Forge Neo contract: Haoming02/sd-webui-forge-classic neo
90c2766d4630f3629d4cc7bd7afad587b6ca690a, with unchanged Ernie blobs:
backend/diffusion_engine/ernie.py 6ec48d7f, backend/nn/ernie.py b71e6ef6,
backend/text_processing/ernie_engine.py e0a4d820.
"""
from __future__ import annotations

import math

import torch

from ..config import NAGConfig, NAGError
from ..math import guide_attention
from .base import check_unmodified, context_tensor, validate_literal_text

MAX_TOKENS = 2048


def validate_model(model, layout_types):
    block_type, attention_type, patch_type = layout_types
    required = (
        "hidden_size", "num_heads", "head_dim", "patch_size", "out_channels",
        "x_embedder", "text_proj", "time_proj", "time_embedding", "pos_embed",
        "adaLN_modulation", "layers", "final_norm", "final_linear",
    )
    if any(not hasattr(model, name) for name in required):
        raise NAGError("Ernie NAG requires the native ErnieImageModel layout.")
    if (model.hidden_size < 1 or model.num_heads < 1
            or model.hidden_size != model.num_heads * model.head_dim
            or model.patch_size < 1 or not model.layers):
        raise NAGError("Unsupported Ernie Image model dimensions.")
    if type(model.x_embedder) is not patch_type:
        raise NAGError("Ernie NAG requires the native dynamic image patch embedder.")
    proj = model.x_embedder.proj
    if (proj.out_channels != model.hidden_size
            or model.x_embedder.patch_size != model.patch_size):
        raise NAGError("Unexpected Ernie image patch projection layout.")
    axes = getattr(model.pos_embed, "axes_dim", None)
    if not isinstance(axes, list) or len(axes) != 3 or sum(axes) != model.head_dim:
        raise NAGError("Unexpected Ernie rotary-position layout.")
    if model.text_proj is None:
        width = model.hidden_size
    else:
        if model.text_proj.out_features != model.hidden_size or model.text_proj.in_features < 1:
            raise NAGError("Unexpected Ernie text projection layout.")
        width = model.text_proj.in_features
    for block in model.layers:
        if type(block) is not block_type:
            raise NAGError("Ernie NAG requires native ErnieImageSharedAdaLNBlock modules.")
        check_unmodified(block, "Ernie block")
        attn = block.self_attention
        if type(attn) is not attention_type:
            raise NAGError("Ernie NAG requires native ErnieImageAttention modules.")
        check_unmodified(attn, "Ernie attention")
        if (attn.heads != model.num_heads or attn.head_dim != model.head_dim
                or attn.inner_dim != model.hidden_size
                or attn.to_q.in_features != model.hidden_size
                or attn.to_q.out_features != model.hidden_size
                or attn.to_k.in_features != model.hidden_size
                or attn.to_k.out_features != model.hidden_size
                or attn.to_v.in_features != model.hidden_size
                or attn.to_v.out_features != model.hidden_size
                or len(attn.to_out) != 1
                or attn.to_out[0].in_features != model.hidden_size
                or attn.to_out[0].out_features != model.hidden_size):
            raise NAGError("Unsupported Ernie Image attention projection layout.")
    return width


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    engine = p.sd_model.text_processing_engine_ministral
    tokens = engine.tokenize(config.negative)
    if not isinstance(tokens, (tuple, list)) or len(tokens) > MAX_TOKENS:
        raise NAGError(
            f"Ernie NAG negative exceeds the {MAX_TOKENS}-token NAG safety limit; "
            "it was not truncated."
        )
    encoded = p.sd_model.get_learned_conditioning([config.negative])
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1:
        raise NAGError("Ernie Ministral encoder must return one conditioning tensor.")
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    width = validate_model(model, bindings.layout_types)
    value = encoded[0].unsqueeze(0) if encoded[0].ndim == 2 else encoded[0]
    return context_tensor(value, width=width, name="Ernie", max_tokens=MAX_TOKENS)


def _project(attn, value, rotary, ops):
    b, length = value.shape[:2]
    q = attn.to_q(value).view(b, length, attn.heads, attn.head_dim)
    k = attn.to_k(value).view(b, length, attn.heads, attn.head_dim)
    v = attn.to_v(value)
    q_scale, _, q_stream = ops.weights_manual_cast(attn.norm_q, q)
    k_scale, _, k_stream = ops.weights_manual_cast(attn.norm_k, k)
    q_ctx = ops.main_stream_worker(q_scale, None, q_stream)
    k_ctx = ops.main_stream_worker(k_scale, None, k_stream)
    with q_ctx, k_ctx:
        q, k = ops.ck.rms_rope_split_half(
            q, k, rotary, q_scale, k_scale, attn.norm_q.eps
        )
    return q.reshape(b, length, -1), k.reshape(b, length, -1), v


class _View:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)


class _BlockView:
    def __init__(self, owner, index, original):
        self.owner, self.index, self.original = owner, index, original

    def __call__(self, x, rotary_pos_emb, temb, attention_mask=None):
        return self.owner._block(
            self.index, self.original, x, rotary_pos_emb, temb, attention_mask
        )


class _ModelView(_View):
    def __init__(self, original, layers):
        super().__init__(original)
        self.layers = layers

    def __call__(self, *args, **kwargs):
        return type(self._original).forward(self, *args, **kwargs)


class ErnieAdapter:
    """Positive-only Ernie adapter; Forge retains native CFG ownership."""
    name = "Ernie Image"
    implementation = "ernie-joint-attention-hybrid-rope-v1"

    def __init__(self, model, negative_context: torch.Tensor, config: NAGConfig, ops, layout_types):
        self.model = model
        self.negative_context = negative_context
        self.config = config
        self.ops = ops
        self.context_width = validate_model(model, layout_types)
        self.targets = tuple(range(len(model.layers)))
        self.attention_calls = 0
        self._seen = {}
        self._negative_text = None
        self._negative_rope = None
        self._image_tokens = None
        self.last_positive_image_pe = None
        self.last_negative_image_pe = None
        self.view = _ModelView(
            model, tuple(_BlockView(self, i, block) for i, block in enumerate(model.layers))
        )

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def _negative_for(self, context):
        if self.negative_context is None:
            raise NAGError("Ernie NAG sampling cache has already been released.")
        negative = self.negative_context.to(
            device=context.device, dtype=context.dtype
        ).expand(context.shape[0], -1, -1)
        if self.model.text_proj is not None and negative.numel() > 0:
            negative = self.model.text_proj(negative)
        return negative

    def _block(self, index, block, positive, positive_rope, temb, attention_mask):
        if attention_mask is not None:
            raise NAGError("Ernie NAG does not support explicit attention masks.")
        negative = self._negative_text
        if negative is None or self._negative_rope is None or self._image_tokens is None:
            raise NAGError("Ernie NAG block executed outside its request-local state.")
        image_tokens = self._image_tokens
        if positive.shape[1] <= image_tokens:
            raise NAGError("Ernie native text token span is empty.")

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = temb

        pos_residual = positive
        pos_norm = block.adaLN_sa_ln(positive)
        pos_norm = (
            pos_norm.float() * (1 + scale_msa.float()) + shift_msa.float()
        ).to(positive.dtype)

        neg_residual = negative
        neg_norm = block.adaLN_sa_ln(negative)
        neg_norm = (
            neg_norm.float() * (1 + scale_msa.float()) + shift_msa.float()
        ).to(negative.dtype)

        pq, pk, pv = _project(
            block.self_attention, pos_norm, positive_rope, self.ops
        )
        nq, nk, nv = _project(
            block.self_attention, neg_norm, self._negative_rope, self.ops
        )

        iq, ik, iv = pq[:, :image_tokens], pk[:, :image_tokens], pv[:, :image_tokens]
        neg_q = torch.cat((iq, nq), dim=1)
        neg_k = torch.cat((ik, nk), dim=1)
        neg_v = torch.cat((iv, nv), dim=1)

        self.last_positive_image_pe = positive_rope[:, :image_tokens].detach()
        self.last_negative_image_pe = positive_rope[:, :image_tokens].detach()

        pos_raw = self.ops.attention(
            pq, pk, pv, block.self_attention.heads, mask=None
        )
        neg_raw = self.ops.attention(
            neg_q, neg_k, neg_v, block.self_attention.heads, mask=None
        )
        guided_image = guide_attention(
            pos_raw[:, :image_tokens], neg_raw[:, :image_tokens],
            phi=self.config.phi, tau=self.config.tau, alpha=self.config.alpha,
        )

        pos_attn = block.self_attention.to_out[0](
            torch.cat((guided_image, pos_raw[:, image_tokens:]), dim=1)
        )
        neg_attn = block.self_attention.to_out[0](neg_raw[:, image_tokens:])

        positive = pos_residual + (
            gate_msa.float() * pos_attn.float()
        ).to(pos_residual.dtype)
        negative = neg_residual + (
            gate_msa.float() * neg_attn.float()
        ).to(neg_residual.dtype)

        pos_residual = positive
        pos_norm = block.adaLN_mlp_ln(positive)
        pos_norm = (
            pos_norm.float() * (1 + scale_mlp.float()) + shift_mlp.float()
        ).to(positive.dtype)
        positive = pos_residual + (
            gate_mlp.float() * block.mlp(pos_norm).float()
        ).to(pos_residual.dtype)

        neg_residual = negative
        neg_norm = block.adaLN_mlp_ln(negative)
        neg_norm = (
            neg_norm.float() * (1 + scale_mlp.float()) + shift_mlp.float()
        ).to(negative.dtype)
        negative = neg_residual + (
            gate_mlp.float() * block.mlp(neg_norm).float()
        ).to(neg_residual.dtype)

        self._negative_text = negative
        self.attention_calls += 1
        self._seen[index] = self._seen.get(index, 0) + 1
        return positive

    @torch.inference_mode()
    def __call__(self, x, timesteps, context, control=None, transformer_options=None, **kwargs):
        if control is not None:
            raise NAGError("Ernie NAG does not support ControlNet.")
        options = transformer_options or {}
        for key in ("patches", "patches_replace", "block_modifiers",
                    "block_inner_modifiers", "attention_override"):
            if options.get(key):
                raise NAGError("Ernie NAG cannot combine with existing transformer patches.")
        if any(value is not None for value in kwargs.values()):
            raise NAGError("Ernie NAG received unsupported extra model conditions.")
        if (x.ndim != 4 or context.ndim != 3 or context.shape[0] != x.shape[0]
                or context.shape[-1] != self.context_width):
            raise NAGError("Unexpected Ernie image/context shape.")
        p = self.model.patch_size
        if x.shape[-2] % p or x.shape[-1] % p:
            raise NAGError("Ernie NAG requires image latent dimensions divisible by patch size.")

        negative = self._negative_for(context)
        neg_len = negative.shape[1]
        neg_ids = torch.zeros(
            x.shape[0], neg_len, 3, dtype=torch.float32, device=x.device
        )
        if neg_len:
            neg_ids[:, :, 0] = torch.linspace(
                0, neg_len - 1, steps=neg_len,
                device=x.device, dtype=torch.float32,
            )
        self._negative_text = negative
        self._negative_rope = self.model.pos_embed(neg_ids)
        self._image_tokens = (x.shape[-2] // p) * (x.shape[-1] // p)
        self._seen = {}

        completed = False
        try:
            result = self.view(
                x, timesteps, context, control=control,
                transformer_options=options, **kwargs
            )
            completed = True
            return result
        finally:
            if completed and (set(self._seen) != set(self.targets)
                              or any(n != 1 for n in self._seen.values())):
                raise NAGError("Ernie NAG did not execute every native block exactly once.")
            self._negative_text = None
            self._negative_rope = None
            self._image_tokens = None
            self._seen = {}

    def memory_reserve(self, x, dtype):
        if x.ndim != 4:
            raise NAGError("Ernie NAG supports four-dimensional image latents only.")
        element = torch.empty((), dtype=dtype).element_size()
        p = self.model.patch_size
        image_tokens = (x.shape[-2] // p) * (x.shape[-1] // p)
        text_tokens = self.negative_context.shape[1]
        features = self.model.hidden_size
        return int(
            x.shape[0] * features * (
                image_tokens * (6 * element + 16)
                + text_tokens * (14 * element + 16)
            )
            + self.negative_context.numel()
            * max(element, self.negative_context.element_size())
        )

    def clear(self):
        self.negative_context = None
        self._negative_text = None
        self._negative_rope = None
        self._image_tokens = None
        self._seen = {}
        self.last_positive_image_pe = None
        self.last_negative_image_pe = None
