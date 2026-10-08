"""Qwen-Image (including 2512) Attention-space NAG for Forge Neo.

Qwen-Image-2.1 uses a different engine and is deliberately excluded. Native
Qwen-Image model.forward handles time, image/reference positions, and output;
request-local block and rotary views introduce the negative text branch. The
initial supported path is text-to-image without edit/reference latents.
"""
from __future__ import annotations

import math
import torch

from ..config import NAGConfig, NAGError
from ..math import guide_attention
from .base import check_unmodified, context_tensor, validate_literal_text

MAX_TOKENS = 2048


def validate_model(model, layout_types):
    block_type, attention_type = layout_types
    required = ("patch_size", "in_channels", "out_channels", "inner_dim",
                "process_img", "pe_embedder", "time_text_embed", "img_in", "txt_norm",
                "txt_in", "transformer_blocks", "norm_out", "proj_out")
    if any(not hasattr(model, attr) for attr in required):
        raise NAGError("Qwen-Image NAG requires the native QwenImageTransformer2DModel layout.")
    if (model.patch_size != 2 or model.out_channels < 1 or model.inner_dim < 1
            or model.in_channels != model.out_channels * 4
            or model.img_in.in_features != model.in_channels
            or model.img_in.out_features != model.inner_dim
            or model.txt_in.out_features != model.inner_dim
            or model.proj_out.in_features != model.inner_dim
            or model.proj_out.out_features != model.out_channels * 4
            or not model.transformer_blocks):
        raise NAGError("Unsupported Qwen-Image transformer dimensions or patch layout.")
    width = model.txt_in.in_features
    if width < 1 or tuple(model.txt_norm.normalized_shape) != (width,):
        raise NAGError("Unexpected Qwen-Image text-conditioning dimensions.")
    for block in model.transformer_blocks:
        if type(block) is not block_type:
            raise NAGError("Qwen-Image NAG requires native QwenImageTransformerBlock modules.")
        check_unmodified(block, "Qwen-Image block")
        attn = block.attn
        if type(attn) is not attention_type:
            raise NAGError("Qwen-Image NAG requires native joint Attention modules.")
        check_unmodified(attn, "Qwen-Image Attention")
        if (attn.heads < 1 or attn.dim_head < 1
                or attn.heads * attn.dim_head != model.inner_dim
                or len(attn.to_out) != 2):
            raise NAGError("Unexpected Qwen-Image head/output layout.")
        for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            layer = getattr(attn, name, None)
            if (layer is None or layer.in_features != model.inner_dim
                    or layer.out_features != model.inner_dim):
                raise NAGError(f"Unexpected Qwen-Image {name} projection layout.")
        if (attn.to_out[0].in_features != model.inner_dim
                or attn.to_out[0].out_features != model.inner_dim
                or attn.to_add_out.in_features != model.inner_dim
                or attn.to_add_out.out_features != model.inner_dim):
            raise NAGError("Unexpected Qwen-Image Attention output layout.")
    return width


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    engine = p.sd_model.text_processing_engine_qwen
    tokens = engine.tokenize(config.negative)
    if not isinstance(tokens, (tuple, list)) or not 0 < len(tokens) <= MAX_TOKENS:
        raise NAGError(f"Qwen-Image NAG negative exceeds {MAX_TOKENS} Qwen2.5-VL tokens; not truncated.")
    prompt = bindings.conditioning_type(
        [config.negative], is_negative_prompt=True, width=p.width, height=p.height,
        distilled_cfg_scale=getattr(p, "distilled_cfg_scale", 1.0),
    )
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1 or not isinstance(encoded[0], torch.Tensor):
        raise NAGError("Qwen-Image NAG requires one native Qwen2.5-VL conditioning tensor.")
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    width = validate_model(model, bindings.layout_types)
    tensor = encoded[0]
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return context_tensor(tensor, width=width, name="Qwen-Image", max_tokens=MAX_TOKENS)


class _View:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)


class _RoPEView:
    def __init__(self, adapter, original):
        self.adapter, self.original = adapter, original

    def __call__(self, ids):
        owner = self.adapter
        if ids.ndim != 3 or ids.shape[2] != 3:
            raise NAGError("Unexpected native Qwen-Image rotary position layout.")
        pos_len = owner._positive_text_len
        if pos_len < 1 or ids.shape[1] <= pos_len or ids.shape[0] != owner._batch:
            raise NAGError("Unexpected Qwen-Image positive text/image token lengths.")
        # Native text positions have identical ascending values on all 3 axes.
        start = ids[:, :1, :]
        expected = start + torch.arange(pos_len, device=ids.device).reshape(1, -1, 1)
        if not torch.equal(ids[:, :pos_len, :], expected.expand(-1, -1, 3)):
            raise NAGError("Qwen-Image text position contract changed in Forge Neo.")
        neg_len = owner._negative_text.shape[1]
        neg_ids = start + torch.arange(neg_len, device=ids.device).reshape(1, -1, 1)
        # Same native EmbedND and same reshape as QwenImageTransformer2DModel.forward.
        owner._negative_rope = self.original(neg_ids).squeeze(1).unsqueeze(2).to(owner._dtype)
        return self.original(ids)


class _ModelView(_View):
    def __init__(self, owner, model):
        super().__init__(model)
        self.owner = owner
        self.pe_embedder = _RoPEView(owner, model.pe_embedder)
        self.transformer_blocks = tuple(_BlockView(owner, i, block)
                                        for i, block in enumerate(model.transformer_blocks))

    def __call__(self, *args, **kwargs):
        return type(self._original).forward(self, *args, **kwargs)


class _BlockView:
    def __init__(self, owner, index, original):
        self.owner, self.index, self.original = owner, index, original

    def __call__(self, hidden_states, encoder_hidden_states,
                 encoder_hidden_states_mask, temb, image_rotary_emb=None):
        return self.owner._block(self.index, self.original, hidden_states,
                                 encoder_hidden_states, encoder_hidden_states_mask,
                                 temb, image_rotary_emb)


class QwenImageAdapter:
    """Positive-only adapter. No shared module mutation, Forge owns CFG."""
    name = "Qwen-Image"
    implementation = "qwen-image-joint-attention-v1"

    def __init__(self, model, negative_context: torch.Tensor, config: NAGConfig,
                 ops, layout_types):
        self.model, self.config, self.ops = model, config, ops
        self.layout_types = layout_types
        self.context_width = validate_model(model, layout_types)
        self.negative_context = negative_context
        self.targets = tuple(range(len(model.transformer_blocks)))
        self._blocks = tuple(model.transformer_blocks)
        self._attentions = tuple(block.attn for block in self._blocks)
        self._block_forward = layout_types[0].forward
        self._attn_forward = layout_types[1].forward
        self.view = _ModelView(self, model)
        self._negative_text = self._negative_rope = None
        self._positive_text_len = self._batch = self._dtype = None
        self._seen = {}
        self.attention_calls = 0

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def _validate_runtime_contract(self):
        current = tuple(self.model.transformer_blocks)
        if (len(current) != len(self._blocks)
                or any(a is not b for a, b in zip(current, self._blocks))
                or any(block.attn is not expected
                       for block, expected in zip(current, self._attentions))
                or self.layout_types[0].forward is not self._block_forward
                or self.layout_types[1].forward is not self._attn_forward):
            raise NAGError("Qwen-Image block/attention changed after NAG setup.")
        if validate_model(self.model, self.layout_types) != self.context_width:
            raise NAGError("Qwen-Image text-conditioning layout changed after setup.")

    def _negative_for(self, context):
        if self.negative_context is None:
            raise NAGError("Qwen-Image NAG sampling cache has already been released.")
        result = self.negative_context.to(device=context.device, dtype=context.dtype)
        result = result.expand(context.shape[0], -1, -1)
        return self.model.txt_in(self.model.txt_norm(result.clone()))

    def _project(self, attn, image, positive, negative, positive_rope):
        b = image.shape[0]
        heads = attn.heads
        project = lambda layer, value: layer(value).unflatten(-1, (heads, -1))
        iq = attn.norm_q(project(attn.to_q, image))
        ik = attn.norm_k(project(attn.to_k, image))
        iv = project(attn.to_v, image)
        pq = attn.norm_added_q(project(attn.add_q_proj, positive))
        pk = attn.norm_added_k(project(attn.add_k_proj, positive))
        pv = project(attn.add_v_proj, positive)
        nq = attn.norm_added_q(project(attn.add_q_proj, negative))
        nk = attn.norm_added_k(project(attn.add_k_proj, negative))
        nv = project(attn.add_v_proj, negative)
        pos_len = positive.shape[1]
        full_q = self.ops.rope(torch.cat((pq, iq), dim=1), positive_rope)
        full_k = self.ops.rope(torch.cat((pk, ik), dim=1), positive_rope)
        full_v = torch.cat((pv, iv), dim=1)
        nr = self._negative_rope
        if nr is None or nr.shape[1] != negative.shape[1]:
            raise NAGError("Qwen-Image negative native RoPE span mismatch.")
        neg_q = torch.cat((self.ops.rope(nq, nr), full_q[:, pos_len:]), dim=1)
        neg_k = torch.cat((self.ops.rope(nk, nr), full_k[:, pos_len:]), dim=1)
        neg_v = torch.cat((nv, full_v[:, pos_len:]), dim=1)
        def flat(tensor):
            return tensor.flatten(start_dim=2)
        return (flat(full_q), flat(full_k), flat(full_v),
                flat(neg_q), flat(neg_k), flat(neg_v))

    def _block(self, index, block, image, positive_text, mask, temb, positive_rope):
        if mask is not None:
            raise NAGError("Qwen-Image NAG does not support attention masks.")
        negative_text = self._negative_text
        if negative_text is None or positive_rope is None:
            raise NAGError("Qwen-Image NAG executed outside its request-local block state.")
        im1, im2 = block.img_mod(temb).chunk(2, dim=-1)
        tx1, tx2 = block.txt_mod(temb).chunk(2, dim=-1)
        img_norm = block.img_norm1(image)
        img_mod, img_gate1 = block._modulate(img_norm, im1)
        pos_norm = block.txt_norm1(positive_text)
        pos_mod, pos_gate1 = block._modulate(pos_norm, tx1)
        neg_norm = block.txt_norm1(negative_text)
        neg_mod, neg_gate1 = block._modulate(neg_norm, tx1)

        attn = block.attn
        pq, pk, pv, nq, nk, nv = self._project(attn, img_mod, pos_mod,
                                               neg_mod, positive_rope)
        positive_raw = self.ops.attention(pq, pk, pv, attn.heads, None)
        negative_raw = self.ops.attention(nq, nk, nv, attn.heads, None)
        pos_len = positive_text.shape[1]
        neg_len = negative_text.shape[1]
        guided_img = guide_attention(
            positive_raw[:, pos_len:], negative_raw[:, neg_len:],
            phi=self.config.phi, tau=self.config.tau, alpha=self.config.alpha,
        )
        image_attention = attn.to_out[1](attn.to_out[0](guided_img))
        positive_attention = attn.to_add_out(positive_raw[:, :pos_len])
        negative_attention = attn.to_add_out(negative_raw[:, :neg_len])
        image = image + img_gate1 * image_attention
        positive_text = positive_text + pos_gate1 * positive_attention
        negative_text = negative_text + neg_gate1 * negative_attention

        img_mod2, img_gate2 = block._modulate(block.img_norm2(image), im2)
        image = torch.addcmul(image, img_gate2, block.img_mlp(img_mod2))
        pos_mod2, pos_gate2 = block._modulate(block.txt_norm2(positive_text), tx2)
        positive_text = torch.addcmul(positive_text, pos_gate2, block.txt_mlp(pos_mod2))
        neg_mod2, neg_gate2 = block._modulate(block.txt_norm2(negative_text), tx2)
        negative_text = torch.addcmul(negative_text, neg_gate2, block.txt_mlp(neg_mod2))
        self._negative_text = negative_text
        self._seen[index] = self._seen.get(index, 0) + 1
        self.attention_calls += 1
        return positive_text, image

    @torch.inference_mode()
    def __call__(self, x, timesteps, context, attention_mask=None, guidance=None,
                 transformer_options=None, control=None, **kwargs):
        self._validate_runtime_contract()
        if control is not None or attention_mask is not None or guidance is not None:
            raise NAGError("Qwen-Image NAG does not support ControlNet, explicit masks, or guidance embeds.")
        if any(value is not None for value in kwargs.values()):
            raise NAGError("Qwen-Image NAG received unsupported extra model conditions.")
        if (x.ndim != 5 or x.shape[2] != 1 or x.shape[1] * 4 != self.model.in_channels
                or context.ndim != 3 or context.shape[0] != x.shape[0]
                or context.shape[-1] != self.context_width):
            raise NAGError("Unexpected Qwen-Image still-image latent/context shape.")
        options = transformer_options or {}
        if not isinstance(options, dict) or any(options.get(key) for key in (
                "patches", "patches_replace", "block_modifiers", "block_inner_modifiers",
                "attention_override", "optimized_attention_override")):
            raise NAGError("Qwen-Image NAG cannot combine with existing transformer patches.")
        if context.shape[1] < 1:
            raise NAGError("Qwen-Image positive text sequence is empty.")
        self._batch, self._dtype, self._positive_text_len = (
            x.shape[0], x.dtype, context.shape[1],
        )
        self._negative_text = self._negative_for(context)
        self._negative_rope = None
        self._seen = {}
        completed = False
        try:
            output = self.view(x, timesteps, context,
                               transformer_options=dict(options))
            completed = True
            return output
        finally:
            incomplete = completed and (set(self._seen) != set(self.targets)
                                        or any(count != 1 for count in self._seen.values()))
            self._negative_text = None
            self._negative_rope = None
            self._positive_text_len = self._batch = self._dtype = None
            self._seen = {}
            if incomplete:
                raise NAGError("Qwen-Image NAG did not execute every native block once.")

    def memory_reserve(self, x, dtype):
        if x.ndim != 5 or x.shape[2] != 1:
            raise NAGError("Qwen-Image NAG supports only 5D single-frame latents.")
        elem = torch.empty((), dtype=dtype).element_size()
        im_tokens = math.ceil(x.shape[-2] / 2) * math.ceil(x.shape[-1] / 2)
        neg_tokens = self.negative_context.shape[1]
        return int(x.shape[0] * self.model.inner_dim * (
            im_tokens * (8 * elem + 16) + neg_tokens * (12 * elem + 16)
        ) + self.negative_context.numel() * max(elem, self.negative_context.element_size()))

    def clear(self):
        self.negative_context = self._negative_text = self._negative_rope = None
        self._seen = {}
