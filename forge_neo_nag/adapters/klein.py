"""Flux.2 Klein adapter using request-local native block replacement hooks.

The native Flux2 model is not monkey-patched. A call-local transformer_options copy
injects one post_input observer plus owned Double/SingleStream block replacements.
Image projections/states are shared between positive and NAG-negative branches;
negative text advances independently through every block.

Reviewed Forge Neo contract: Haoming02/sd-webui-forge-classic neo
831d242d4cf45a2ba1a64c2087ef0a17934a9daf, backend/nn/flux.py,
backend/diffusion_engine/flux2.py and backend/text_processing/klein_engine.py.
"""
from __future__ import annotations

import math

import torch

from ..config import NAGConfig, NAGError
from ..math import guide_attention
from .base import check_unmodified, context_tensor, validate_literal_text

MAX_TOKENS = 2048


def validate_model(model, layout_types):
    double_type, single_type, attention_type = layout_types
    required = (
        "double_blocks", "single_blocks", "txt_in", "img_in", "time_in",
        "pe_embedder", "final_layer", "process_img", "forward_orig",
        "patch_size", "hidden_size", "num_heads", "axes_dim", "txt_ids_dims",
        "global_modulation",
    )
    if any(not hasattr(model, name) for name in required):
        raise NAGError("Klein NAG requires the native Flux2 transformer layout.")
    if not model.double_blocks or not model.single_blocks:
        raise NAGError("Klein NAG requires native DoubleStream and SingleStream blocks.")
    # Current Forge Flux.2 Klein uses one-pixel latent patches, global modulation,
    # four RoPE axes and the text-only fourth axis. These invariants are what let
    # the adapter share image Q/K/V and image RoPE between the two NAG branches.
    if (model.patch_size != 1 or model.hidden_size < 1 or model.num_heads < 1
            or not model.global_modulation
            or tuple(model.axes_dim) != (32, 32, 32, 32)
            or tuple(model.txt_ids_dims) != (3,)):
        raise NAGError("Unsupported Klein / Flux.2 positional or modulation layout.")
    if model.hidden_size % model.num_heads:
        raise NAGError("Klein hidden size must be divisible by the number of heads.")
    for name in ("double_stream_modulation_img", "double_stream_modulation_txt",
                 "single_stream_modulation"):
        if not hasattr(model, name):
            raise NAGError(f"Klein global modulation is missing '{name}'.")
    for block in model.double_blocks:
        if type(block) is not double_type:
            raise NAGError("Klein NAG requires native DoubleStreamBlock modules.")
        check_unmodified(block, "Klein DoubleStreamBlock")
        for attn in (block.img_attn, block.txt_attn):
            if type(attn) is not attention_type:
                raise NAGError("Klein NAG requires native Flux SelfAttention modules.")
            if attn.num_heads != block.num_heads:
                raise NAGError("Klein attention head layout changed.")
    for block in model.single_blocks:
        if type(block) is not single_type:
            raise NAGError("Klein NAG requires native SingleStreamBlock modules.")
        check_unmodified(block, "Klein SingleStreamBlock")
        if block.num_heads != model.num_heads or block.hidden_size != model.hidden_size:
            raise NAGError("Klein SingleStream block layout changed.")
    width = model.txt_in.in_features
    if getattr(model, "txt_norm", None) is not None:
        normalized_shape = tuple(model.txt_norm.normalized_shape)
        if normalized_shape != (width,):
            raise NAGError("Unsupported Klein text normalization width.")
    return width


def encode_negative(p, config, bindings):
    validate_literal_text(config.negative)
    engine = p.sd_model.text_processing_engine_qwen
    tokens = engine.tokenize(config.negative)
    if not isinstance(tokens, (tuple, list)) or len(tokens) > MAX_TOKENS:
        raise NAGError(
            f"Klein NAG negative exceeds {MAX_TOKENS} Qwen tokens including the template; "
            "it was not truncated."
        )
    prompt = bindings.conditioning_type(
        [config.negative], is_negative_prompt=True, width=p.width, height=p.height,
        distilled_cfg_scale=getattr(p, "distilled_cfg_scale", 1.0),
    )
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, (tuple, list)) or len(encoded) != 1:
        raise NAGError("Klein encoder must return one conditioning tensor.")
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    width = validate_model(model, bindings.layout_types)
    return context_tensor(encoded[0].unsqueeze(0) if encoded[0].ndim == 2 else encoded[0],
                          width=width, name="Klein", max_tokens=MAX_TOKENS)


def _qkv(attn, x):
    qkv = attn.qkv(x)
    q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, attn.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k = attn.norm(q, k, v)
    return q, k, v


class KleinAdapter:
    """Positive-only Flux2 adapter. Host wrapper keeps the native CFG equation."""
    name = "Klein"
    implementation = "klein-flux2-joint-attention-v1"

    def __init__(self, model, negative_context: torch.Tensor, config: NAGConfig, ops, layout_types):
        self.model = model
        self.negative_context = negative_context
        self.config = config
        self.ops = ops
        self.context_width = validate_model(model, layout_types)
        self.double_targets = tuple(range(len(model.double_blocks)))
        self.single_targets = tuple(range(len(model.single_blocks)))
        self.attention_calls = 0
        self._negative_text = None
        self._negative_pe = None
        self._negative_raw = None
        self._seen_double = {}
        self._seen_single = {}

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def _negative_for(self, context):
        if self.negative_context is None:
            raise NAGError("Klein NAG sampling cache has already been released.")
        return self.negative_context.to(device=context.device, dtype=context.dtype).expand(
            context.shape[0], -1, -1
        ).clone()

    def _post_input(self, values):
        if self._negative_raw is None or self._negative_text is not None:
            raise NAGError("Klein NAG post_input hook executed out of order.")
        img_ids = values.get("img_ids")
        txt = values.get("txt")
        img = values.get("img")
        if not isinstance(img_ids, torch.Tensor) or not isinstance(txt, torch.Tensor) or not isinstance(img, torch.Tensor):
            raise NAGError("Klein NAG did not receive native Flux2 post_input tensors.")
        if txt.ndim != 3 or img.ndim != 3 or txt.shape[0] != img.shape[0] or img_ids.shape[0] != img.shape[0]:
            raise NAGError("Unexpected Klein post_input batch layout.")

        negative = self._negative_raw
        if getattr(self.model, "txt_norm", None) is not None:
            negative = self.model.txt_norm(negative)
        negative = self.model.txt_in(negative)
        if negative.shape[-1] != self.model.hidden_size:
            raise NAGError("Unexpected Klein projected negative-text width.")
        self._negative_text = negative

        txt_ids = torch.zeros(
            (img.shape[0], negative.shape[1], len(self.model.axes_dim)),
            device=img_ids.device, dtype=img_ids.dtype,
        )
        for dim in self.model.txt_ids_dims:
            if dim < 0 or dim >= txt_ids.shape[-1]:
                raise NAGError("Unsupported Klein text position axis.")
            txt_ids[:, :, dim] = torch.linspace(
                0, negative.shape[1] - 1, steps=negative.shape[1],
                device=img_ids.device, dtype=img_ids.dtype,
            )
        self._negative_pe = self.model.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))
        return values

    def _double_block(self, index, args):
        block = self.model.double_blocks[index]
        img, txt, vec, pe = args["img"], args["txt"], args["vec"], args["pe"]
        mask = args.get("attn_mask")
        options = args.get("transformer_options") or {}
        if mask is not None:
            raise NAGError("Klein NAG does not support attention masks.")
        neg = self._negative_text
        if neg is None or self._negative_pe is None:
            raise NAGError("Klein negative state was not initialized before DoubleStreamBlock.")
        if img.shape[0] != txt.shape[0] or img.shape[0] != neg.shape[0]:
            raise NAGError("Klein positive/negative batch layout changed.")

        if block.modulation:
            img_mod1, img_mod2 = block.img_mod(vec)
            txt_mod1, txt_mod2 = block.txt_mod(vec)
        else:
            try:
                (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec
            except Exception as exc:
                raise NAGError("Klein global modulation layout changed.") from exc

        apply_mod = self.ops.apply_mod
        img_pre = apply_mod(block.img_norm1(img), 1 + img_mod1.scale, img_mod1.shift, None)
        pos_pre = apply_mod(block.txt_norm1(txt), 1 + txt_mod1.scale, txt_mod1.shift, None)
        neg_pre = apply_mod(block.txt_norm1(neg), 1 + txt_mod1.scale, txt_mod1.shift, None)

        iq, ik, iv = _qkv(block.img_attn, img_pre)
        pq, pk, pv = _qkv(block.txt_attn, pos_pre)
        nq, nk, nv = _qkv(block.txt_attn, neg_pre)
        pos_raw = self.ops.attention(
            torch.cat((pq, iq), dim=2), torch.cat((pk, ik), dim=2), torch.cat((pv, iv), dim=2),
            pe=pe, mask=None, transformer_options=options,
        )
        neg_raw = self.ops.attention(
            torch.cat((nq, iq), dim=2), torch.cat((nk, ik), dim=2), torch.cat((nv, iv), dim=2),
            pe=self._negative_pe, mask=None, transformer_options=options,
        )
        pos_len, neg_len = txt.shape[1], neg.shape[1]
        guided_img = guide_attention(
            pos_raw[:, pos_len:], neg_raw[:, neg_len:],
            phi=self.config.phi, tau=self.config.tau, alpha=self.config.alpha,
        )

        img = img + apply_mod(block.img_attn.proj(guided_img), img_mod1.gate, None, None)
        img = img + apply_mod(
            block.img_mlp(apply_mod(block.img_norm2(img), 1 + img_mod2.scale, img_mod2.shift, None)),
            img_mod2.gate, None, None,
        )
        txt = txt + apply_mod(block.txt_attn.proj(pos_raw[:, :pos_len]), txt_mod1.gate, None, None)
        txt = txt + apply_mod(
            block.txt_mlp(apply_mod(block.txt_norm2(txt), 1 + txt_mod2.scale, txt_mod2.shift, None)),
            txt_mod2.gate, None, None,
        )
        neg = neg + apply_mod(block.txt_attn.proj(neg_raw[:, :neg_len]), txt_mod1.gate, None, None)
        neg = neg + apply_mod(
            block.txt_mlp(apply_mod(block.txt_norm2(neg), 1 + txt_mod2.scale, txt_mod2.shift, None)),
            txt_mod2.gate, None, None,
        )
        self._negative_text = self.ops.fp16_fix(neg)
        txt = self.ops.fp16_fix(txt)
        self.attention_calls += 1
        self._seen_double[index] = self._seen_double.get(index, 0) + 1
        return {"img": img, "txt": txt}

    def _single_block(self, index, args):
        block = self.model.single_blocks[index]
        x, vec, pe = args["img"], args["vec"], args["pe"]
        mask = args.get("attn_mask")
        options = args.get("transformer_options") or {}
        if mask is not None:
            raise NAGError("Klein NAG does not support attention masks.")
        neg = self._negative_text
        if neg is None or self._negative_pe is None:
            raise NAGError("Klein negative state was not initialized before SingleStreamBlock.")
        # Native PE length exactly equals positive text + image sequence length.
        # The negative text length is known; infer positive text length from the
        # image length retained after DoubleStream by tracking the original split.
        pos_text_len = self._positive_text_len
        image_len = x.shape[1] - pos_text_len
        if image_len < 1:
            raise NAGError("Klein SingleStream image span is empty.")

        if block.modulation:
            mod, _ = block.modulation(vec)
        else:
            mod = vec
        apply_mod = self.ops.apply_mod
        pos_pre = apply_mod(block.pre_norm(x), 1 + mod.scale, mod.shift, None)
        neg_pre = apply_mod(block.pre_norm(neg), 1 + mod.scale, mod.shift, None)
        pos_linear = block.linear1(pos_pre)
        neg_linear = block.linear1(neg_pre)
        pos_qkv, pos_mlp = torch.split(pos_linear, [3 * block.hidden_size, block.mlp_hidden_dim_first], dim=-1)
        neg_qkv, neg_mlp = torch.split(neg_linear, [3 * block.hidden_size, block.mlp_hidden_dim_first], dim=-1)

        def split_qkv(value):
            q, k, v = value.view(value.shape[0], value.shape[1], 3, block.num_heads, -1).permute(2, 0, 3, 1, 4)
            q, k = block.norm(q, k, v)
            return q, k, v
        pq, pk, pv = split_qkv(pos_qkv)
        nq, nk, nv = split_qkv(neg_qkv)
        neg_len = neg.shape[1]
        iq, ik, iv = pq[:, :, pos_text_len:], pk[:, :, pos_text_len:], pv[:, :, pos_text_len:]
        neg_q = torch.cat((nq, iq), dim=2)
        neg_k = torch.cat((nk, ik), dim=2)
        neg_v = torch.cat((nv, iv), dim=2)
        pos_attn = self.ops.attention(pq, pk, pv, pe=pe, mask=None, transformer_options=options)
        neg_attn = self.ops.attention(neg_q, neg_k, neg_v, pe=self._negative_pe, mask=None, transformer_options=options)
        guided_img = guide_attention(
            pos_attn[:, pos_text_len:], neg_attn[:, neg_len:],
            phi=self.config.phi, tau=self.config.tau, alpha=self.config.alpha,
        )
        pos_attn = torch.cat((pos_attn[:, :pos_text_len], guided_img), dim=1)

        def activate(value):
            if block.yak_mlp:
                return block.mlp_act(value[..., block.mlp_hidden_dim_first // 2:]) * value[..., :block.mlp_hidden_dim_first // 2]
            return block.mlp_act(value)
        pos_out = block.linear2(torch.cat((pos_attn, activate(pos_mlp)), dim=2))
        neg_out = block.linear2(torch.cat((neg_attn[:, :neg_len], activate(neg_mlp)), dim=2))
        x = x + apply_mod(pos_out, mod.gate, None, None)
        neg = neg + apply_mod(neg_out, mod.gate, None, None)
        self._negative_text = self.ops.fp16_fix(neg)
        x = self.ops.fp16_fix(x)
        self.attention_calls += 1
        self._seen_single[index] = self._seen_single.get(index, 0) + 1
        return {"img": x}

    def _owned_options(self, transformer_options):
        options = dict(transformer_options or {})
        if any(options.get(name) for name in ("patches", "patches_replace")):
            raise NAGError("Klein NAG cannot combine with existing Flux block/input patches.")
        patches = {"post_input": [self._post_input]}
        replacements = {}
        for index in self.double_targets:
            replacements[("double_block", index)] = (
                lambda args, extra, index=index: self._double_block(index, args)
            )
        for index in self.single_targets:
            replacements[("single_block", index)] = (
                lambda args, extra, index=index: self._single_block(index, args)
            )
        options["patches"] = patches
        options["patches_replace"] = {"dit": replacements}
        return options

    @torch.inference_mode()
    def __call__(self, x, timestep, context, y=None, guidance=None, control=None,
                 transformer_options=None, **kwargs):
        if control is not None:
            raise NAGError("Klein NAG does not support ControlNet.")
        if kwargs.get("attention_mask") is not None:
            raise NAGError("Klein NAG does not support attention masks.")
        if any(value is not None for key, value in kwargs.items() if key not in ("attention_mask",)):
            raise NAGError("Klein NAG received unsupported extra model conditions.")
        if x.ndim != 4 or context.ndim != 3 or context.shape[0] != x.shape[0] or context.shape[-1] != self.context_width:
            raise NAGError("Unexpected Klein image/context shape.")
        if self._negative_text is not None or self._negative_raw is not None:
            raise NAGError("Nested Klein NAG model calls are not supported.")
        self._negative_raw = self._negative_for(context)
        self._negative_text = None
        self._negative_pe = None
        self._positive_text_len = context.shape[1]
        self._seen_double = {}
        self._seen_single = {}
        completed = False
        try:
            options = self._owned_options(transformer_options)
            result = type(self.model).forward(
                self.model, x, timestep, context, y=y, guidance=guidance,
                control=None, transformer_options=options, **kwargs,
            )
            completed = True
            return result
        finally:
            if completed:
                if (set(self._seen_double) != set(self.double_targets)
                        or any(n != 1 for n in self._seen_double.values())
                        or set(self._seen_single) != set(self.single_targets)
                        or any(n != 1 for n in self._seen_single.values())):
                    raise NAGError("Klein NAG did not execute every native transformer block exactly once.")
            self._negative_raw = None
            self._negative_text = None
            self._negative_pe = None
            self._seen_double = {}
            self._seen_single = {}

    def memory_reserve(self, x, dtype):
        if x.ndim != 4:
            raise NAGError("Klein NAG supports four-dimensional image latents only.")
        element = torch.empty((), dtype=dtype).element_size()
        h = math.ceil(x.shape[-2] / self.model.patch_size)
        w = math.ceil(x.shape[-1] / self.model.patch_size)
        image_tokens = h * w
        text_tokens = self.negative_context.shape[1]
        features = self.model.hidden_size
        # Two attention streams + FP32 normalization + negative text state.
        return int(
            x.shape[0] * features * (image_tokens * (8 * element + 16)
                                     + text_tokens * (10 * element + 16))
            + self.negative_context.numel() * max(element, self.negative_context.element_size())
        )

    def clear(self):
        self.negative_context = None
        self._negative_raw = None
        self._negative_text = None
        self._negative_pe = None
        self._seen_double = {}
        self._seen_single = {}
