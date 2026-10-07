"""Request-scoped Forge integration without shared-model monkey patches.

A processing object's sample method is guarded, not Forge's global classes.
The request's ScriptRunner is delegated through a temporary view so our final
pre-sampling validation runs *outside* Forge's exception-swallowing Script loop.
The host KModel.apply_model is reused with a read-only forwarding view, keeping
its predictor/sigma conversions intact while substituting only the DiT call.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable
from types import MethodType

import torch

from .adapters.krea2 import Krea2Adapter, Krea2Ops
from .config import NAGConfig, NAGError

LOGGER = logging.getLogger("forge_neo_nag")
MAX_NEGATIVE_TOKENS = 2048


@dataclass(frozen=True)
class HostBindings:
    engine_type: type
    dit_type: type
    kmodel_type: type
    txt2img_type: type
    conditioning_type: Callable
    dynamic_args: Any
    ops: Krea2Ops


def load_bindings() -> HostBindings:
    # No backend is imported unless a request actually enables nonzero NAG.
    try:
        from backend.args import dynamic_args
        from backend.attention import attention_function
        from backend.diffusion_engine.krea import Krea2
        from backend.modules.k_model import KModel
        from backend.nn.krea import SingleStreamDiT
        from backend.nn.flux import timestep_embedding
        from backend.quant_ops import ck
        from backend.utils import pad_to_patch_size
        from modules.processing import StableDiffusionProcessingTxt2Img
        from modules.prompt_parser import SdConditioning
    except (ImportError, AttributeError) as exc:
        raise NAGError(
            "Forge Neo with native Krea2 support is required. / Krea2対応のForge Neoが必要です。"
        ) from exc
    return HostBindings(Krea2, SingleStreamDiT, KModel, StableDiffusionProcessingTxt2Img,
                        SdConditioning, dynamic_args,
                        Krea2Ops(attention_function, ck.apply_rope, timestep_embedding, pad_to_patch_size))


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return bool(value)


def validate_request(p, bindings: HostBindings) -> None:
    if not isinstance(p, bindings.txt2img_type):
        raise NAGError("NAG V1 supports txt2img only. / V1はtxt2imgのみ対応です。")
    try:
        cfg = float(p.cfg_scale)
    except (ValueError, TypeError) as exc:
        raise NAGError("NAG requires CFG Scale = 1.0.") from exc
    if isinstance(p.cfg_scale, bool) or not math.isfinite(cfg) or cfg != 1.0:
        raise NAGError(f"NAG requires CFG Scale = 1.0 (current: {p.cfg_scale}). / CFGを1.0にしてください。")
    if not isinstance(p.sd_model, bindings.engine_type):
        raise NAGError("NAG V1 supports Krea2 only. Disable NAG for other models. / V1はKrea2専用です。")
    if getattr(p, "enable_hr", False) or getattr(p, "is_hr_pass", False) or getattr(p, "txt2img_upscale", False):
        raise NAGError("NAG V1 does not support Hires fix. / V1ではHires fixをOFFにしてください。")
    if getattr(p, "refiner_checkpoint", None) not in (None, "", "None", "none"):
        raise NAGError("NAG V1 does not support refiner switching. / Refinerは非対応です。")
    if "cfg++" in str(getattr(p, "sampler_name", "")).lower():
        raise NAGError("NAG V1 does not support CFG++ samplers. / CFG++は非対応です。")
    if getattr(p, "tiling", False):
        raise NAGError("NAG V1 does not support tiling. / Tilingは非対応です。")
    ratio = p.get_token_merging_ratio() if hasattr(p, "get_token_merging_ratio") else 0
    if ratio:
        raise NAGError("NAG V1 does not support token merging. / Token mergingは非対応です。")
    if _has_value(getattr(bindings.dynamic_args, "ref_latents", None)):
        raise NAGError("NAG V1 does not support Reference/Edit. / Reference・Editは非対応です。")
    if _has_value(getattr(bindings.dynamic_args, "context_handler", None)):
        raise NAGError("NAG V1 does not support temporal/context handlers.")


def validate_patcher(patcher, bindings: HostBindings):
    kmodel = patcher.model
    if type(kmodel) is not bindings.kmodel_type:
        raise NAGError("Unsupported KModel replacement; NAG cannot preserve its predictor behavior.")
    model = kmodel.diffusion_model
    if type(model) is not bindings.dit_type or hasattr(model, "_orig_mod"):
        raise NAGError("NAG V1 requires native, uncompiled Krea2 SingleStreamDiT.")
    if "forward" in vars(model) or getattr(model, "_forward_hooks", {}) or getattr(model, "_forward_pre_hooks", {}):
        raise NAGError("Another extension has patched Krea2 forward; disable it before using NAG.")
    required = ("first", "last", "blocks", "txtfusion", "txtmlp", "tmlp", "tproj", "pe_embedder", "patch", "channels", "tdim", "txtlayers", "txtdim")
    if any(not hasattr(model, attr) for attr in required):
        raise NAGError("The installed Krea2 model layout is not supported by NAG V1.")
    if not len(model.blocks):
        raise NAGError("Krea2 has no transformer blocks.")
    for block in model.blocks:
        if "forward" in vars(block) or getattr(block, "_forward_hooks", {}) or getattr(block, "_forward_pre_hooks", {}):
            raise NAGError("Another extension has patched a Krea2 block; NAG refuses to bypass it.")
    if getattr(patcher, "controlnet_linked_list", None) is not None:
        raise NAGError("NAG V1 does not support ControlNet. / V1はControlNet非対応です。")
    # Unknown object patches might change model objects only at sampling_prepare.
    if getattr(patcher, "object_patches", {}):
        raise NAGError("NAG V1 cannot be combined with model object patches.")
    options = patcher.model_options
    conflicts = ("model_function_wrapper", "sampler_cfg_function", "sampler_pre_cfg_function",
                 "sampler_post_cfg_function", "conditioning_modifiers", "disable_cfg1_optimization",
                 "sampler_calc_cond_batch_function")
    if any(_has_value(options.get(key)) for key in conflicts):
        raise NAGError("Another sampling/guidance wrapper is active; disable it before enabling NAG.")
    validate_transformer_options(options.get("transformer_options", {}))
    return model


def validate_transformer_options(options: dict) -> None:
    for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "attention_override"):
        if _has_value(options.get(key)):
            raise NAGError(f"NAG V1 cannot safely combine with transformer option '{key}'.")


@torch.inference_mode()
def encode_negative(p, config: NAGConfig, bindings: HostBindings) -> torch.Tensor:
    engine = p.sd_model.text_processing_engine_qwen
    tokens = engine.tokenize(config.negative)
    if len(tokens) > MAX_NEGATIVE_TOKENS:
        raise NAGError(f"NAG negative exceeds {MAX_NEGATIVE_TOKENS} tokens including the template; it was not truncated.")
    prompt = bindings.conditioning_type([config.negative], is_negative_prompt=True,
                                       width=p.width, height=p.height,
                                       distilled_cfg_scale=getattr(p, "distilled_cfg_scale", 1.0))
    # The negative flag avoids consuming/resetting Forge's reference-image state.
    encoded = p.sd_model.get_learned_conditioning(prompt)
    if not isinstance(encoded, (list, tuple)) or len(encoded) != 1 or not isinstance(encoded[0], torch.Tensor):
        raise NAGError("Unexpected Krea2 encoder result; expected one text conditioning tensor.")
    context = encoded[0].detach().to(device="cpu").contiguous()
    model = p.sd_model.forge_objects.unet.model.diffusion_model
    if context.ndim != 3 or tuple(context.shape[1:]) != (model.txtlayers, model.txtdim) or not context.shape[0]:
        raise NAGError("Unexpected Krea2 negative conditioning shape.")
    if not context.is_floating_point() or not bool(torch.isfinite(context).all()):
        raise NAGError("Negative conditioning contains non-finite or non-floating values.")
    return context.unsqueeze(0)


class _KModelView:
    """Read-only attribute delegation; never assigns anything to the shared model."""
    def __init__(self, original, diffusion_call):
        self._original = original
        self.diffusion_model = diffusion_call

    def __getattr__(self, name):
        return getattr(self._original, name)


class NAGModelWrapper:
    def __init__(self, original, adapter: Krea2Adapter, config: NAGConfig, bindings: HostBindings):
        self.original = original
        self.adapter = adapter
        self.config = config
        self.bindings = bindings
        self.calls = 0
        self.active_calls = 0

    def to(self, *args, **kwargs):
        # Forge may move wrapper objects. Tensors are cast lazily at the DiT
        # boundary; never move or dequantize the shared model's weights here.
        return self

    def __deepcopy__(self, memo):
        return self

    @torch.inference_mode()
    def __call__(self, apply_model, payload):
        if getattr(apply_model, "__self__", None) is not self.original or getattr(apply_model, "__func__", None) is not self.bindings.kmodel_type.apply_model:
            raise NAGError("KModel.apply_model changed during sampling; NAG refuses to bypass it.")
        if _has_value(getattr(self.bindings.dynamic_args, "ref_latents", None)):
            raise NAGError("Reference/Edit appeared during sampling; NAG V1 cannot process it.")
        x, sigma = payload["input"], payload["timestep"]
        conditions = dict(payload["c"])
        options = conditions.get("transformer_options", {})
        validate_transformer_options(options)
        if any(branch != 0 for branch in payload.get("cond_or_uncond", [0])) or any(
                branch != 0 for branch in options.get("cond_or_uncond", [0])):
            raise NAGError("NAG received an unconditional CFG branch; use CFG=1 without other guidance extensions.")
        if conditions.get("control") is not None or conditions.get("c_concat") is not None:
            raise NAGError("NAG V1 does not support extra concatenated or control conditions.")
        if sigma.numel() == 0 or sigma.numel() not in (1, x.shape[0]):
            raise NAGError("Unexpected sigma batch layout.")
        # One small host transfer per denoiser call, never a GPU sync per block.
        levels = sigma.detach().float().reshape(-1).cpu().tolist()
        if not all(math.isfinite(level) for level in levels):
            raise NAGError("Sampling sigma is not finite.")
        active = [self.config.sigma_end <= level <= self.config.sigma_start for level in levels]
        self.calls += 1
        if not any(active):
            return apply_model(x, sigma, **conditions)
        if not all(active):
            raise NAGError("NAG V1 cannot mix in-range and out-of-range sigma rows in one model call.")

        # This is the host's own method, not a reimplementation of sigma/input
        # scaling. The only substituted attribute is the diffusion callable.
        view = _KModelView(self.original, self.adapter)
        result = self.bindings.kmodel_type.apply_model(view, x, sigma, **conditions)
        if not bool(torch.isfinite(result).all()):
            raise NAGError("NAG output is not finite. Reduce Phi/Alpha or check model precision; no image was saved by this call.")
        self.active_calls += 1
        return result


def memory_reserve(model, x: torch.Tensor, negative_context: torch.Tensor, dtype) -> int:
    """Conservative additional activation estimate, not a measured VRAM promise."""
    if x.ndim not in (4, 5) or (x.ndim == 5 and x.shape[2] != 1):
        raise NAGError("NAG V1 supports 4D or single-frame 5D image latents only.")
    element = torch.empty((), dtype=dtype).element_size()
    image_tokens = math.ceil(x.shape[-2] / model.patch) * math.ceil(x.shape[-1] / model.patch)
    features = model.first.out_features
    batch = x.shape[0]
    negative_tokens = negative_context.shape[1]
    return int(batch * features * (image_tokens * (8 * element + 16) + negative_tokens * 12 * element)
               + negative_context.numel() * negative_context.element_size())


class SamplingSession:
    def __init__(self, p, config: NAGConfig, bindings: HostBindings, negative_context):
        self.p = p
        self.config = config
        self.bindings = bindings
        self.negative_context = negative_context
        self.wrapper = None
        self.original_patcher = None
        self.installed_patcher = None
        self.reserve_bytes = 0

    def install(self, p, **kwargs):
        if p is not self.p or self.wrapper is not None:
            raise NAGError("Nested or repeated NAG sampling passes are not supported in V1.")
        validate_request(p, self.bindings)
        patcher = p.sd_model.forge_objects.unet
        model = validate_patcher(patcher, self.bindings)
        if "x" not in kwargs:
            raise NAGError("Forge's pre-sampling hook did not supply an image latent.")
        reserve = memory_reserve(model, kwargs["x"], self.negative_context, patcher.model.computation_dtype)
        adapter = Krea2Adapter(model, self.negative_context, self.config, self.bindings.ops)
        wrapper = NAGModelWrapper(patcher.model, adapter, self.config, self.bindings)
        clone = patcher.clone()
        if clone is patcher or clone.model_options is patcher.model_options:
            raise NAGError("Forge Patcher clone did not isolate model_options.")
        clone.set_model_unet_function_wrapper(wrapper)
        clone.add_extra_preserved_memory_during_sampling(reserve)
        self.original_patcher = patcher
        self.installed_patcher = clone
        self.wrapper = wrapper
        self.reserve_bytes = reserve
        p.sd_model.forge_objects.unet = clone

    def check_completed(self):
        if self.wrapper is None or self.wrapper.calls == 0:
            raise NAGError("NAG was not executed by this sampling path. Generation was rejected rather than saved as NAG.")
        if self.wrapper.active_calls == 0:
            raise NAGError("No sampled sigma was inside the NAG interval. Widen Sigma Start/End and retry.")

    def metadata(self):
        data = self.config.metadata()
        data.update({
            "Forge NAG Status": "applied",
            "Forge NAG Active Calls": self.wrapper.active_calls,
            "Forge NAG Text Fusion Calls": self.wrapper.adapter.text_fusion_calls,
            "Forge NAG Implementation": "shared-image-qkv-v1",
        })
        return data

    def close(self):
        if self.installed_patcher is not None:
            objects = self.p.sd_model.forge_objects
            if objects.unet is self.installed_patcher:
                objects.unet = self.original_patcher
        if self.wrapper is not None:
            self.wrapper.adapter.clear()
        self.negative_context = None
        self.installed_patcher = None
        self.original_patcher = None


class _SamplingScriptsView:
    def __init__(self, original, session: SamplingSession):
        self.original = original
        self.session = session

    def __getattr__(self, name):
        return getattr(self.original, name)

    def process_before_every_sampling(self, p, **kwargs):
        self.original.process_before_every_sampling(p, **kwargs)
        # Deliberately outside the host ScriptRunner's try/except. A rejected
        # configuration must stop sampling, not fall back to an ordinary image.
        self.session.install(p, **kwargs)


class _RequestSampleGuard:
    """An identifiable callable, so functools.wraps cannot counterfeit ownership."""
    def __init__(self, p, original, raw_args, bindings_loader):
        self.p = p
        self.original = original
        self.raw_args = raw_args
        self.bindings_loader = bindings_loader

    def __call__(self, *args, **kwargs):
        p, original = self.p, self.original
        config = NAGConfig.parse(*self.raw_args)
        if not config.active:
            if config.enabled:
                p.extra_generation_params.update(config.metadata())
                p.extra_generation_params["Forge NAG Status"] = f"bypassed ({config.bypass_reason})"
                LOGGER.info("NAG bypassed: %s", config.bypass_reason)
            return original(*args, **kwargs)
        bindings = self.bindings_loader()
        validate_request(p, bindings)
        validate_patcher(p.sd_model.forge_objects.unet, bindings)
        if p.scripts is None:
            raise NAGError("The Forge processing ScriptRunner is unavailable.")
        negative = encode_negative(p, config, bindings)
        session = SamplingSession(p, config, bindings, negative)
        old_scripts = p.scripts
        view = _SamplingScriptsView(old_scripts, session)
        p.scripts = view
        try:
            result = original(*args, **kwargs)
            session.check_completed()
            p.extra_generation_params.update(session.metadata())
            LOGGER.info("Krea2 NAG applied: model calls=%d, active=%d, negative text fusion=%d, reserve=%.1f MiB",
                        session.wrapper.calls, session.wrapper.active_calls,
                        session.wrapper.adapter.text_fusion_calls, session.reserve_bytes / 2**20)
            return result
        finally:
            if p.scripts is view:
                p.scripts = old_scripts
            session.close()
            negative = None


def _rebind_sample(sample, p):
    # Selected WebUI scripts can shallow-copy a processing object. A copied
    # instance attribute holding a bound method otherwise still targets the
    # old processing object. Rebind only an actual method, never a foreign
    # callable/closure whose ownership cannot be established.
    if isinstance(sample, MethodType) and sample.__self__ is not p:
        return MethodType(sample.__func__, p)
    return sample


def arm_request(p, raw_args: tuple, bindings_loader: Callable = load_bindings):
    """Install only a processing-instance guard. No global ScriptRunner mutation."""
    original = p.sample
    if isinstance(original, _RequestSampleGuard):
        original = _rebind_sample(original.original, p)
        p.sample = original
    for key in list(p.extra_generation_params):
        if key.startswith("Forge NAG"):
            del p.extra_generation_params[key]
    # Leave an ordinary OFF request's sample path unwrapped and unencoded.
    if raw_args and (raw_args[0] is False or raw_args[0] == "false" or raw_args[0] == "False"):
        return
    p.sample = _RequestSampleGuard(p, original, raw_args, bindings_loader)


def disarm_request(p):
    sample = p.sample
    if isinstance(sample, _RequestSampleGuard):
        p.sample = _rebind_sample(sample.original, p)
