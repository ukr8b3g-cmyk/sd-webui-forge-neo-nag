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
import re
from dataclasses import dataclass
from typing import Any, Callable
from types import MethodType, SimpleNamespace

import torch

from .adapters.krea2 import Krea2Adapter, Krea2Ops
from .config import NAGConfig, NAGError
from .adapters.base import BranchLayout, check_unmodified

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
    ops: Any
    adapter_id: str = "krea2"
    layout_types: tuple = ()


def load_bindings(p=None, requested="auto") -> HostBindings:
    # No backend is imported for OFF or any true bypass request.
    from .registry import choose_adapter
    selected = "krea2" if p is None else choose_adapter(p.sd_model, requested)
    try:
        from backend.args import dynamic_args
        from backend.modules.k_model import KModel
        from modules.processing import StableDiffusionProcessingTxt2Img
        from modules.prompt_parser import SdConditioning
        if selected == "krea2":
            from backend.attention import attention_function
            from backend.diffusion_engine.krea import Krea2 as engine_type
            from backend.nn.krea import SingleStreamDiT as dit_type
            from backend.nn.flux import timestep_embedding
            from backend.quant_ops import ck
            from backend.utils import pad_to_patch_size
            ops = Krea2Ops(attention_function, ck.apply_rope, timestep_embedding, pad_to_patch_size)
            layout = ()
        elif selected == "anima":
            from backend.diffusion_engine.anima import Anima as engine_type
            from backend.nn.anima import Anima as dit_type, Block, SelfCrossAttention
            ops, layout = SimpleNamespace(), (Block, SelfCrossAttention)
        elif selected == "klein":
            from backend.diffusion_engine.flux2 import Flux2 as engine_type
            from backend.nn.flux import (IntegratedFluxTransformer2DModel as dit_type,
                                         DoubleStreamBlock, SingleStreamBlock, SelfAttention,
                                         attention, apply_mod, fp16_fix)
            ops = SimpleNamespace(attention=attention, apply_mod=apply_mod, fp16_fix=fp16_fix)
            layout = (DoubleStreamBlock, SingleStreamBlock, SelfAttention)
        else:
            from backend.diffusion_engine.sdxl import StableDiffusionXL as engine_type
            from backend.nn.unet import (IntegratedUNet2DConditionModel as dit_type,
                                         SpatialTransformer, BasicTransformerBlock,
                                         CrossAttention, attention_function)
            ops = SimpleNamespace(attention=attention_function)
            layout = (SpatialTransformer, BasicTransformerBlock, CrossAttention)
    except (ImportError, AttributeError) as exc:
        raise NAGError(
            f"Native Forge Neo {selected} support is unavailable. Select a supported adapter "
            "or disable NAG. / 選択アダプターに対応するForge Neoが必要です。"
        ) from exc
    return HostBindings(engine_type, dit_type, KModel, StableDiffusionProcessingTxt2Img,
                        SdConditioning, dynamic_args, ops, selected, layout)


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
    except (ValueError, TypeError, OverflowError) as exc:
        raise NAGError("CFG Scale must be finite; NAG does not change it.") from exc
    if isinstance(p.cfg_scale, bool) or not math.isfinite(cfg):
        raise NAGError("CFG Scale must be a finite number; NAG does not change it.")
    if not isinstance(p.sd_model, bindings.engine_type):
        raise NAGError(f"Selected {bindings.engine_type.__name__} / {dict(krea2='Krea2', anima='Anima', sdxl='SDXL', klein='Klein')[bindings.adapter_id]} adapter is incompatible with the loaded engine. Choose a compatible adapter or turn NAG OFF. / アダプターを変更するかNAGをOFFにしてください。")
    if bindings.adapter_id in ("anima", "sdxl", "klein"):
        prompts = getattr(p, "all_prompts", None) or getattr(p, "prompt", "")
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if any(re.search(r"(?<!\w)AND(?!\w)", text) for text in prompts if isinstance(text, str)):
            raise NAGError("Positive AND composition is not supported by this NAG adapter.")
    if bindings.adapter_id == "sdxl" and getattr(p.sd_model, "use_shift", False):
        raise NAGError("Rectified-Flow SDXL is not supported by this adapter.")
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
        raise NAGError(f"NAG requires native, uncompiled {bindings.dit_type.__name__} for {bindings.adapter_id}.")
    check_unmodified(model, f"{bindings.adapter_id} forward")
    if bindings.adapter_id == "krea2":
        required = ("first", "last", "blocks", "txtfusion", "txtmlp", "tmlp", "tproj", "pe_embedder", "patch", "channels", "tdim", "txtlayers", "txtdim")
        if any(not hasattr(model, attr) for attr in required) or not len(model.blocks):
            raise NAGError("The installed Krea2 model layout is not supported by NAG.")
        for block in model.blocks:
            check_unmodified(block, "Krea2 block")
    elif bindings.adapter_id == "anima":
        from .adapters.anima import validate_model
        validate_model(model, bindings.layout_types)
    elif bindings.adapter_id == "sdxl":
        from .adapters.sdxl import enumerate_attn2
        enumerate_attn2(model, bindings.layout_types)
    elif bindings.adapter_id == "klein":
        from .adapters.klein import validate_model
        validate_model(model, bindings.layout_types)
    else:
        raise NAGError("Unknown adapter binding.")
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


def validate_transformer_options(options: dict, adapter=None) -> None:
    if not isinstance(options, dict):
        raise NAGError("Unsupported transformer_options container.")
    for key in ("patches", "patches_replace", "block_modifiers", "block_inner_modifiers", "attention_override"):
        if key == "patches_replace" and adapter is not None and hasattr(adapter, "validate_owned"):
            adapter.validate_owned(options)
        elif _has_value(options.get(key)):
            raise NAGError(f"NAG cannot safely combine with transformer option '{key}'.")
    for key in ("attention_mask", "attn_mask", "mask"):
        if options.get(key) is not None:
            raise NAGError("NAG does not support additional attention masks.")


@torch.inference_mode()
def encode_negative(p, config: NAGConfig, bindings: HostBindings) -> torch.Tensor:
    if bindings.adapter_id == "sdxl":
        from .adapters.sdxl import encode_negative as encode
        return encode(p, config, bindings)
    if bindings.adapter_id == "anima":
        from .adapters.anima import encode_negative as encode
        return encode(p, config, bindings)
    if bindings.adapter_id == "klein":
        from .adapters.klein import encode_negative as encode
        return encode(p, config, bindings)
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
    """Preserve native CFG; guide conditional rows only, regardless of order.

    SDXL/Anima keep the native batched UNet/DiT call. Krea2's existing positive-
    only adapter is unchanged; a mixed CFG call is split by labelled branch.
    The host, not this extension, performs final CFG combination.
    """
    def __init__(self, original, adapter, config: NAGConfig, bindings: HostBindings):
        self.original, self.adapter, self.config, self.bindings = original, adapter, config, bindings
        self.calls = self.active_calls = 0
        self._apply_model_fn = bindings.kmodel_type.apply_model
        self._forward_fn = type(adapter.model).forward

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def _native_conditions(self, conditions):
        if not hasattr(self.adapter, "without_patches"):
            return conditions
        result = dict(conditions)
        result["transformer_options"] = self.adapter.without_patches(conditions["transformer_options"])
        return result

    def _guided_model(self, x, sigma, conditions):
        view = _KModelView(self.original, self.adapter)
        return self._apply_model_fn(view, x, sigma, **conditions)

    def _krea_call(self, apply_model, x, sigma, conditions, layout):
        if len(layout.positive_rows) == x.shape[0]:
            return self._guided_model(x, sigma, conditions)
        outputs = []
        for index, label in enumerate(layout.labels):
            part = slice(index * layout.chunk_size, (index + 1) * layout.chunk_size)
            c = dict(conditions)
            for key, value in c.items():
                if isinstance(value, torch.Tensor):
                    if value.ndim < 1 or value.shape[0] != x.shape[0]:
                        raise NAGError(f"Cannot safely split CFG condition '{key}'.")
                    c[key] = value[part]
            options = dict(c.get("transformer_options", {}))
            options["cond_or_uncond"] = [label]
            options["cond_indices"] = list(range(layout.chunk_size)) if label == 0 else []
            options["uncond_indices"] = list(range(layout.chunk_size)) if label == 1 else []
            if isinstance(options.get("cond_mark"), torch.Tensor):
                options["cond_mark"] = options["cond_mark"][part]
            c["transformer_options"] = options
            level = sigma if sigma.numel() == 1 else sigma[part]
            outputs.append(self._guided_model(x[part], level, c) if label == 0
                           else apply_model(x[part], level, **c))
        return torch.cat(outputs, dim=0)

    @torch.inference_mode()
    def __call__(self, apply_model, payload):
        if (getattr(apply_model, "__self__", None) is not self.original
                or getattr(apply_model, "__func__", None) is not self._apply_model_fn):
            raise NAGError("KModel.apply_model changed during sampling; NAG refuses to bypass it.")
        if (self.original.diffusion_model is not self.adapter.model
                or type(self.adapter.model).forward is not self._forward_fn):
            raise NAGError("The model changed during NAG sampling.")
        check_unmodified(self.adapter.model, "active model forward")
        if _has_value(getattr(self.bindings.dynamic_args, "ref_latents", None)):
            raise NAGError("Reference/Edit appeared during sampling; NAG cannot process it.")
        x, sigma = payload["input"], payload["timestep"]
        conditions = dict(payload["c"])
        options = conditions.get("transformer_options", {})
        validate_transformer_options(options, self.adapter)
        # Native UNet updates diagnostics in this dict. Keep them call-local.
        conditions["transformer_options"] = dict(options)
        layout = BranchLayout.from_payload(payload, x.shape[0])
        if conditions.get("control") is not None or conditions.get("c_concat") is not None:
            raise NAGError("NAG does not support extra concatenated or control conditions.")
        allowed = {"c_crossattn", "transformer_options", "control", "c_concat"}
        if self.bindings.adapter_id == "sdxl":
            allowed.add("y")
        if self.bindings.adapter_id == "klein":
            allowed.update(("y", "guidance"))
        if any(key not in allowed and value is not None for key, value in conditions.items()):
            raise NAGError("NAG received unsupported extra model conditions.")
        if sigma.numel() == 0 or sigma.numel() not in (1, x.shape[0]):
            raise NAGError("Unexpected sigma batch layout.")
        levels = sigma.detach().float().reshape(-1).cpu().tolist()
        if not all(math.isfinite(level) for level in levels):
            raise NAGError("Sampling sigma is not finite.")
        active = [self.config.sigma_end <= level <= self.config.sigma_start for level in levels]
        self.calls += 1
        # No negative K/V or adapter model path for a native unconditional call.
        if not layout.positive_rows or not any(active):
            result = apply_model(x, sigma, **self._native_conditions(conditions))
            if not bool(torch.isfinite(result).all()):
                raise NAGError("Native CFG branch output is not finite; generation rejected.")
            return result
        if not all(active):
            raise NAGError("NAG cannot mix in-range and out-of-range sigma rows in one model call.")
        if self.bindings.adapter_id in ("krea2", "klein"):
            result = self._krea_call(apply_model, x, sigma, conditions, layout)
        else:
            self.adapter.begin_call(layout.positive_rows, x.shape[0], x.device)
            completed = False
            try:
                result = (apply_model(x, sigma, **conditions) if self.bindings.adapter_id == "sdxl"
                          else self._guided_model(x, sigma, conditions))
                completed = True
            finally:
                self.adapter.end_call(completed=completed)
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


def _conditioning_stamp(p, bindings):
    objects = p.sd_model.forge_objects
    clip = getattr(objects, "clip", None)
    clip_patcher = getattr(clip, "patcher", None)
    engines = tuple((id(getattr(p.sd_model, name, None)),
                     getattr(getattr(p.sd_model, name, None), "clip_skip", None))
                    for name in ("text_processing_engine_l", "text_processing_engine_g",
                                 "text_processing_engine_qwen"))
    return (id(p.sd_model), id(objects.unet.model.diffusion_model),
            getattr(objects.unet, "patches_uuid", None), id(clip),
            getattr(clip_patcher, "patches_uuid", None), engines)


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
        self.conditioning_stamp = _conditioning_stamp(p, bindings)
        self.objects = None

    def install(self, p, **kwargs):
        if p is not self.p or self.wrapper is not None:
            raise NAGError("Nested or repeated NAG sampling passes are not supported in V1.")
        validate_request(p, self.bindings)
        patcher = p.sd_model.forge_objects.unet
        model = validate_patcher(patcher, self.bindings)
        if _conditioning_stamp(p, self.bindings) != self.conditioning_stamp:
            raise NAGError("Model/CLIP/LoRA changed after NAG negative encoding; refusing stale conditions.")
        if "x" not in kwargs:
            raise NAGError("Forge's pre-sampling hook did not supply an image latent.")
        if self.bindings.adapter_id == "krea2":
            adapter = Krea2Adapter(model, self.negative_context, self.config, self.bindings.ops)
            reserve = memory_reserve(model, kwargs["x"], self.negative_context, patcher.model.computation_dtype)
            if float(p.cfg_scale) != 1.0:
                reserve *= 2  # native cond/uncond batching may double the call batch
        else:
            if self.bindings.adapter_id == "anima":
                from .adapters.anima import AnimaAdapter as AdapterClass
            elif self.bindings.adapter_id == "klein":
                from .adapters.klein import KleinAdapter as AdapterClass
            else:
                from .adapters.sdxl import SDXLAdapter as AdapterClass
            adapter = AdapterClass(model, self.negative_context, self.config, self.bindings.ops,
                                   self.bindings.layout_types)
            reserve = adapter.memory_reserve(kwargs["x"], patcher.model.computation_dtype)
            if self.bindings.adapter_id == "klein" and float(p.cfg_scale) != 1.0:
                reserve *= 2
        wrapper = NAGModelWrapper(patcher.model, adapter, self.config, self.bindings)
        clone = patcher.clone()
        if clone is patcher or clone.model_options is patcher.model_options:
            raise NAGError("Forge Patcher clone did not isolate model_options.")
        if clone.model_options.get("transformer_options") is patcher.model_options.get("transformer_options"):
            raise NAGError("Forge Patcher clone did not isolate transformer_options.")
        if hasattr(adapter, "install"):
            adapter.install(clone)
        clone.set_model_unet_function_wrapper(wrapper)
        clone.add_extra_preserved_memory_during_sampling(reserve)
        self.original_patcher = patcher
        self.installed_patcher = clone
        self.wrapper = wrapper
        self.reserve_bytes = reserve
        self.objects = p.sd_model.forge_objects
        self.objects.unet = clone

    def check_completed(self):
        if self.wrapper is None or self.wrapper.calls == 0:
            raise NAGError("NAG was not executed by this sampling path. Generation was rejected rather than saved as NAG.")
        if self.wrapper.active_calls == 0:
            raise NAGError("No sampled sigma was inside the NAG interval. Widen Sigma Start/End and retry.")

    def metadata(self):
        adapter = self.wrapper.adapter
        data = self.config.metadata(model=adapter.name)
        data.update({
            "Forge NAG Adapter": self.bindings.adapter_id,
            "Forge NAG Status": "applied",
            "Forge NAG Active Calls": self.wrapper.active_calls,
            "Forge NAG CFG": self.p.cfg_scale,
            "Forge NAG Implementation": getattr(adapter, "implementation", "shared-image-qkv-v1"),
        })
        if self.bindings.adapter_id == "krea2":
            data["Forge NAG Text Fusion Calls"] = adapter.text_fusion_calls
        elif self.bindings.adapter_id == "klein":
            data["Forge NAG Joint Attention Calls"] = adapter.attention_calls
        else:
            data["Forge NAG Cross Attention Calls"] = adapter.attention_calls
        return data

    def close(self):
        if self.installed_patcher is not None:
            objects = self.objects
            if objects.unet is self.installed_patcher:
                objects.unet = self.original_patcher
        if self.wrapper is not None:
            self.wrapper.adapter.clear()
        self.negative_context = None
        self.installed_patcher = None
        self.original_patcher = None
        self.objects = None


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
    def __init__(self, p, original, raw_args, bindings_loader, preset=None):
        self.p = p
        self.original = original
        self.raw_args = raw_args
        self.bindings_loader = bindings_loader
        self.preset = preset

    def __call__(self, *args, **kwargs):
        p, original = self.p, self.original
        config = NAGConfig.parse(*self.raw_args)
        if not config.active:
            if config.enabled:
                p.extra_generation_params.update(config.metadata())
                p.extra_generation_params["Forge NAG Status"] = f"bypassed ({config.bypass_reason})"
                LOGGER.info("NAG bypassed: %s", config.bypass_reason)
            return original(*args, **kwargs)
        bindings = self.bindings_loader() if self.bindings_loader is not None else load_bindings(p, config.adapter)
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
            if isinstance(self.preset, str):
                p.extra_generation_params["Forge NAG UI Preset"] = self.preset
            LOGGER.info("%s NAG applied: selection=%s, CFG=%s, calls=%d, active=%d, reserve=%.1f MiB",
                        session.wrapper.adapter.name, config.adapter, p.cfg_scale,
                        session.wrapper.calls, session.wrapper.active_calls, session.reserve_bytes / 2**20)
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


def arm_request(p, raw_args: tuple, bindings_loader: Callable | None = None, *, preset=None):
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
    p.sample = _RequestSampleGuard(p, original, raw_args, bindings_loader, preset)


def disarm_request(p):
    sample = p.sample
    if isinstance(sample, _RequestSampleGuard):
        p.sample = _rebind_sample(sample.original, p)
