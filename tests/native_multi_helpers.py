"""Load reviewed native model definitions for optional, reduced CPU probes.

Set FORGE_NAG_TEST_SOURCE to a local Forge Neo checkout. Native model methods
are evaluated with CPU SDPA/offload/RoPE test operators and random tiny weights.
This does NOT load Forge, tokenizers, checkpoints or any CUDA kernel. No network.
"""
from __future__ import annotations

import ast
import contextlib
import functools
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Optional

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from forge_neo_nag.host import HostBindings, SamplingSession
from forge_neo_nag.config import NAGConfig
from .helpers import Patcher, Conditioning, Predictor, Runner


def attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options=None):
    if not skip_reshape:
        q, k, v = [t.unflatten(-1, (heads, -1)).transpose(1, 2) for t in (q, k, v)]
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def pad(x, sizes):
    pads = tuple(n for size, patch in zip(reversed(x.shape), reversed(sizes))
                 for n in (0, (-size) % patch))
    return F.pad(x, pads, mode="replicate")


def rms_rope(q, k, rope, qw, kw, eps):
    # Portable oracle, not Forge's fused CUDA operation. Native and NAG use
    # exactly the same test operator. q/k are [B,L,H,D], rope [1,L,1,D/2,2,2].
    def rotate(x, weight):
        y = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps).float()
        pair = torch.stack(y.chunk(2, dim=-1), dim=-1).unsqueeze(-1)
        result = (rope @ pair).squeeze(-1)
        return torch.cat((result[..., 0], result[..., 1]), dim=-1).to(x.dtype)
    return rotate(q, qw), rotate(k, kw)


def source_file(relative, fallback):
    root = os.environ.get("FORGE_NAG_TEST_SOURCE")
    if not root:
        pytest.skip("Set FORGE_NAG_TEST_SOURCE for native reduced CPU probes")
    root = Path(root)
    path = root / relative
    if not path.is_file():
        path = root / fallback
    if not path.is_file():
        pytest.fail(f"Missing native test source: {relative}")
    return path


def definitions(path, names=None):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                 and (names is None or n.name in names)]
    ns = dict(torch=torch, nn=nn, math=math, rearrange=rearrange, repeat=repeat,
              Rearrange=Rearrange, Callable=Callable, Optional=Optional,
              attention_function=attention, dynamic_args=SimpleNamespace(ref_latents=[]),
              weights_manual_cast=lambda layer, x: (layer.weight.to(x), None, None),
              main_stream_worker=lambda *args: contextlib.nullcontext(),
              ck=SimpleNamespace(rms_rope_split_half=rms_rope), pad_to_patch_size=pad)
    exec(compile(tree, str(path), "exec"), ns)
    return ns


@functools.lru_cache()
def native_classes():
    unet = definitions(source_file("backend/nn/unet.py", "unet.py"))
    anima = definitions(source_file("backend/nn/anima.py", "anima_dit.py"), {
        "_fn", "VideoRopePosition3DEmb", "GPT2FeedForward", "SelfCrossAttention",
        "Timesteps", "TimestepEmbedding", "PatchEmbed", "FinalLayer", "Block", "Anima"})
    kmodel = definitions(source_file("backend/modules/k_model.py", "kmodel_methods.py"), {"KModel"})
    return unet, anima, kmodel["KModel"]


class TestPatcher(Patcher):
    __test__ = False

    def set_model_attn2_replace(self, patch, block_name, number, transformer_index=None):
        # Same nested-copy contract as Forge ModelPatcher. The native UNet,
        # not this fixture, performs callback lookup and the final projection.
        options = self.model_options["transformer_options"].copy()
        all_patches = options.get("patches_replace", {}).copy()
        mapping = all_patches.get("attn2", {}).copy()
        key = (block_name, number, transformer_index) if transformer_index is not None else (block_name, number)
        mapping[key] = patch
        all_patches["attn2"] = mapping
        options["patches_replace"] = all_patches
        self.model_options["transformer_options"] = options


class Engine:
    def __init__(self, model, kmodel, family, dtype):
        km = kmodel.__new__(kmodel)
        nn.Module.__init__(km)
        km.diffusion_model = model
        km.computation_dtype = dtype
        km.predictor = Predictor()
        self.forge_objects = SimpleNamespace(unet=TestPatcher(km))
        self.family, self.encode_calls = family, 0
        self.chunks = [1, 1]
        self.tokens = [4, 5]
        self.fixes = []
        def tokenizer(index):
            return lambda text: ([SimpleNamespace(tokens=[0]*77, fixes=self.fixes)
                                 for _ in range(self.chunks[index])], 5)
        self.text_processing_engine_l = SimpleNamespace(tokenize_line=tokenizer(0), clip_skip=2)
        self.text_processing_engine_g = SimpleNamespace(tokenize_line=tokenizer(1), clip_skip=2)
        self.text_processing_engine_qwen = SimpleNamespace(tokenize=lambda text: tuple(list(range(n)) for n in self.tokens))
        self.negative = torch.randn(1, 77 if family == "sdxl" else 512, 24)
        self.use_shift = False
        self.prompts = []

    def get_learned_conditioning(self, prompt):
        self.encode_calls += 1
        self.prompts.append(prompt)
        assert prompt.is_negative_prompt
        if self.family == "sdxl":
            return dict(crossattn=self.negative, vector=torch.full((1,16), 999.))
        return [self.negative]


class Processing:
    def __init__(self, family, batch=1, dtype=torch.float32):
        u, a, km = native_classes()
        if family == "sdxl":
            model = u["IntegratedUNet2DConditionModel"](
                in_channels=4, model_channels=32, out_channels=4, num_res_blocks=1,
                channel_mult=(1, 2), num_heads=4, num_classes="sequential", adm_in_channels=16,
                use_spatial_transformer=True, use_checkpoint=False, context_dim=24,
                transformer_depth=[1,2], transformer_depth_middle=2,
                transformer_depth_output=[1,1,2,2], use_linear_in_transformer=True)
            layout = tuple(u[n] for n in ("SpatialTransformer", "BasicTransformerBlock", "CrossAttention"))
            self.x = torch.randn(batch, 4, 6, 8)
        else:
            model = a["Anima"](in_channels=4, out_channels=4, patch_spatial=2, patch_temporal=1,
                model_channels=48, crossattn_emb_channels=24, adaln_lora_dim=8,
                num_blocks=2, num_heads=4, mlp_ratio=2)
            layout = tuple(a[n] for n in ("Block", "SelfCrossAttention"))
            self.x = torch.randn(batch, 4, 1, 5, 7)
        model = model.to(dtype).eval()
        self.family, self.dtype, self.batch = family, dtype, batch
        self.sd_model = Engine(model, km, family, dtype)
        self.base = self.sd_model.forge_objects.unet
        self.bindings = HostBindings(Engine, type(model), km, Processing, Conditioning,
                         SimpleNamespace(ref_latents=[], context_handler=None), SimpleNamespace(attention=attention), family, layout)
        self.cfg_scale, self.width, self.height = 6.0, 64, 64
        self.extra_generation_params = {}
        self.scripts = Runner()
        self.sampler_name = "Euler"
        self.context = torch.randn(batch, 9, 24)
        self.uncond = torch.randn(batch, 9, 24)
        self.y, self.uy = torch.randn(batch,16), torch.randn(batch,16)
        self.labels = [1,0]
        self.separate = False
        self.sigmas = [1., .5]
        self.after_hook = None
        self.saved = 0
        self.observed_wrappers = []
        self.negative_prompt = "ordinary native negative"
        self.prompt = "one character"

    def payload(self, x, sigma, labels):
        options = dict(self.sd_model.forge_objects.unet.model_options.get("transformer_options", {}))
        options["cond_or_uncond"] = labels.copy()
        options["cond_indices"] = [r for i,c in enumerate(labels) if c == 0 for r in range(i*self.batch,(i+1)*self.batch)]
        options["uncond_indices"] = [r for i,c in enumerate(labels) if c == 1 for r in range(i*self.batch,(i+1)*self.batch)]
        c = dict(c_crossattn=torch.cat([self.context if l == 0 else self.uncond for l in labels]), transformer_options=options)
        if self.family == "sdxl":
            c["y"] = torch.cat([self.y if l == 0 else self.uy for l in labels])
        return dict(input=torch.cat([x for _ in labels]), timestep=torch.full((x.shape[0]*len(labels),), sigma), c=c, cond_or_uncond=labels.copy())

    def evaluate(self, payload):
        patcher = self.sd_model.forge_objects.unet
        wrapper = patcher.model_options.get("model_function_wrapper")
        if wrapper:
            self.observed_wrappers.append(wrapper)
            return wrapper(patcher.model.apply_model, payload)
        return patcher.model.apply_model(payload["input"], payload["timestep"], **payload["c"])

    def sample(self):
        self.sd_model.forge_objects.unet = self.base
        x = self.x.clone()
        self.scripts.process_before_every_sampling(self, x=x, c=None, uc=None)
        if self.after_hook:
            self.after_hook()
        for sigma in self.sigmas:
            if math.isclose(self.cfg_scale, 1.):
                x = self.evaluate(self.payload(x, sigma, [0]))
                continue
            if self.separate:
                u = self.evaluate(self.payload(x, sigma, [1]))
                c = self.evaluate(self.payload(x, sigma, [0]))
            else:
                chunks = self.evaluate(self.payload(x, sigma, self.labels)).chunk(2)
                c = chunks[self.labels.index(0)]
                u = chunks[self.labels.index(1)]
            # Host CFG equation; production extension neither replaces it nor
            # calculates a separate unconditional prompt.
            x = u + (c-u)*self.cfg_scale
        return x

    def generate_and_save(self):
        result = self.sample()
        self.saved += 1
        return result


def session(p, config=None):
    cfg = config or NAGConfig.parse(True, "glasses", 2., 2.5, .25, 1000., 0., p.family)
    s = SamplingSession(p, cfg, p.bindings, p.sd_model.negative.clone())
    s.install(p, x=p.x)
    return s
