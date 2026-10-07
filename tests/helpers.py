"""Small CPU contract models, not pretrained Krea2 and not a GPU substitute."""
from __future__ import annotations

import copy
import hashlib
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from forge_neo_nag.adapters.krea2 import Krea2Ops
from forge_neo_nag.host import HostBindings


def attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options=None):
    assert skip_reshape is True
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def pad(x, patch, padding_mode="replicate"):
    return F.pad(x, (0, (-x.shape[-1]) % patch[1], 0, (-x.shape[-2]) % patch[0]), mode=padding_mode)


def embedding(t, dim, max_period=10000, time_factor=1000.0):
    freq = torch.exp(-math.log(max_period) * torch.arange(dim // 2, device=t.device).float() / (dim // 2))
    angle = (t.float() * time_factor)[:, None] * freq
    result = torch.cat((torch.cos(angle), torch.sin(angle)), dim=-1)
    return F.pad(result, (0, dim % 2)).to(t.dtype)


class EmbedND(nn.Module):
    def __init__(self, dim, theta, axes_dim):
        super().__init__()
        self.dim, self.theta, self.axes_dim = dim, theta, axes_dim

    def forward(self, ids):
        chunks = []
        for axis, width in enumerate(self.axes_dim):
            freq = self.theta ** (-torch.arange(0, width, 2, device=ids.device).double() / max(1, width))
            angle = ids[..., axis, None] * freq
            co, si = angle.cos(), angle.sin()
            chunks.append(torch.stack((co, -si, si, co), -1).reshape(*angle.shape, 2, 2))
        return torch.cat(chunks, dim=-3).float().unsqueeze(1)


def rope(q, k, freqs):
    def apply(x):
        pairs = x.float().reshape(*x.shape[:-1], -1, 2, 1)
        return (freqs @ pairs).squeeze(-1).flatten(-2).to(x.dtype)
    return apply(q), apply(k)


OPS = Krea2Ops(attention, rope, embedding, pad)


class QKNorm(nn.Module):
    def forward(self, q, k):
        return F.rms_norm(q, (q.shape[-1],)), F.rms_norm(k, (k.shape[-1],))


class TinyAttention(nn.Module):
    def __init__(self, features=64, heads=4, kvheads=2):
        super().__init__()
        self.heads, self.kvheads = heads, kvheads
        kvdim = features // heads * kvheads
        self.wq, self.wk, self.wv = nn.Linear(features, features), nn.Linear(features, kvdim), nn.Linear(features, kvdim)
        self.gate, self.wo = nn.Linear(features, features), nn.Linear(features, features)
        self.qknorm = QKNorm()

    def forward(self, x, freqs=None, mask=None, transformer_options=None):
        shape = lambda t, h: t.reshape(t.shape[0], t.shape[1], h, -1).transpose(1, 2)
        q, k, v = shape(self.wq(x), self.heads), shape(self.wk(x), self.kvheads), shape(self.wv(x), self.kvheads)
        q, k = self.qknorm(q, k)
        if freqs is not None:
            q, k = rope(q, k, freqs)
        k = k.repeat_interleave(self.heads // self.kvheads, 1)
        v = v.repeat_interleave(self.heads // self.kvheads, 1)
        return self.wo(attention(q, k, v, self.heads, mask, True, transformer_options) * torch.sigmoid(self.gate(x)))


class Modulation(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(features * 6) * 0.05)

    def forward(self, x):
        return (x + self.bias).chunk(6, -1)


class TinyBlock(nn.Module):
    def __init__(self, features=64, heads=4, kvheads=2):
        super().__init__()
        self.prenorm, self.postnorm = nn.LayerNorm(features), nn.LayerNorm(features)
        self.mod = Modulation(features)
        self.attn = TinyAttention(features, heads, kvheads)
        self.mlp = nn.Sequential(nn.Linear(features, features * 2), nn.SiLU(), nn.Linear(features * 2, features))

    def forward(self, x, vec, freqs, mask=None, transformer_options=None):
        s, h, g, s2, h2, g2 = self.mod(vec)
        x = torch.addcmul(x, g, self.attn(torch.addcmul(h, 1 + s, self.prenorm(x)), freqs, mask, transformer_options))
        return torch.addcmul(x, g2, self.mlp(torch.addcmul(h2, 1 + s2, self.postnorm(x))))


class TinyFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.calls = 0

    def forward(self, x, mask=None, transformer_options=None):
        self.calls += 1
        x.add_(0.01)  # Model the host's in-place text fusion behavior.
        return self.linear(x.mean(2))


class Last(nn.Module):
    def __init__(self, f, outputs):
        super().__init__()
        self.proj = nn.Linear(f, outputs)

    def forward(self, x, t):
        return self.proj(x + 0.02 * t)


class TinyDiT(nn.Module):
    def __init__(self, layers=3, kvheads=2):
        super().__init__()
        self.patch, self.channels, self.tdim, self.txtlayers, self.txtdim = 2, 4, 16, 3, 16
        f, heads = 64, 4
        self.first = nn.Linear(self.channels * self.patch**2, f)
        self.tmlp = nn.Sequential(nn.Linear(self.tdim, f), nn.SiLU(), nn.Linear(f, f))
        self.tproj = nn.Linear(f, f * 6)
        self.txtfusion = TinyFusion(self.txtdim)
        self.txtmlp = nn.Sequential(nn.Linear(self.txtdim, f), nn.SiLU(), nn.Linear(f, f))
        self.pe_embedder = EmbedND(16, 1000, [4, 6, 6])
        self.blocks = nn.ModuleList([TinyBlock(f, heads, kvheads) for _ in range(layers)])
        self.last = Last(f, self.channels * self.patch**2)

    def forward(self, x, timesteps, context, transformer_options=None, **kwargs):
        from einops import rearrange
        temporal = x.ndim == 5
        if temporal:
            x = x[:, :, 0]
        h0, w0 = x.shape[-2:]
        x = pad(x, (self.patch, self.patch))
        h, w = x.shape[-2] // self.patch, x.shape[-1] // self.patch
        image = self.first(rearrange(x, 'b c (h p) (w q) -> b (h w) (c p q)', p=self.patch, q=self.patch))
        t = self.tmlp(embedding(timesteps, self.tdim).unsqueeze(1).to(image.dtype))
        vec = self.tproj(t)
        txt = self.txtmlp(self.txtfusion(context.clone(), transformer_options=transformer_options))
        seq = torch.cat((txt, image), 1)
        ids = torch.zeros(x.shape[0], seq.shape[1], 3)
        ids[:, txt.shape[1]:, 1] = torch.arange(h).repeat_interleave(w)
        ids[:, txt.shape[1]:, 2] = torch.arange(w).repeat(h)
        freq = self.pe_embedder(ids)
        for block in self.blocks:
            seq = block(seq, vec, freq, transformer_options=transformer_options)
        result = rearrange(self.last(seq, t)[:, txt.shape[1]:], 'b (h w) (c p q) -> b c (h p) (w q)', h=h, w=w, p=self.patch, q=self.patch)
        result = result[:, :, :h0, :w0]
        return result.unsqueeze(2) if temporal else result


class Predictor:
    def __init__(self):
        self.calls = []

    def calculate_input(self, sigma, x):
        self.calls.append("input")
        return x / (1 + sigma.reshape(-1, *([1] * (x.ndim - 1))))

    def timestep(self, sigma):
        self.calls.append("time")
        return 0.03 + 0.2 * sigma

    def calculate_denoised(self, sigma, prediction, x):
        self.calls.append("output")
        return x - prediction * sigma.reshape(-1, *([1] * (x.ndim - 1)))


class TinyKModel:
    def __init__(self, model):
        self.diffusion_model = model
        self.computation_dtype = torch.float32
        self.predictor = Predictor()

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
        s = t
        z = self.predictor.calculate_input(s, x).to(self.computation_dtype)
        mt = self.predictor.timestep(t).float()
        pred = self.diffusion_model(z, mt, context=c_crossattn.to(self.computation_dtype), control=control,
                                    transformer_options=transformer_options, **kwargs).float()
        return self.predictor.calculate_denoised(s, pred, x)


class Patcher:
    def __init__(self, model):
        self.model = model
        self.model_options = {"transformer_options": {}}
        self.object_patches = {}
        self.controlnet_linked_list = None
        self.extra_preserved_memory_during_sampling = 0

    def clone(self):
        clone = copy.copy(self)
        clone.model_options = copy.deepcopy(self.model_options)
        return clone

    def set_model_unet_function_wrapper(self, wrapper):
        self.model_options["model_function_wrapper"] = wrapper

    def add_extra_preserved_memory_during_sampling(self, amount):
        self.extra_preserved_memory_during_sampling += amount


class Conditioning(list):
    def __init__(self, texts, **kwargs):
        super().__init__(texts)
        self.__dict__.update(kwargs)


class Engine:
    def __init__(self, model):
        self.forge_objects = SimpleNamespace(unet=Patcher(TinyKModel(model)))
        self.text_processing_engine_qwen = SimpleNamespace(tokenize=lambda s: s.split() + ["system", "user"])
        self.encode_calls = 0
        self.last_encoded = None

    def get_learned_conditioning(self, prompt):
        assert prompt.is_negative_prompt
        self.encode_calls += 1
        seed = int.from_bytes(hashlib.sha256(prompt[0].encode()).digest()[:4], 'little')
        gen = torch.Generator().manual_seed(seed)
        self.last_encoded = torch.randn(5 + len(prompt[0]) % 3, 3, 16, generator=gen)
        return [self.last_encoded]


class Runner:
    def __init__(self):
        self.hooks = []
        self.errors = []
        self.calls = 0
        self.attribute = "delegated"

    def process_before_every_sampling(self, p, **kwargs):
        self.calls += 1
        for hook in self.hooks:
            try:
                hook(p, **kwargs)
            except Exception as exc:
                self.errors.append(exc)


class Processing:
    def __init__(self, model=None, batch=1):
        self.sd_model = Engine(model or TinyDiT())
        self.base = self.sd_model.forge_objects.unet
        self.scripts = Runner()
        self.cfg_scale, self.width, self.height, self.distilled_cfg_scale = 1.0, 64, 64, 1.0
        self.enable_hr, self.is_hr_pass, self.txt2img_upscale, self.tiling = False, False, False, False
        self.refiner_checkpoint, self.sampler_name = None, "Euler"
        self.extra_generation_params = {}
        self.negative_prompt = "standard negative remains untouched"
        self.x = torch.randn(batch, 4, 1, 4, 6)
        self.context = torch.randn(batch, 8, 3, 16)
        self.sigmas = [1.0, 0.5, 0.1]
        self.after_hook = None
        self.skip_model = False
        self.observed_wrappers = []
        self.saved = 0

    def sample(self):
        self.sd_model.forge_objects.unet = self.base
        x = self.x.clone()
        self.scripts.process_before_every_sampling(self, x=x, noise=x, c=None, uc=None)
        if self.after_hook:
            self.after_hook()
        if self.skip_model:
            return x
        for level in self.sigmas:
            patcher = self.sd_model.forge_objects.unet
            wrapper = patcher.model_options.get("model_function_wrapper")
            sigma = torch.full((x.shape[0],), level)
            c = {"c_crossattn": self.context.clone(), "transformer_options": {"cond_or_uncond": [0]}}
            if wrapper is None:
                x = patcher.model.apply_model(x, sigma, **c)
            else:
                self.observed_wrappers.append(wrapper)
                x = wrapper(patcher.model.apply_model, {"input": x, "timestep": sigma, "c": c, "cond_or_uncond": [0]})
        return x

    def generate_and_save(self):
        result = self.sample()
        self.saved += 1
        return result


def make_bindings():
    return HostBindings(Engine, TinyDiT, TinyKModel, Processing, Conditioning,
                        SimpleNamespace(ref_latents=[], context_handler=None), OPS)
