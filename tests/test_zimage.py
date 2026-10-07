from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from forge_neo_nag.adapters.zimage import ZImageAdapter, encode_negative, validate_model
from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.host import HostBindings, SamplingSession, validate_request
from forge_neo_nag.registry import choose_adapter, normalize_choice
from .helpers import Conditioning, Patcher, Predictor, Runner, EmbedND


def attention(q, k, v, heads, mask=None, skip_reshape=False, transformer_options=None):
    assert skip_reshape is True
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    return y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def rms_rope(q, k, rope, qw, kw, eps):
    def rotate(x, weight):
        y = F.rms_norm(x, (x.shape[-1],), weight=weight, eps=eps).float()
        pair = torch.stack(y.chunk(2, dim=-1), dim=-1).unsqueeze(-1)
        result = (rope @ pair).squeeze(-1)
        return torch.cat((result[..., 0], result[..., 1]), dim=-1).to(x.dtype)
    return rotate(q, qw), rotate(k, kw)


def weights_manual_cast(layer, x):
    return layer.weight.to(x), None, None


def pad_to_patch_size(x, sizes):
    return F.pad(x, (0, (-x.shape[-1]) % sizes[-1], 0, (-x.shape[-2]) % sizes[-2]))


def modulate(x, scale):
    return x * (1 + scale.unsqueeze(1))


def clamp_fp16(x):
    return x


OPS = SimpleNamespace(
    attention=attention,
    modulate=modulate,
    clamp_fp16=clamp_fp16,
    main_stream_worker=lambda *args: contextlib.nullcontext(),
    weights_manual_cast=weights_manual_cast,
    ck=SimpleNamespace(rms_rope=rms_rope),
    pad_to_patch_size=pad_to_patch_size,
)


class JointAttention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.n_local_heads = heads
        self.n_local_kv_heads = heads
        self.n_rep = 1
        self.head_dim = dim // heads
        self.qk_norm = True
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

    def forward(self, x, x_mask, freqs_cis, transformer_options=None):
        b, l = x.shape[:2]
        q, k, v = torch.split(self.qkv(x), [32, 32, 32], dim=-1)
        q = q.view(b, l, self.n_local_heads, self.head_dim)
        k = k.view(b, l, self.n_local_kv_heads, self.head_dim)
        v = v.view(b, l, self.n_local_kv_heads, self.head_dim)
        q, k = rms_rope(q, k, freqs_cis, self.q_norm.weight, self.k_norm.weight, self.q_norm.eps)
        return self.out(attention(q.movedim(1,2), k.movedim(1,2), v.movedim(1,2), self.n_local_heads,
                                  x_mask, True, transformer_options))


class FeedForward(nn.Module):
    def __init__(self, dim=32):
        super().__init__()
        self.w1 = nn.Linear(dim, dim * 2, bias=False)
        self.w2 = nn.Linear(dim * 2, dim, bias=False)
        self.w3 = nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointTransformerBlock(nn.Module):
    def __init__(self, dim=32, heads=4, modulation=True, adaln_dim=8):
        super().__init__()
        self.dim = dim
        self.head_dim = dim // heads
        self.attention = JointAttention(dim, heads)
        self.feed_forward = FeedForward(dim)
        self.attention_norm1 = nn.RMSNorm(dim)
        self.ffn_norm1 = nn.RMSNorm(dim)
        self.attention_norm2 = nn.RMSNorm(dim)
        self.ffn_norm2 = nn.RMSNorm(dim)
        self.modulation = modulation
        if modulation:
            self.adaLN_modulation = nn.Sequential(nn.Linear(adaln_dim, 4 * dim))

    def forward(self, x, x_mask, freqs_cis, adaln_input=None, transformer_options=None):
        if self.modulation:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(adaln_input).chunk(4, dim=1)
            x = x + gate_msa.unsqueeze(1).tanh() * self.attention_norm2(
                self.attention(modulate(self.attention_norm1(x), scale_msa), x_mask, freqs_cis,
                               transformer_options=transformer_options))
            x = x + gate_mlp.unsqueeze(1).tanh() * self.ffn_norm2(
                self.feed_forward(modulate(self.ffn_norm1(x), scale_mlp)))
            return x
        x = x + self.attention_norm2(self.attention(self.attention_norm1(x), x_mask, freqs_cis,
                                                     transformer_options=transformer_options))
        return x + self.ffn_norm2(self.feed_forward(self.ffn_norm1(x)))


class TEmbed(nn.Module):
    def __init__(self, out=8):
        super().__init__(); self.proj = nn.Linear(1, out)
    def forward(self, t, dtype, **kwargs):
        return self.proj(t[:,None].to(dtype))


class FinalLayer(nn.Module):
    def __init__(self, dim=32, patch=2, channels=16, adaln_dim=8):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.mod=nn.Linear(adaln_dim,dim); self.out=nn.Linear(dim,patch*patch*channels)
    def forward(self, x, c):
        return self.out(self.norm(x) * (1 + self.mod(c).unsqueeze(1)))


class TinyZImage(nn.Module):
    def __init__(self, width=12, dim=32, heads=4, pad_multiple=4, layers=3):
        super().__init__()
        self.patch_size=2; self.in_channels=16; self.out_channels=16
        self.time_scale=1000.; self.dim=dim; self.n_heads=heads
        self.axes_dims=[2,2,4]; self.axes_lens=[1536,512,512]
        self.pad_tokens_multiple=pad_multiple
        self.x_embedder=nn.Linear(2*2*16,dim)
        self.cap_embedder=nn.Sequential(nn.RMSNorm(width),nn.Linear(width,dim))
        self.t_embedder=TEmbed(8)
        self.rope_embedder=EmbedND(dim//heads,256,[2,2,4])
        self.context_refiner=nn.ModuleList([JointTransformerBlock(dim,heads,False) for _ in range(2)])
        self.noise_refiner=nn.ModuleList([JointTransformerBlock(dim,heads,True) for _ in range(2)])
        self.layers=nn.ModuleList([JointTransformerBlock(dim,heads,True) for _ in range(layers)])
        self.final_layer=FinalLayer(dim,2,16,8)
        self.clip_text_pooled_proj=None
        self.cap_pad_token=nn.Parameter(torch.randn(1,dim)*.01)
        self.x_pad_token=nn.Parameter(torch.randn(1,dim)*.01)
        self.config={}

    def unpatchify(self,x,img_size,cap_size,return_tensor=False):
        imgs=[]; p=2
        for i in range(x.size(0)):
            h,w=img_size[i]; begin=cap_size[i]; end=begin+(h//p)*(w//p)
            imgs.append(x[i,begin:end].view(h//p,w//p,p,p,self.out_channels).permute(4,0,2,1,3).flatten(3,4).flatten(1,2))
        return torch.stack(imgs) if return_tensor else imgs

    def patchify_and_embed(self,x,cap_feats,cap_mask,t,num_tokens,transformer_options=None):
        b,c,h,w=x.shape; p=2; dtype=x.dtype; options=transformer_options or {}
        if self.pad_tokens_multiple:
            extra=(-cap_feats.shape[1])%self.pad_tokens_multiple
            if extra:
                cap_feats=torch.cat((cap_feats,self.cap_pad_token.to(cap_feats).unsqueeze(0).repeat(b,extra,1)),1)
        cap_ids=torch.zeros(b,cap_feats.shape[1],3,device=x.device)
        cap_ids[:,:,0]=torch.arange(cap_feats.shape[1],device=x.device)+1
        image=self.x_embedder(x.view(b,c,h//p,p,w//p,p).permute(0,2,4,3,5,1).flatten(3).flatten(1,2))
        ht,wt=h//p,w//p
        img_ids=torch.zeros(b,image.shape[1],3,device=x.device)
        img_ids[:,:,0]=cap_feats.shape[1]+1
        img_ids[:,:,1]=torch.arange(ht,device=x.device).view(-1,1).repeat(1,wt).flatten()
        img_ids[:,:,2]=torch.arange(wt,device=x.device).view(1,-1).repeat(ht,1).flatten()
        if self.pad_tokens_multiple:
            extra=(-image.shape[1])%self.pad_tokens_multiple
            if extra:
                image=torch.cat((image,self.x_pad_token.to(image).unsqueeze(0).repeat(b,extra,1)),1)
                img_ids=F.pad(img_ids,(0,0,0,extra))
        freqs=self.rope_embedder(torch.cat((cap_ids,img_ids),1)).movedim(1,2).to(dtype)
        for layer in self.context_refiner:
            cap_feats=layer(cap_feats,cap_mask,freqs[:,:cap_ids.shape[1]],transformer_options=options)
        t_emb=self.t_embedder(t*self.time_scale,dtype=dtype)
        for layer in self.noise_refiner:
            image=layer(image,None,freqs[:,cap_ids.shape[1]:],t_emb,transformer_options=options)
        return torch.cat((cap_feats,image),1),None,[(h,w)]*b,[cap_feats.shape[1]]*b,freqs

    def forward(self,x,timesteps,context,num_tokens=None,attention_mask=None,transformer_options=None,**kwargs):
        h,w=x.shape[-2:]; x=pad_to_patch_size(x,(2,2)); t=1-timesteps
        adaln=self.t_embedder(t*self.time_scale,dtype=x.dtype); cap=self.cap_embedder(context)
        seq,mask,sizes,cap_size,freqs=self.patchify_and_embed(x,cap,attention_mask,t,num_tokens,transformer_options or {})
        for layer in self.layers:
            seq=layer(seq,mask,freqs,adaln,transformer_options=transformer_options or {})
        seq=self.final_layer(seq,adaln)
        return -self.unpatchify(seq,sizes,cap_size,True)[:,:,:h,:w]


class KModel(nn.Module):
    def __init__(self,model):
        super().__init__(); self.diffusion_model=model; self.computation_dtype=torch.float32; self.predictor=Predictor()
    def apply_model(self,x,t,c_crossattn=None,transformer_options=None,control=None,c_concat=None,**kwargs):
        z=self.predictor.calculate_input(t,x); mt=self.predictor.timestep(t)
        pred=self.diffusion_model(z,mt,context=c_crossattn,transformer_options=transformer_options,control=control,**kwargs)
        return self.predictor.calculate_denoised(t,pred,x)


class Engine:
    def __init__(self,model):
        self.forge_objects=SimpleNamespace(unet=Patcher(KModel(model)),clip=SimpleNamespace(patcher=SimpleNamespace(patches_uuid=None)))
        self.text_processing_engine_qwen=SimpleNamespace(tokenize=lambda text:list(range(8)))
        self.encode_calls=0; self.negative=torch.randn(6,12)
    def get_learned_conditioning(self,prompt):
        assert prompt.is_negative_prompt; self.encode_calls+=1; return [self.negative.clone()]


class Processing:
    def __init__(self,batch=1):
        model=TinyZImage(); self.sd_model=Engine(model); self.base=self.sd_model.forge_objects.unet; self.scripts=Runner()
        self.cfg_scale=1.; self.width=self.height=64; self.distilled_cfg_scale=1.; self.enable_hr=self.is_hr_pass=self.txt2img_upscale=self.tiling=False
        self.refiner_checkpoint=None; self.sampler_name='Euler'; self.extra_generation_params={}; self.prompt='portrait'; self.all_prompts=['portrait']
        self.x=torch.randn(batch,16,5,7); self.context=torch.randn(batch,5,12); self.uncond=torch.randn(batch,5,12)
        self.bindings=HostBindings(Engine,TinyZImage,KModel,Processing,Conditioning,SimpleNamespace(ref_latents=[],context_handler=None),OPS,'zimage',(JointTransformerBlock,JointAttention))


def adapter(p,alpha=.25,negative=None):
    neg=negative if negative is not None else torch.randn(1,7,12)
    return ZImageAdapter(p.base.model.diffusion_model,neg,NAGConfig.parse(True,'glasses',2.,2.5,alpha,1000,0,'zimage'),OPS,p.bindings.layout_types)


def test_zimage_registry_and_model_contract():
    Fake=type('ZImage',(),{}); Fake.__module__='backend.diffusion_engine.zimage'
    assert choose_adapter(Fake())=='zimage'
    assert normalize_choice('ZIMAGE')=='zimage'
    p=Processing(); assert validate_model(p.base.model.diffusion_model,p.bindings.layout_types)==12


def test_zimage_encoder_negative_flag_and_limit():
    p=Processing(); out=encode_negative(p,NAGConfig.parse(True,'glasses',adapter='zimage'),p.bindings)
    assert out.shape==(1,6,12) and p.sd_model.encode_calls==1
    p.sd_model.text_processing_engine_qwen.tokenize=lambda text:list(range(2049))
    with pytest.raises(NAGError,match='2048'): encode_negative(p,NAGConfig.parse(True,'glasses',adapter='zimage'),p.bindings)


@pytest.mark.parametrize('batch,pos_len,neg_len',[(1,5,7),(2,8,3)])
def test_zimage_zero_alpha_matches_native_with_different_lengths(batch,pos_len,neg_len):
    p=Processing(batch); model=p.base.model.diffusion_model; x=p.x.clone(); ctx=torch.randn(batch,pos_len,12); t=torch.ones(batch)*.5
    base=model(x,t,ctx,transformer_options={})
    got=adapter(p,alpha=0,negative=torch.randn(1,neg_len,12))(x,t,ctx,transformer_options={})
    torch.testing.assert_close(got,base,rtol=1e-5,atol=2e-5)


def test_zimage_nonzero_changes_finite_and_hybrid_image_rope_is_shared():
    p=Processing(); model=p.base.model.diffusion_model; x=p.x.clone(); ctx=p.context.clone(); t=torch.ones(1)*.5
    a=adapter(p,negative=torch.zeros(1,7,12)); b=adapter(p,negative=torch.ones(1,3,12))
    ya=a(x,t,ctx,transformer_options={}); yb=b(x,t,ctx,transformer_options={})
    assert torch.isfinite(ya).all() and not torch.allclose(ya,yb,atol=1e-7,rtol=1e-7)
    assert a.attention_calls==len(model.layers)
    assert torch.equal(a.last_positive_image_pe,a.last_negative_image_pe)


def test_zimage_image_qkv_is_not_projected_by_negative_branch():
    p=Processing(); a=adapter(p); block=p.base.model.diffusion_model.layers[0]
    calls=[]
    handle=block.attention.qkv.register_forward_hook(lambda m,args,out:calls.append(args[0].shape[1]))
    try: a(p.x,torch.ones(1)*.5,p.context,transformer_options={})
    finally: handle.remove()
    # Main block: one positive full sequence + one negative TEXT-only sequence.
    # Context/noise refiner qkv calls are separate modules and are not counted here.
    pos_padded=(-p.context.shape[1])%p.base.model.diffusion_model.pad_tokens_multiple+p.context.shape[1]
    neg_padded=(-7)%p.base.model.diffusion_model.pad_tokens_multiple+7
    assert len(calls)==2 and calls[0] > pos_padded and calls[1]==neg_padded


def test_zimage_pad_multiple_and_negative_text_advances_each_block():
    p=Processing(); a=adapter(p,negative=torch.randn(1,5,12))
    a(p.x,torch.ones(1)*.5,p.context,transformer_options={})
    assert a.attention_calls==3


@pytest.mark.parametrize('labels',[[0,1],[1,0]])
def test_zimage_host_wrapper_preserves_native_unconditional_branch(labels):
    p=Processing(); p.cfg_scale=5.; cfg=NAGConfig.parse(True,'glasses',2,2.5,.25,1000,0,'zimage')
    s=SamplingSession(p,cfg,p.bindings,torch.randn(1,7,12)); s.install(p,x=p.x)
    x=torch.cat((p.x,p.x)); context=torch.cat([p.context if label==0 else p.uncond for label in labels]); sigma=torch.ones(2)*.5
    c={'c_crossattn':context,'transformer_options':{'cond_or_uncond':labels}}
    ui=labels.index(1); part=slice(ui,ui+1)
    native=p.base.model.apply_model(x[part],sigma[part],c_crossattn=context[part],transformer_options={'cond_or_uncond':[1]})
    try:
        out=s.wrapper(p.base.model.apply_model,{'input':x,'timestep':sigma,'c':c,'cond_or_uncond':labels})
        torch.testing.assert_close(out[part],native,atol=0,rtol=0)
        assert s.wrapper.active_calls==1
    finally:s.close()


def test_zimage_rejects_positive_and_reference_patch_mask_and_nunchaku_shape():
    p=Processing(); p.prompt='girl AND landscape';p.all_prompts=[p.prompt]
    with pytest.raises(NAGError,match='AND composition'): validate_request(p,p.bindings)
    p.prompt='portrait';p.all_prompts=[p.prompt];p.bindings.dynamic_args.ref_latents=[torch.zeros(1)]
    with pytest.raises(NAGError,match='Reference'): validate_request(p,p.bindings)
    p.bindings.dynamic_args.ref_latents=[]; a=adapter(p)
    with pytest.raises(NAGError,match='existing transformer patches'):
        a(p.x,torch.ones(1)*.5,p.context,transformer_options={'patches':{'x':[1]}})
    with pytest.raises(NAGError,match='attention masks'):
        a(p.x,torch.ones(1)*.5,p.context,attention_mask=torch.ones(1,5),transformer_options={})
    class ForeignAttention(JointAttention): pass
    p.base.model.diffusion_model.layers[0].attention=ForeignAttention()
    with pytest.raises(NAGError,match='native JointAttention'): validate_model(p.base.model.diffusion_model,p.bindings.layout_types)
