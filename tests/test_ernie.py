"""Reduced Ernie CPU contract tests; no pretrained weights or GPU validation."""
import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from forge_neo_nag.adapters.ernie import ErnieAdapter, encode_negative, validate_model
from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.host import HostBindings, SamplingSession, validate_request
from forge_neo_nag.registry import choose_adapter, normalize_choice
from .helpers import Conditioning, Patcher, Runner, TinyKModel


def weights_manual_cast(module, value):
    return module.weight, None, None


def main_stream_worker(*args):
    return contextlib.nullcontext()


def rms_rope_split_half(q, k, rotary, q_scale, k_scale, eps):
    q = F.rms_norm(q, (q.shape[-1],), weight=q_scale, eps=eps)
    k = F.rms_norm(k, (k.shape[-1],), weight=k_scale, eps=eps)
    return q + rotary.to(q), k + rotary.to(k)


def attention(q, k, v, heads, mask=None, **kwargs):
    b, qlen = q.shape[:2]
    klen = k.shape[1]
    dim = q.shape[-1] // heads
    qh = q.view(b, qlen, heads, dim).transpose(1, 2)
    kh = k.view(b, klen, heads, dim).transpose(1, 2)
    vh = v.view(b, klen, heads, dim).transpose(1, 2)
    y = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    return y.transpose(1, 2).reshape(b, qlen, heads * dim)


OPS = SimpleNamespace(
    attention=attention,
    weights_manual_cast=weights_manual_cast,
    main_stream_worker=main_stream_worker,
    ck=SimpleNamespace(rms_rope_split_half=rms_rope_split_half),
)


class ErnieImageEmbedND3(nn.Module):
    def __init__(self, dim=8, axes=(2,2,4)):
        super().__init__(); self.dim=dim; self.axes_dim=list(axes)
    def forward(self, ids):
        return (ids.sum(-1, keepdim=True).unsqueeze(-1) * 0.001)


class ErnieImagePatchEmbedDynamic(nn.Module):
    def __init__(self, channels=4, dim=32, patch=1):
        super().__init__(); self.patch_size=patch
        self.proj=nn.Conv2d(channels,dim,patch,patch)
    def forward(self,x):
        x=self.proj(x); b,d,h,w=x.shape
        return x.reshape(b,d,h*w).transpose(1,2).contiguous()


class ErnieImageAttention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__(); self.heads=heads; self.head_dim=dim//heads; self.inner_dim=dim
        self.to_q=nn.Linear(dim,dim,bias=False); self.to_k=nn.Linear(dim,dim,bias=False)
        self.to_v=nn.Linear(dim,dim,bias=False)
        self.norm_q=nn.RMSNorm(self.head_dim); self.norm_k=nn.RMSNorm(self.head_dim)
        self.to_out=nn.ModuleList([nn.Linear(dim,dim,bias=False)])
    def forward(self,x,attention_mask=None,image_rotary_emb=None):
        b,s=x.shape[:2]
        q=self.to_q(x).view(b,s,self.heads,self.head_dim)
        k=self.to_k(x).view(b,s,self.heads,self.head_dim)
        v=self.to_v(x)
        q,k=rms_rope_split_half(q,k,image_rotary_emb,self.norm_q.weight,self.norm_k.weight,self.norm_q.eps)
        return self.to_out[0](attention(q.reshape(b,s,-1),k.reshape(b,s,-1),v,self.heads,mask=attention_mask))


class MLP(nn.Module):
    def __init__(self,dim=32):
        super().__init__(); self.gate_proj=nn.Linear(dim,dim*2,bias=False)
        self.up_proj=nn.Linear(dim,dim*2,bias=False); self.linear_fc2=nn.Linear(dim*2,dim,bias=False)
    def forward(self,x):
        return self.linear_fc2(self.up_proj(x)*F.gelu(self.gate_proj(x)))


class ErnieImageSharedAdaLNBlock(nn.Module):
    def __init__(self,dim=32,heads=4):
        super().__init__(); self.adaLN_sa_ln=nn.RMSNorm(dim); self.self_attention=ErnieImageAttention(dim,heads)
        self.adaLN_mlp_ln=nn.RMSNorm(dim); self.mlp=MLP(dim)
    def forward(self,x,rotary_pos_emb,temb,attention_mask=None):
        sm,cm,gm,sf,cf,gf=temb
        residual=x; n=self.adaLN_sa_ln(x); n=(n.float()*(1+cm.float())+sm.float()).to(x.dtype)
        x=residual+(gm.float()*self.self_attention(n,attention_mask,image_rotary_emb=rotary_pos_emb).float()).to(x.dtype)
        residual=x; n=self.adaLN_mlp_ln(x); n=(n.float()*(1+cf.float())+sf.float()).to(x.dtype)
        return residual+(gf.float()*self.mlp(n).float()).to(x.dtype)


class FinalNorm(nn.Module):
    def __init__(self,dim=32):
        super().__init__(); self.norm=nn.LayerNorm(dim); self.linear=nn.Linear(dim,dim*2)
    def forward(self,x,c):
        scale,shift=self.linear(c).chunk(2,-1); x=self.norm(x)
        return x*(1+scale.unsqueeze(1))+shift.unsqueeze(1)


class TinyErnie(nn.Module):
    def __init__(self,width=12,dim=32,heads=4,layers=3):
        super().__init__(); self.hidden_size=dim; self.num_heads=heads; self.head_dim=dim//heads
        self.patch_size=1; self.out_channels=4
        self.x_embedder=ErnieImagePatchEmbedDynamic(4,dim,1)
        self.text_proj=nn.Linear(width,dim,bias=False)
        self.time_proj=nn.Linear(1,dim); self.time_embedding=nn.Sequential(nn.SiLU(),nn.Linear(dim,dim))
        self.pos_embed=ErnieImageEmbedND3(self.head_dim,(2,2,4))
        self.adaLN_modulation=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        self.layers=nn.ModuleList([ErnieImageSharedAdaLNBlock(dim,heads) for _ in range(layers)])
        self.final_norm=FinalNorm(dim); self.final_linear=nn.Linear(dim,4)
    def forward(self,x,timesteps,context,**kwargs):
        b,c,h,w=x.shape; image=self.x_embedder(x); text=self.text_proj(context)
        tlen=text.shape[1]; hidden=torch.cat((image,text),1)
        text_ids=torch.zeros(b,tlen,3,device=x.device); text_ids[:,:,0]=torch.linspace(0,tlen-1,steps=tlen,device=x.device)
        image_ids=torch.zeros(b,h*w,3,device=x.device); image_ids[:,:,0]=float(tlen)
        image_ids[:,:,1]=torch.arange(h,device=x.device).view(-1,1).repeat(1,w).flatten()
        image_ids[:,:,2]=torch.arange(w,device=x.device).view(1,-1).repeat(h,1).flatten()
        rotary=self.pos_embed(torch.cat((image_ids,text_ids),1))
        sample=self.time_proj(timesteps[:,None]).to(x.dtype); cond=self.time_embedding(sample)
        temb=[z.unsqueeze(1).contiguous() for z in self.adaLN_modulation(cond).chunk(6,-1)]
        for layer in self.layers: hidden=layer(hidden,rotary,temb)
        hidden=self.final_norm(hidden,cond).type_as(hidden)
        patches=self.final_linear(hidden)[:,:h*w]
        return patches.transpose(1,2).reshape(b,4,h,w)


class Engine:
    def __init__(self,model):
        self.forge_objects=SimpleNamespace(unet=Patcher(TinyKModel(model)),
            clip=SimpleNamespace(patcher=SimpleNamespace(patches_uuid=None)))
        self.text_processing_engine_ministral=SimpleNamespace(tokenize=lambda text:list(range(8)))
        self.encode_calls=0; self.negative=torch.randn(6,12)
    def get_learned_conditioning(self,prompt):
        self.encode_calls+=1; return [self.negative.clone()]


class Processing:
    def __init__(self,batch=1):
        model=TinyErnie(); self.sd_model=Engine(model); self.base=self.sd_model.forge_objects.unet
        self.scripts=Runner(); self.cfg_scale=1.; self.width=self.height=64
        self.distilled_cfg_scale=1.; self.enable_hr=self.is_hr_pass=self.txt2img_upscale=self.tiling=False
        self.refiner_checkpoint=None; self.sampler_name='Euler'; self.extra_generation_params={}
        self.prompt='portrait'; self.all_prompts=['portrait']
        self.x=torch.randn(batch,4,4,5); self.context=torch.randn(batch,5,12); self.uncond=torch.randn(batch,5,12)
        self.bindings=HostBindings(Engine,TinyErnie,TinyKModel,Processing,Conditioning,
            SimpleNamespace(ref_latents=[],context_handler=None),OPS,'ernie',
            (ErnieImageSharedAdaLNBlock,ErnieImageAttention,ErnieImagePatchEmbedDynamic))


def adapter(p,alpha=.25,negative=None):
    neg=negative if negative is not None else torch.randn(1,7,12)
    return ErnieAdapter(p.base.model.diffusion_model,neg,
        NAGConfig.parse(True,'glasses',2.,2.5,alpha,1000,0,'ernie'),OPS,p.bindings.layout_types)


def test_ernie_registry_model_and_encoder_contract():
    Fake=type('ErnieImage',(),{}); Fake.__module__='backend.diffusion_engine.ernie'
    assert choose_adapter(Fake())=='ernie' and normalize_choice('ERNIE')=='ernie'
    p=Processing(); assert validate_model(p.base.model.diffusion_model,p.bindings.layout_types)==12
    out=encode_negative(p,NAGConfig.parse(True,'glasses',adapter='ernie'),p.bindings)
    assert out.shape==(1,6,12) and p.sd_model.encode_calls==1
    p.sd_model.text_processing_engine_ministral.tokenize=lambda text:list(range(2049))
    with pytest.raises(NAGError,match='2048'):
        encode_negative(p,NAGConfig.parse(True,'glasses',adapter='ernie'),p.bindings)


@pytest.mark.parametrize('batch,pos_len,neg_len',[(1,5,7),(2,8,3)])
def test_ernie_zero_alpha_matches_native_with_different_lengths(batch,pos_len,neg_len):
    p=Processing(batch); model=p.base.model.diffusion_model
    x=p.x.clone(); ctx=torch.randn(batch,pos_len,12); t=torch.ones(batch)*.5
    base=model(x,t,ctx)
    got=adapter(p,alpha=0,negative=torch.randn(1,neg_len,12))(x,t,ctx,transformer_options={})
    torch.testing.assert_close(got,base,rtol=1e-5,atol=2e-5)


def test_ernie_nonzero_hybrid_rope_and_image_qkv_shared():
    p=Processing(); model=p.base.model.diffusion_model; a=adapter(p,negative=torch.zeros(1,7,12))
    block=model.layers[0]; calls=[]
    h=block.self_attention.to_q.register_forward_hook(lambda m,args,out:calls.append(args[0].shape[1]))
    try:
        y=a(p.x,torch.ones(1)*.5,p.context,transformer_options={})
    finally:
        h.remove()
    assert torch.isfinite(y).all() and a.attention_calls==len(model.layers)
    assert torch.equal(a.last_positive_image_pe,a.last_negative_image_pe)
    assert len(calls)==2 and calls[0]==p.x.shape[-2]*p.x.shape[-1]+p.context.shape[1] and calls[1]==7


@pytest.mark.parametrize('labels',[[0,1],[1,0]])
def test_ernie_host_preserves_native_unconditional(labels):
    p=Processing(); p.cfg_scale=5.; cfg=NAGConfig.parse(True,'glasses',2,2.5,.25,1000,0,'ernie')
    s=SamplingSession(p,cfg,p.bindings,torch.randn(1,7,12)); s.install(p,x=p.x)
    x=torch.cat((p.x,p.x)); context=torch.cat([p.context if label==0 else p.uncond for label in labels]); sigma=torch.ones(2)*.5
    c={'c_crossattn':context,'transformer_options':{'cond_or_uncond':labels}}
    ui=labels.index(1); part=slice(ui,ui+1)
    native=p.base.model.apply_model(x[part],sigma[part],c_crossattn=context[part],transformer_options={'cond_or_uncond':[1]})
    try:
        out=s.wrapper(p.base.model.apply_model,{'input':x,'timestep':sigma,'c':c,'cond_or_uncond':labels})
        torch.testing.assert_close(out[part],native,atol=0,rtol=0)
        assert s.wrapper.active_calls==1
    finally:
        s.close()


def test_ernie_rejects_positive_and_and_detects_ministral_stamp_change():
    p=Processing(); p.prompt='girl AND landscape'; p.all_prompts=[p.prompt]
    with pytest.raises(NAGError,match='AND composition'): validate_request(p,p.bindings)
    p.prompt='portrait'; p.all_prompts=[p.prompt]
    cfg=NAGConfig.parse(True,'glasses',adapter='ernie')
    s=SamplingSession(p,cfg,p.bindings,torch.randn(1,7,12))
    p.sd_model.text_processing_engine_ministral=SimpleNamespace(tokenize=lambda text:[1])
    with pytest.raises(NAGError,match='changed after NAG negative encoding'):
        s.install(p,x=p.x)


def test_ernie_rejects_late_block_forward_patch_after_session_install():
    p=Processing()
    cfg=NAGConfig.parse(True,'glasses',2,2.5,.25,1000,0,'ernie')
    s=SamplingSession(p,cfg,p.bindings,torch.randn(1,7,12))
    s.install(p,x=p.x)
    block=p.base.model.diffusion_model.layers[0]
    block.forward=lambda *args,**kwargs: (_ for _ in ()).throw(
        RuntimeError('foreign extension block forward')
    )
    payload={
        'input':p.x.clone(),
        'timestep':torch.ones(1)*.5,
        'c':{
            'c_crossattn':p.context.clone(),
            'transformer_options':{'cond_or_uncond':[0]},
        },
        'cond_or_uncond':[0],
    }
    try:
        with pytest.raises(NAGError,match='patched|changed after NAG setup'):
            s.wrapper(p.base.model.apply_model,payload)
    finally:
        del block.forward
        s.close()
