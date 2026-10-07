"""Reduced CPU contract tests for the Flux.2 / Klein NAG adapter.

These are random tiny models that mirror the reviewed Forge Neo block contracts;
they do not load a Klein checkpoint and are not a GPU/image-quality test.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from forge_neo_nag.adapters.klein import KleinAdapter, encode_negative, validate_model
from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.host import HostBindings, SamplingSession, validate_request
from .helpers import Conditioning, Patcher, Predictor, Runner


@dataclass
class ModOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    def __init__(self, dim, double):
        super().__init__()
        self.double = double
        self.lin = nn.Linear(dim, dim * (6 if double else 3))

    def forward(self, vec):
        if vec.ndim == 2:
            vec = vec[:, None]
        values = self.lin(F.silu(vec)).chunk(6 if self.double else 3, -1)
        return ModOut(*values[:3]), (ModOut(*values[3:]) if self.double else None)


def apply_mod(tensor, mult, add=None, dims=None):
    assert dims is None
    return torch.addcmul(add, tensor, mult) if add is not None else tensor * mult


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.RMSNorm(dim)
        self.k = nn.RMSNorm(dim)

    def forward(self, q, k, v):
        return self.q(q).to(v), self.k(k).to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim=32, heads=4):
        super().__init__()
        self.num_heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.norm = QKNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)


def attention(q, k, v, pe=None, mask=None, transformer_options=None):
    assert mask is None
    if pe is not None:
        # Lightweight position transform with the same per-token independence as RoPE.
        pos = pe.to(q).transpose(1, 2).unsqueeze(-1) * 0.001
        q = q + pos
        k = k + pos
    y = F.scaled_dot_product_attention(q, k, v)
    return y.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


def fp16_fix(x):
    return x


class DoubleStreamBlock(nn.Module):
    def __init__(self, dim=32, heads=4, modulation=False):
        super().__init__()
        self.num_heads, self.hidden_size, self.modulation = heads, dim, modulation
        if modulation:
            self.img_mod, self.txt_mod = Modulation(dim, True), Modulation(dim, True)
        self.img_norm1, self.img_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.txt_norm1, self.txt_norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.img_attn, self.txt_attn = SelfAttention(dim, heads), SelfAttention(dim, heads)
        self.img_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.txt_mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, img, txt, vec, pe, attn_mask=None, modulation_dims_img=None,
                modulation_dims_txt=None, transformer_options=None):
        assert modulation_dims_img is modulation_dims_txt is None
        if self.modulation:
            im1, im2 = self.img_mod(vec)
            tm1, tm2 = self.txt_mod(vec)
        else:
            (im1, im2), (tm1, tm2) = vec
        ip = apply_mod(self.img_norm1(img), 1 + im1.scale, im1.shift)
        tp = apply_mod(self.txt_norm1(txt), 1 + tm1.scale, tm1.shift)
        def qkv(attn, value):
            q, k, v = attn.qkv(value).view(value.shape[0], value.shape[1], 3, self.num_heads, -1).permute(2,0,3,1,4)
            q, k = attn.norm(q, k, v)
            return q, k, v
        iq,ik,iv=qkv(self.img_attn,ip); tq,tk,tv=qkv(self.txt_attn,tp)
        raw=attention(torch.cat((tq,iq),2),torch.cat((tk,ik),2),torch.cat((tv,iv),2),pe=pe)
        tr,ir=raw[:,:txt.shape[1]],raw[:,txt.shape[1]:]
        img=img+apply_mod(self.img_attn.proj(ir),im1.gate)
        img=img+apply_mod(self.img_mlp(apply_mod(self.img_norm2(img),1+im2.scale,im2.shift)),im2.gate)
        txt=txt+apply_mod(self.txt_attn.proj(tr),tm1.gate)
        txt=txt+apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt),1+tm2.scale,tm2.shift)),tm2.gate)
        return img,txt


class SiLUActivation(nn.Module):
    def forward(self, value):
        a, b = value.chunk(2, dim=-1)
        return F.silu(a) * b


class SingleStreamBlock(nn.Module):
    def __init__(self, dim=32, heads=4, modulation=False):
        super().__init__()
        self.hidden_dim=self.hidden_size=dim; self.num_heads=heads
        self.modulation=Modulation(dim,False) if modulation else None
        self.mlp_hidden_dim=dim*3; self.mlp_hidden_dim_first=self.mlp_hidden_dim*2
        self.yak_mlp=False; self.mlp_act=SiLUActivation()
        self.linear1=nn.Linear(dim,dim*3+self.mlp_hidden_dim_first,bias=False)
        self.linear2=nn.Linear(dim+self.mlp_hidden_dim,dim)
        self.norm=QKNorm(dim//heads); self.pre_norm=nn.LayerNorm(dim)

    def forward(self,x,vec,pe,attn_mask=None,modulation_dims=None,transformer_options=None):
        if self.modulation:
            mod,_=self.modulation(vec)
        else:
            mod=vec
        qkv,mlp=torch.split(self.linear1(apply_mod(self.pre_norm(x),1+mod.scale,mod.shift)),[3*self.hidden_size,self.mlp_hidden_dim_first],-1)
        q,k,v=qkv.view(qkv.shape[0],qkv.shape[1],3,self.num_heads,-1).permute(2,0,3,1,4)
        q,k=self.norm(q,k,v)
        raw=attention(q,k,v,pe=pe)
        out=self.linear2(torch.cat((raw,self.mlp_act(mlp)),2))
        return x+apply_mod(out,mod.gate)


class Embed(nn.Module):
    def forward(self, ids):
        return ids.sum(-1, keepdim=True)


class Final(nn.Module):
    def __init__(self, dim=32, out=4):
        super().__init__(); self.proj=nn.Linear(dim,out)
    def forward(self,x,vec): return self.proj(x)


class TinyFlux2(nn.Module):
    def __init__(self, context_dim=12, dim=32, double=2, single=2):
        super().__init__()
        self.patch_size=1; self.in_channels=4; self.out_channels=4
        self.hidden_size=dim; self.num_heads=4; self.axes_dim=[32,32,32,32]; self.txt_ids_dims=[3]
        self.global_modulation=True; self.default_ref_method='index'
        self.img_in=nn.Linear(4,dim,bias=False); self.txt_in=nn.Linear(context_dim,dim,bias=False); self.txt_norm=nn.RMSNorm(context_dim)
        self.time_in=nn.Sequential(nn.Linear(1,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.pe_embedder=Embed(); self.double_blocks=nn.ModuleList([DoubleStreamBlock(dim,4,modulation=False) for _ in range(double)])
        self.single_blocks=nn.ModuleList([SingleStreamBlock(dim,4,modulation=False) for _ in range(single)])
        self.double_stream_modulation_img=Modulation(dim,True); self.double_stream_modulation_txt=Modulation(dim,True)
        self.single_stream_modulation=Modulation(dim,False)
        self.final_layer=Final(dim,4)

    def process_img(self,x,index=0,h_offset=0,w_offset=0):
        b,c,h,w=x.shape; seq=x.permute(0,2,3,1).reshape(b,h*w,c)
        ids=torch.zeros(b,h*w,len(self.axes_dim),device=x.device,dtype=torch.float32)
        ids[:,:,0]=index; ids[:,:,1]=torch.arange(h,device=x.device).repeat_interleave(w); ids[:,:,2]=torch.arange(w,device=x.device).repeat(h)
        return seq,ids

    def forward_orig(self,img,img_ids,txt,txt_ids,timesteps,y=None,guidance=None,control=None,transformer_options=None,attn_mask=None):
        options=transformer_options or {}; patches=options.get('patches',{}); replace=options.get('patches_replace',{}).get('dit',{})
        img=self.img_in(img); vec=self.time_in(timesteps[:,None].to(img.dtype)); txt=self.txt_in(self.txt_norm(txt))
        vec_orig=vec
        if self.global_modulation:
            vec=(self.double_stream_modulation_img(vec_orig),self.double_stream_modulation_txt(vec_orig))
        if 'post_input' in patches:
            for fn in patches['post_input']:
                out=fn({'img':img,'txt':txt,'img_ids':img_ids,'txt_ids':txt_ids}); img,txt,img_ids,txt_ids=out['img'],out['txt'],out['img_ids'],out['txt_ids']
        pe=self.pe_embedder(torch.cat((txt_ids,img_ids),1))
        for i,block in enumerate(self.double_blocks):
            args={'img':img,'txt':txt,'vec':vec,'pe':pe,'attn_mask':attn_mask,'transformer_options':options}
            if ('double_block',i) in replace: out=replace[('double_block',i)](args,{'original_block':None}); img,txt=out['img'],out['txt']
            else: img,txt=block(**args)
        pos_len=txt.shape[1]; seq=torch.cat((txt,img),1)
        if self.global_modulation:
            vec,_=self.single_stream_modulation(vec_orig)
        for i,block in enumerate(self.single_blocks):
            args={'img':seq,'vec':vec,'pe':pe,'attn_mask':attn_mask,'transformer_options':options}
            if ('single_block',i) in replace: seq=replace[('single_block',i)](args,{'original_block':None})['img']
            else: seq=block(seq,vec=vec,pe=pe,attn_mask=attn_mask,transformer_options=options)
        return self.final_layer(seq[:,pos_len:],vec_orig)

    def forward(self,x,timestep,context,y=None,guidance=None,control=None,transformer_options=None,**kwargs):
        b,c,h,w=x.shape; img,img_ids=self.process_img(x)
        txt_ids=torch.zeros(b,context.shape[1],len(self.axes_dim),device=x.device)
        txt_ids[:,:,3]=torch.linspace(0,context.shape[1]-1,context.shape[1],device=x.device)
        out=self.forward_orig(img,img_ids,context,txt_ids,timestep,y,guidance,control,transformer_options,kwargs.get('attention_mask'))
        return out.reshape(b,h,w,c).permute(0,3,1,2).contiguous()


OPS=SimpleNamespace(attention=attention,apply_mod=apply_mod,fp16_fix=fp16_fix)


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
        self.text_processing_engine_qwen=SimpleNamespace(tokenize=lambda text:list(range(512)))
        self.encode_calls=0; self.negative=torch.randn(6,12)
    def get_learned_conditioning(self,prompt):
        assert prompt.is_negative_prompt; self.encode_calls+=1; return [self.negative.clone()]


class Processing:
    def __init__(self,batch=1):
        model=TinyFlux2(); self.sd_model=Engine(model); self.base=self.sd_model.forge_objects.unet; self.scripts=Runner()
        self.cfg_scale=1.; self.width=self.height=64; self.distilled_cfg_scale=1.; self.enable_hr=self.is_hr_pass=self.txt2img_upscale=self.tiling=False
        self.refiner_checkpoint=None; self.sampler_name='Euler'; self.extra_generation_params={}; self.prompt='portrait'; self.all_prompts=['portrait']
        self.x=torch.randn(batch,4,3,4); self.context=torch.randn(batch,5,12); self.uncond=torch.randn(batch,5,12)
        self.bindings=HostBindings(Engine,TinyFlux2,KModel,Processing,Conditioning,SimpleNamespace(ref_latents=[],context_handler=None),OPS,'klein',(DoubleStreamBlock,SingleStreamBlock,SelfAttention))


def adapter(p, alpha=.25, negative=None):
    neg=(negative if negative is not None else torch.randn(1,7,12))
    return KleinAdapter(p.base.model.diffusion_model,neg,NAGConfig.parse(True,'glasses',2.,2.5,alpha,1000,0,'klein'),OPS,p.bindings.layout_types)


def test_klein_model_contract_and_encoder_negative_flag():
    p=Processing(); assert validate_model(p.base.model.diffusion_model,p.bindings.layout_types)==12
    out=encode_negative(p,NAGConfig.parse(True,'glasses',adapter='klein'),p.bindings)
    assert out.shape==(1,6,12) and p.sd_model.encode_calls==1
    p.sd_model.text_processing_engine_qwen.tokenize=lambda text:list(range(2049))
    with pytest.raises(NAGError,match='2048'): encode_negative(p,NAGConfig.parse(True,'glasses',adapter='klein'),p.bindings)


@pytest.mark.parametrize('batch,pos_len,neg_len',[(1,5,7),(2,8,3)])
def test_klein_zero_alpha_matches_native_and_different_text_lengths(batch,pos_len,neg_len):
    p=Processing(batch); model=p.base.model.diffusion_model; x=p.x.clone(); ctx=torch.randn(batch,pos_len,12); t=torch.ones(batch)*.5
    base=model(x,t,ctx,transformer_options={})
    got=adapter(p,alpha=0,negative=torch.randn(1,neg_len,12))(x,t,ctx,transformer_options={'cond_or_uncond':[0]})
    torch.testing.assert_close(got,base,rtol=1e-5,atol=1e-6)


def test_klein_nonzero_negative_changes_output_and_inputs_weights_unchanged():
    p=Processing(); model=p.base.model.diffusion_model; x=p.x.clone(); ctx=p.context.clone(); t=torch.ones(1)*.5
    x0,ctx0=x.clone(),ctx.clone(); weights={k:v.clone() for k,v in model.state_dict().items()}
    a=adapter(p,negative=torch.zeros(1,7,12)); b=adapter(p,negative=torch.ones(1,7,12))
    ya=a(x,t,ctx,transformer_options={'cond_or_uncond':[0]}); yb=b(x,t,ctx,transformer_options={'cond_or_uncond':[0]})
    assert torch.isfinite(ya).all() and not torch.allclose(ya,yb,atol=1e-7,rtol=1e-7)
    assert a.attention_calls==len(model.double_blocks)+len(model.single_blocks)
    assert torch.equal(x,x0) and torch.equal(ctx,ctx0)
    for k,v in model.state_dict().items(): assert torch.equal(v,weights[k])


def test_klein_image_projections_are_shared_not_recomputed_for_negative_branch():
    p=Processing(); model=p.base.model.diffusion_model; a=adapter(p)
    counts={'img':0,'txt':0,'single':0}
    h1=model.double_blocks[0].img_attn.qkv.register_forward_hook(lambda *x:counts.__setitem__('img',counts['img']+1))
    h2=model.double_blocks[0].txt_attn.qkv.register_forward_hook(lambda *x:counts.__setitem__('txt',counts['txt']+1))
    h3=model.single_blocks[0].linear1.register_forward_hook(lambda *x:counts.__setitem__('single',counts['single']+1))
    try: a(p.x,torch.ones(1)*.5,p.context,transformer_options={'cond_or_uncond':[0]})
    finally: h1.remove();h2.remove();h3.remove()
    assert counts=={'img':1,'txt':2,'single':2}


@pytest.mark.parametrize('labels',[[0,1],[1,0]])
def test_klein_host_wrapper_preserves_native_unconditional_branch(labels):
    p=Processing(); p.cfg_scale=5.; cfg=NAGConfig.parse(True,'glasses',2,2.5,.25,1000,0,'klein')
    s=SamplingSession(p,cfg,p.bindings,torch.randn(1,7,12)); s.install(p,x=p.x)
    x=torch.cat((p.x,p.x)); context=torch.cat((p.context,p.uncond)); sigma=torch.ones(2)*.5
    # Arrange context to match the labels.
    context=torch.cat([p.context if label==0 else p.uncond for label in labels])
    c={'c_crossattn':context,'transformer_options':{'cond_or_uncond':labels}}
    ui=labels.index(1); part=slice(ui,ui+1)
    native=p.base.model.apply_model(x[part],sigma[part],c_crossattn=context[part],transformer_options={'cond_or_uncond':[1]})
    try:
        out=s.wrapper(p.base.model.apply_model,{'input':x,'timestep':sigma,'c':c,'cond_or_uncond':labels})
        torch.testing.assert_close(out[part],native,atol=0,rtol=0)
        assert s.wrapper.active_calls==1
    finally:s.close()


def test_klein_rejects_positive_and_reference_and_foreign_patch():
    p=Processing(); p.prompt='girl AND landscape'; p.all_prompts=[p.prompt]
    with pytest.raises(NAGError,match='AND composition'):validate_request(p,p.bindings)
    p.prompt='portrait';p.all_prompts=[p.prompt];p.bindings.dynamic_args.ref_latents=[torch.zeros(1)]
    with pytest.raises(NAGError,match='Reference'):validate_request(p,p.bindings)
    p.bindings.dynamic_args.ref_latents=[]
    a=adapter(p)
    with pytest.raises(NAGError,match='existing Flux'):
        a(p.x,torch.ones(1)*.5,p.context,transformer_options={'patches':{'foreign':[lambda x:x]}})
