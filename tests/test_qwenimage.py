"""Reduced native-contract Qwen-Image 2512 CPU tests: no pretrained checkpoints."""
from __future__ import annotations
import math
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from forge_neo_nag.adapters.qwenimage import QwenImageAdapter, validate_model, encode_negative
from forge_neo_nag.config import NAGConfig, NAGError


def rope(x, freqs):
    if x.shape[1] == 0: return x
    pair=x.reshape(*x.shape[:-1],-1,1,2)
    return (freqs[...,0]*pair[...,0]+freqs[...,1]*pair[...,1]).reshape(*x.shape)

class EmbedND(nn.Module):
    def __init__(self,axes=(4,4,8)):
        super().__init__();self.axes_dim=axes
    def forward(self,ids):
        parts=[]
        for axis,dim in enumerate(self.axes_dim):
            f=10000**(-torch.arange(0,dim,2,device=ids.device).double()/dim)
            angle=ids[:,:,axis,None]*f
            c,s=angle.cos(),angle.sin()
            parts.append(torch.stack((c,-s,s,c),-1).reshape(*angle.shape,2,2))
        return torch.cat(parts,2).float().unsqueeze(1)

def attention(q,k,v,heads,mask=None):
    b,l,d=q.shape;dh=d//heads
    q=q.view(b,l,heads,dh).transpose(1,2)
    k=k.view(b,-1,heads,dh).transpose(1,2)
    v=v.view(b,-1,heads,dh).transpose(1,2)
    return F.scaled_dot_product_attention(q,k,v,attn_mask=mask).transpose(1,2).reshape(b,l,d)

OPS=SimpleNamespace(rope=rope,attention=attention)

class Attention(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__()
        self.heads=heads;self.dim_head=dim//heads;self.inner_dim=dim
        self.norm_q=nn.RMSNorm(self.dim_head);self.norm_k=nn.RMSNorm(self.dim_head)
        self.norm_added_q=nn.RMSNorm(self.dim_head);self.norm_added_k=nn.RMSNorm(self.dim_head)
        for n in ('to_q','to_k','to_v','add_q_proj','add_k_proj','add_v_proj'):
            setattr(self,n,nn.Linear(dim,dim))
        self.to_out=nn.ModuleList((nn.Linear(dim,dim),nn.Dropout(0)))
        self.to_add_out=nn.Linear(dim,dim)
    def forward(self,hidden_states,encoder_hidden_states,encoder_hidden_states_mask=None,attention_mask=None,image_rotary_emb=None):
        b,ln,d=hidden_states.shape;bs,tn,td=encoder_hidden_states.shape
        h=self.heads
        project=lambda name,val:getattr(self,name)(val).unflatten(-1,(h,-1))
        qi=self.norm_q(project('to_q',hidden_states));ki=self.norm_k(project('to_k',hidden_states));vi=project('to_v',hidden_states)
        qt=self.norm_added_q(project('add_q_proj',encoder_hidden_states));kt=self.norm_added_k(project('add_k_proj',encoder_hidden_states));vt=project('add_v_proj',encoder_hidden_states)
        q=rope(torch.cat((qt,qi),1),image_rotary_emb).flatten(2)
        k=rope(torch.cat((kt,ki),1),image_rotary_emb).flatten(2)
        v=torch.cat((vt,vi),1).flatten(2)
        out=attention(q,k,v,h,mask=attention_mask)
        return self.to_out[1](self.to_out[0](out[:,tn:])),self.to_add_out(out[:,:tn])

class QwenImageTransformerBlock(nn.Module):
    def __init__(self,dim=64,heads=4):
        super().__init__()
        self.img_mod=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        self.txt_mod=nn.Sequential(nn.SiLU(),nn.Linear(dim,6*dim))
        self.img_norm1=nn.LayerNorm(dim,elementwise_affine=False)
        self.img_norm2=nn.LayerNorm(dim,elementwise_affine=False)
        self.txt_norm1=nn.LayerNorm(dim,elementwise_affine=False)
        self.txt_norm2=nn.LayerNorm(dim,elementwise_affine=False)
        self.img_mlp=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.txt_mlp=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
        self.attn=Attention(dim,heads)
    def _modulate(self,x,params):
        shift,scale,gate=params.chunk(3,-1)
        return torch.addcmul(shift.unsqueeze(1),x,1+scale.unsqueeze(1)),gate.unsqueeze(1)
    def forward(self,hidden_states,encoder_hidden_states,encoder_hidden_states_mask,temb,image_rotary_emb=None):
        im1,im2=self.img_mod(temb).chunk(2,-1)
        tx1,tx2=self.txt_mod(temb).chunk(2,-1)
        im,gim=self._modulate(self.img_norm1(hidden_states),im1)
        txt,gtx=self._modulate(self.txt_norm1(encoder_hidden_states),tx1)
        attn_img,attn_txt=self.attn(im,txt,encoder_hidden_states_mask=encoder_hidden_states_mask,image_rotary_emb=image_rotary_emb)
        hidden_states=hidden_states+gim*attn_img
        encoder_hidden_states=encoder_hidden_states+gtx*attn_txt
        im,gi=self._modulate(self.img_norm2(hidden_states),im2)
        tx,gt=self._modulate(self.txt_norm2(encoder_hidden_states),tx2)
        return torch.addcmul(encoder_hidden_states,gt,self.txt_mlp(tx)),torch.addcmul(hidden_states,gi,self.img_mlp(im))

class FinalLayer(nn.Module):
    def __init__(self,dim=64):
        super().__init__(); self.norm=nn.LayerNorm(dim,elementwise_affine=False); self.linear=nn.Linear(dim,dim*2)
    def forward(self,x,emb):
        scale,shift=self.linear(F.silu(emb)).chunk(2,dim=-1)
        return torch.addcmul(shift[:,None,:],self.norm(x),(1+scale)[:,None,:])

class TinyQwenImage(nn.Module):
    def __init__(self,context_dim=12,hidden=64,num_layers=3):
        super().__init__()
        self.patch_size=2;self.out_channels=4;self.in_channels=16;self.inner_dim=hidden
        self.img_in=nn.Linear(16,hidden)
        self.txt_norm=nn.RMSNorm(context_dim);self.txt_in=nn.Linear(context_dim,hidden)
        self.pe_embedder=EmbedND()
        self.time_text_embed=nn.Sequential(nn.Linear(1,hidden),nn.SiLU(),nn.Linear(hidden,hidden))
        self.transformer_blocks=nn.ModuleList([QwenImageTransformerBlock(hidden) for _ in range(num_layers)])
        self.norm_out=FinalLayer(hidden);self.proj_out=nn.Linear(hidden,16)
    def process_img(self,x,index=0,h_offset=0,w_offset=0):
        b,c,t,h,w=x.shape
        h0,w0=h,w
        y=F.pad(x,(0,(-w)%2,0,(-h)%2));shape=y.shape
        patches=y.view(b,c,h//2 if h%2==0 else (h+1)//2,2,w//2 if w%2==0 else (w+1)//2,2)
        patches=patches.permute(0,2,4,1,3,5).reshape(b,-1,c*4)
        ht,wt=(h+1)//2,(w+1)//2
        img_ids=torch.zeros(ht,wt,3,device=x.device)
        img_ids[:,:,0]=index
        img_ids[:,:,1]=torch.arange(ht,device=x.device).view(-1,1)-ht//2
        img_ids[:,:,2]=torch.arange(wt,device=x.device).view(1,-1)-wt//2
        return patches,img_ids.reshape(1,ht*wt,3).expand(b,-1,-1),shape
    def forward(self,x,timesteps,context,attention_mask=None,guidance=None,transformer_options=None,control=None,**kwargs):
        b,c,t,h,w=x.shape
        hidden_states,img_ids,orig_shape=self.process_img(x)
        num_embeds=hidden_states.shape[1]
        length=context.shape[1]
        start=round(max(((w+1)//2)//2,((h+1)//2)//2))
        txt_ids=torch.arange(start,start+length,device=x.device).reshape(1,-1,1).repeat(b,1,3)
        image_rotary_emb=self.pe_embedder(torch.cat((txt_ids,img_ids),1)).squeeze(1).unsqueeze(2).to(x.dtype)
        hidden_states=self.img_in(hidden_states)
        encoder_hidden_states=self.txt_in(self.txt_norm(context))
        temb=self.time_text_embed(timesteps[:,None].to(x.dtype))
        for i,block in enumerate(self.transformer_blocks):
            encoder_hidden_states,hidden_states=block(hidden_states=hidden_states,encoder_hidden_states=encoder_hidden_states,
                                                      encoder_hidden_states_mask=attention_mask,temb=temb,image_rotary_emb=image_rotary_emb)
        hidden_states=self.norm_out(hidden_states,temb)
        hidden_states=self.proj_out(hidden_states)
        hidden_states=hidden_states[:,:num_embeds].view(orig_shape[0],orig_shape[-2]//2,orig_shape[-1]//2,orig_shape[1],2,2)
        hidden_states=hidden_states.permute(0,3,1,4,2,5)
        return hidden_states.reshape(orig_shape)[:,:,:,:h,:w]

@pytest.fixture(autouse=True)
def seed():
    torch.set_num_threads(1);torch.manual_seed(4917)
    with torch.inference_mode(): yield


def create(batch=1,positive=5,negative=7,alpha=.25):
    model=TinyQwenImage().eval()
    cfg=NAGConfig.parse(True,'wings',2.5,2.5,alpha,1000,0,'auto')
    adapter=QwenImageAdapter(model,torch.randn(1,negative,12),cfg,OPS,(QwenImageTransformerBlock,Attention))
    x=torch.randn(batch,4,1,5,7)
    c=torch.randn(batch,positive,12)
    t=torch.ones(batch)*.5
    return model,adapter,x,c,t

@pytest.mark.parametrize('batch,positive,negative',[(1,3,7),(2,8,2),(1,1,1)])
def test_native_alpha_zero_parity(batch,positive,negative):
    model,a,x,c,t=create(batch,positive,negative,alpha=0)
    y=model(x,t,c)
    got=a(x,t,c)
    torch.testing.assert_close(y,got,rtol=1e-5,atol=1e-5)


def test_model_contract():
    model,a,x,c,t=create()
    assert validate_model(model,(QwenImageTransformerBlock,Attention))==12
    class OtherBlock(QwenImageTransformerBlock):pass
    model.transformer_blocks[0]=OtherBlock()
    with pytest.raises(NAGError,match='native'):validate_model(model,(QwenImageTransformerBlock,Attention))


def test_positive_negative_rope_uses_full_native_three_axes():
    model,a,x,c,t=create()
    seen=[]
    original=a.ops.rope
    def observe(tokens,pe):
        seen.append((tokens.shape,pe.clone()))
        return original(tokens,pe)
    a.ops=SimpleNamespace(rope=observe,attention=attention)
    a(x,t,c)
    assert len(seen)>2
    # Positive full attention has 5 text positions and native image positions.
    assert seen[0][0][1] == c.shape[1]+math.ceil(x.shape[-2]/2)*math.ceil(x.shape[-1]/2)
    assert seen[2][0][1] == 7  # negative context length, independently generated RoPE


def test_shared_image_qkv_and_negative_text_advances():
    model,a,x,c,t=create()
    values=[]
    def observe(q,k,v,heads,mask=None):
        values.append(tuple(vv.detach().clone() for vv in (q,k,v)))
        return attention(q,k,v,heads,mask)
    a.ops=SimpleNamespace(rope=rope,attention=observe)
    a(x,t,c)
    assert a.attention_calls==len(model.transformer_blocks)
    assert len(values)==2*len(model.transformer_blocks)
    pos_len=c.shape[1];neg_len=7
    for p,n in zip(values[::2],values[1::2]):
        for x1,x2 in zip(p,n):
            assert torch.equal(x1[:,pos_len:],x2[:,neg_len:])
    assert torch.isfinite(a(x,t,c)).all()


def test_late_patch_fail_closed_and_recovery():
    model,a,x,c,t=create()
    orig=model.transformer_blocks[0].forward
    model.transformer_blocks[0].forward=lambda *x,**k:None
    with pytest.raises(NAGError,match='patched'):
        a(x,t,c)
    model.transformer_blocks[0].forward=orig
    # An instance-level override remains a patch; deleting restores native dispatch.
    del model.transformer_blocks[0].forward
    assert torch.isfinite(a(x,t,c)).all()
    a.clear()
    with pytest.raises(NAGError,match='released'):a(x,t,c)

@pytest.mark.parametrize('bad', ['dim','ref','mask','guidance','patch'])
def test_reject_unsupported(bad):
    model,a,x,c,t=create()
    kwargs={}
    if bad=='dim':x=x[:,:,0]
    if bad=='ref':kwargs['ref_latents']=[torch.randn(1)]
    if bad=='mask':kwargs['attention_mask']=torch.ones(1,c.shape[1])
    if bad=='guidance':kwargs['guidance']=torch.ones(1)
    if bad=='patch':kwargs['transformer_options']={'patches':{'double_block':[1]}}
    with pytest.raises(NAGError):a(x,t,c,**kwargs)


def test_encode_negative_native_qwen():
    model,a,x,c,t=create()
    class Prompts(list):
        def __init__(self,text,**kw):super().__init__(text);self.__dict__.update(kw)
    class Engine:
        def __init__(self):
            self.forge_objects=SimpleNamespace(unet=SimpleNamespace(model=SimpleNamespace(diffusion_model=model)))
            self.text_processing_engine_qwen=SimpleNamespace(tokenize=lambda t:list(range(19)))
            self.negative_calls=0
        def get_learned_conditioning(self,prompt):
            assert prompt.is_negative_prompt
            self.negative_calls+=1
            return [torch.randn(4,12)]
    p=SimpleNamespace(sd_model=Engine(),width=64,height=64,distilled_cfg_scale=1)
    binding=SimpleNamespace(conditioning_type=Prompts,layout_types=(QwenImageTransformerBlock,Attention))
    got=encode_negative(p,a.config,binding)
    assert got.shape==(1,4,12) and p.sd_model.negative_calls==1
    p.sd_model.text_processing_engine_qwen.tokenize=lambda t:list(range(2049))
    with pytest.raises(NAGError,match='2048'):encode_negative(p,a.config,binding)
