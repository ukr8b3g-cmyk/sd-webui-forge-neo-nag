from dataclasses import replace
import pytest
import torch

from forge_neo_nag.adapters.krea2 import Krea2Adapter, nag_block
from forge_neo_nag.config import NAGConfig, NAGError
from .helpers import TinyDiT, TinyBlock, OPS, attention, rope
from .test_math_config import reference


def reference_block(block, ptext, ntext, image, vec, pfreq, nfreq, cfg, options, ops=OPS):
    """Unoptimized two-full-projection oracle for the upstream NAG block equations."""
    s, h, g, s2, h2, g2 = block.mod(vec)
    pl, nl = ptext.shape[1], ntext.shape[1]
    p = torch.cat((ptext, image), 1)
    n = torch.cat((ntext, image), 1)

    def raw(x, freq):
        attn = block.attn
        q = attn.wq(x).unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        k = attn.wk(x).unflatten(-1, (attn.kvheads, -1)).transpose(1, 2)
        v = attn.wv(x).unflatten(-1, (attn.kvheads, -1)).transpose(1, 2)
        gate = attn.gate(x)
        q, k = attn.qknorm(q, k)
        q, k = ops.rope(q, k, freq)
        factor = attn.heads // attn.kvheads
        k, v = k.repeat_interleave(factor, 1), v.repeat_interleave(factor, 1)
        return ops.attention(q,k,v,attn.heads,mask=None,skip_reshape=True,transformer_options=options), gate

    pr, pg = raw(torch.addcmul(h, 1+s, block.prenorm(p)), pfreq)
    nr, ng = raw(torch.addcmul(h, 1+s, block.prenorm(n)), nfreq)
    guided = reference(pr[:,pl:],nr[:,nl:],cfg.phi,cfg.tau,cfg.alpha).to(pr.dtype)
    pr = torch.cat((pr[:,:pl],guided),1)
    p = torch.addcmul(p,g,block.attn.wo(pr*torch.sigmoid(pg)))
    ntext = torch.addcmul(ntext,g,block.attn.wo(nr[:,:nl]*torch.sigmoid(ng[:,:nl])))
    p = torch.addcmul(p,g2,block.mlp(torch.addcmul(h2,1+s2,block.postnorm(p))))
    ntext = torch.addcmul(ntext,g2,block.mlp(torch.addcmul(h2,1+s2,block.postnorm(ntext))))
    return p[:,:pl],ntext,p[:,pl:]


@pytest.mark.parametrize("batch", [1,2])
@pytest.mark.parametrize("pl,nl", [(3,7),(8,1)])
@pytest.mark.parametrize("kvheads", [1,2,4])
def test_shared_projection_matches_full_projection(batch,pl,nl,kvheads):
    model = TinyDiT(kvheads=kvheads)
    block = model.blocks[0]
    image = torch.randn(batch,6,64)
    pt, nt = torch.randn(batch,pl,64), torch.randn(batch,nl,64)
    vec = torch.randn(batch,1,64*6)*.2
    image_ids = torch.zeros(batch,6,3)
    image_ids[:,:,1] = torch.arange(6)
    pfreq = model.pe_embedder(torch.cat((torch.zeros(batch,pl,3),image_ids),1))
    nfreq = model.pe_embedder(torch.cat((torch.zeros(batch,nl,3),image_ids),1))
    ntfreq = model.pe_embedder(torch.zeros(batch,nl,3))
    cfg = NAGConfig.parse(True,"wings")
    expected = reference_block(block,pt,nt,image,vec,pfreq,nfreq,cfg,{})
    got = nag_block(block,pt,nt,image,vec,pfreq,ntfreq,cfg,{},OPS)
    for actual, ref in zip(got,expected):
        torch.testing.assert_close(actual,ref,rtol=2e-5,atol=2e-6)
    assert not torch.equal(got[1],nt), "Negative text must advance between blocks."


def test_image_queries_keys_values_are_identical_and_projected_once():
    model = TinyDiT()
    block = model.blocks[0]
    pl,nl,il = 3,7,6
    image,pt,nt,vec = torch.randn(1,il,64),torch.randn(1,pl,64),torch.randn(1,nl,64),torch.randn(1,1,384)
    pf = model.pe_embedder(torch.zeros(1,pl+il,3))
    nf = model.pe_embedder(torch.zeros(1,nl,3))
    recorded, projected = [], []
    def observe(q,k,v,*args,**kwargs):
        recorded.append((q.clone(),k.clone(),v.clone()))
        return attention(q,k,v,*args,**kwargs)
    handle = block.attn.wq.register_forward_hook(lambda module,args,out: projected.append(args[0].shape[1]))
    try:
        nag_block(block,pt,nt,image,vec,pf,nf,NAGConfig.parse(True,"wings"),{},replace(OPS,attention=observe))
    finally:
        handle.remove()
    assert projected == [pl+il,nl]  # not [pl+il,nl+il]
    for pos,neg in zip(recorded[0],recorded[1]):
        assert torch.equal(pos[:,:,pl:],neg[:,:,nl:])


@pytest.mark.parametrize("batch,temporal,h,w", [(1,False,4,6),(2,True,4,6),(2,True,5,7),(1,False,7,5)])
def test_forward_shape_cache_and_inputs_unchanged(batch,temporal,h,w):
    model = TinyDiT()
    x = torch.randn(batch,4,h,w)
    if temporal:
        x = x.unsqueeze(2)
    ctx = torch.randn(batch,8,3,16)
    neg = torch.randn(1,5,3,16)
    x0,ctx0,neg0 = x.clone(),ctx.clone(),neg.clone()
    adapter = Krea2Adapter(model,neg,NAGConfig.parse(True,"wings"),OPS)
    y1 = adapter(x,torch.ones(batch),ctx)
    y2 = adapter(x,torch.ones(batch),ctx)
    assert y1.shape == x.shape
    assert torch.equal(y1,y2)
    assert adapter.text_fusion_calls == 1
    assert torch.equal(ctx,ctx0) and torch.equal(neg,neg0) and torch.equal(x,x0)
    adapter.clear()
    assert adapter.negative_context is None and adapter._negative_text is None
    with pytest.raises(NAGError,match="released"):
        adapter(x,torch.ones(batch),ctx)


@pytest.mark.parametrize("dtype", [torch.float32,torch.float16,torch.bfloat16])
def test_native_path_equivalence_at_zero_alpha(dtype):
    model = TinyDiT().to(dtype)
    x,ctx,neg = torch.randn(2,4,1,4,6).to(dtype),torch.randn(2,8,3,16).to(dtype),torch.randn(1,5,3,16).to(dtype)
    t = torch.ones(2)
    cfg = NAGConfig.parse(True,"wings",alpha=0)
    result = Krea2Adapter(model,neg,cfg,OPS)(x,t,ctx)
    base = model(x,t,ctx)
    torch.testing.assert_close(result,base,atol=1e-6 if dtype==torch.float32 else .01,
                               rtol=1e-5 if dtype==torch.float32 else .01)


def test_negative_text_changes_computation():
    model = TinyDiT()
    x,ctx = torch.randn(1,4,1,4,6),torch.randn(1,8,3,16)
    cfg = NAGConfig.parse(True,"wings")
    a = Krea2Adapter(model,torch.randn(1,5,3,16),cfg,OPS)(x,torch.ones(1),ctx)
    b = Krea2Adapter(model,torch.randn(1,5,3,16),cfg,OPS)(x,torch.ones(1),ctx)
    assert not torch.allclose(a,b,atol=1e-7,rtol=1e-7)


@pytest.mark.parametrize("case", ["video","context","uncond","control","reference"])
def test_reject_invalid_adapter_layout(case):
    model = TinyDiT()
    adapter = Krea2Adapter(model,torch.randn(1,5,3,16),NAGConfig.parse(True,"wings"),OPS)
    x,ctx,kwargs = torch.randn(1,4,1,4,6),torch.randn(1,8,3,16),{}
    if case=="video": x=x.expand(-1,-1,2,-1,-1)
    if case=="context": ctx=torch.randn(1,8,48)
    if case=="uncond": kwargs={"transformer_options":{"cond_or_uncond":[1]}}
    if case=="control": kwargs={"control":{}}
    if case=="reference": kwargs={"ref_latents":[torch.zeros(1)]}
    with pytest.raises(NAGError):
        adapter(x,torch.ones(1),ctx,**kwargs)
