from types import SimpleNamespace
import pytest
import torch

from forge_neo_nag.config import NAGConfig,NAGError
from forge_neo_nag.host import SamplingSession,validate_patcher,validate_transformer_options
from .helpers import Processing,make_bindings


@pytest.mark.parametrize('mutation', ['controlnet','object_patch','forward','block_forward','compiled'])
def test_patched_model_is_not_silently_bypassed(mutation):
    p=Processing()
    bindings=make_bindings()
    model=p.base.model.diffusion_model
    if mutation=='controlnet':p.base.controlnet_linked_list=object()
    if mutation=='object_patch':p.base.object_patches={'diffusion_model.forward':object()}
    if mutation=='forward':model.forward=lambda *a:None
    if mutation=='block_forward':model.blocks[0].forward=lambda *a:None
    if mutation=='compiled':model._orig_mod=model
    with pytest.raises(NAGError):validate_patcher(p.base,bindings)


def wrapper_setup(config=None):
    p=Processing(batch=2)
    session=SamplingSession(p,config or NAGConfig.parse(True,'wings'),make_bindings(),torch.randn(1,5,3,16))
    session.install(p,x=p.x)
    payload={'input':p.x.clone(),'timestep':torch.ones(2),
             'c':{'c_crossattn':p.context.clone(),'transformer_options':{'cond_or_uncond':[0]}},'cond_or_uncond':[0]}
    return p,session,payload


@pytest.mark.parametrize('case',['uncond','control','c_concat','sigma_nan','sigma_empty','sigma_batch','changed_method','late_patch'])
def test_model_boundary_errors(case):
    p,session,payload=wrapper_setup()
    method=p.base.model.apply_model
    if case=='uncond':payload['cond_or_uncond']=[1]
    if case=='control':payload['c']['control']={}
    if case=='c_concat':payload['c']['c_concat']=torch.zeros(1)
    if case=='sigma_nan':payload['timestep']=torch.full((2,),float('nan'))
    if case=='sigma_empty':payload['timestep']=torch.zeros(0)
    if case=='sigma_batch':payload['timestep']=torch.ones(3)
    if case=='changed_method':method=lambda *a,**kw:None
    if case=='late_patch':payload['c']['transformer_options']['patches']={'late':True}
    try:
        with pytest.raises(NAGError):session.wrapper(method,payload)
    finally:
        session.close()
    assert p.sd_model.forge_objects.unet is p.base


def test_mixed_range_sigma_does_not_silently_disable_guidance():
    p,session,payload=wrapper_setup(NAGConfig.parse(True,'wings',sigma_start=.6,sigma_end=.2))
    payload['timestep']=torch.tensor([.5,.8])
    try:
        with pytest.raises(NAGError,match='mix'):session.wrapper(p.base.model.apply_model,payload)
    finally:session.close()


def test_clone_isolation_required():
    p=Processing()
    p.base.clone=lambda:p.base
    session=SamplingSession(p,NAGConfig.parse(True,'wings'),make_bindings(),torch.randn(1,5,3,16))
    with pytest.raises(NAGError,match='isolate'):session.install(p,x=p.x)
    assert 'model_function_wrapper' not in p.base.model_options


def test_live_lora_style_linear_is_used_not_replaced():
    p,session,payload=wrapper_setup()
    linear=p.base.model.diffusion_model.blocks[0].attn.wq
    calls=[]
    hook=linear.register_forward_hook(lambda layer,args,out:calls.append(args[0].shape))
    try:
        session.wrapper(p.base.model.apply_model,payload)
        assert len(calls)==2
        assert p.base.model.diffusion_model.blocks[0].attn.wq is linear
    finally:
        hook.remove()
        session.close()


def test_sparse_attention_override_is_rejected_at_preflight():
    with pytest.raises(NAGError,match="optimized_attention_override"):
        validate_transformer_options({"optimized_attention_override":lambda *a,**k:None})


def test_sparse_attention_override_is_rejected_if_added_after_install():
    p,session,payload=wrapper_setup()
    payload["c"]["transformer_options"]["optimized_attention_override"]=lambda *a,**k:None
    try:
        with pytest.raises(NAGError,match="optimized_attention_override"):
            session.wrapper(p.base.model.apply_model,payload)
    finally:
        session.close()
    assert p.sd_model.forge_objects.unet is p.base
