"""Optional native-method CPU tests. No model weights or full Forge runtime."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from forge_neo_nag.config import NAGConfig,NAGError
from forge_neo_nag.host import arm_request,encode_negative,validate_patcher,validate_request
from forge_neo_nag.adapters.sdxl import enumerate_attn2
from .native_multi_helpers import Processing,session


@pytest.mark.parametrize("family",["sdxl","anima"])
@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16])
@pytest.mark.parametrize("batch",[1,2])
def test_native_zero_alpha_full_forward_and_single_output_projection(family,dtype,batch):
    p=Processing(family,batch,dtype)
    cfg=NAGConfig.parse(True,"glasses",alpha=0,adapter=family)
    baseline=p.evaluate(p.payload(p.x,.5,[0]))
    s=session(p,cfg)
    model=p.base.model.diffusion_model
    layers=list(s.wrapper.adapter.layers.values()) if family=="sdxl" else [b.cross_attn for b in model.blocks]
    counts=[0]*len(layers);handles=[]
    for i,layer in enumerate(layers):
        projection=layer.to_out if family=="sdxl" else layer.output_proj
        def observe(module,args,out,i=i):counts[i]+=1
        handles.append(projection.register_forward_hook(observe))
    try:
        got=p.evaluate(p.payload(p.x,.5,[0]))
        torch.testing.assert_close(got,baseline,atol=1e-6 if dtype==torch.float32 else .004,rtol=1e-5 if dtype==torch.float32 else .01)
        assert counts==[1]*len(layers)
    finally:
        for handle in handles:handle.remove()
        s.close()


@pytest.mark.parametrize("family",["sdxl","anima"])
@pytest.mark.parametrize("labels",[[0,1],[1,0]])
@pytest.mark.parametrize("batch",[1,2])
def test_native_mixed_cfg_guides_positive_only(family,labels,batch):
    p=Processing(family,batch)
    payload=p.payload(p.x,.5,labels)
    baseline=p.evaluate(copy.deepcopy(payload))
    weights={k:v.clone() for k,v in p.base.model.diffusion_model.state_dict().items()}
    s=session(p)
    try:
        payload=p.payload(p.x,.5,labels)
        x0=payload['input'].clone();ctx0=payload['c']['c_crossattn'].clone()
        options=payload['c']['transformer_options'];keys=set(options)
        y0=payload['c'].get('y');y0=None if y0 is None else y0.clone()
        got=p.evaluate(payload)
        u=slice(labels.index(1)*batch,(labels.index(1)+1)*batch)
        c=slice(labels.index(0)*batch,(labels.index(0)+1)*batch)
        # Exact on CPU SDPA for untouched rows in the same-size native batch.
        assert torch.equal(got[u],baseline[u])
        assert not torch.equal(got[c],baseline[c])
        assert torch.isfinite(got).all()
        assert torch.equal(payload['input'],x0) and torch.equal(payload['c']['c_crossattn'],ctx0)
        assert set(options)==keys
        if y0 is not None:assert torch.equal(payload['c']['y'],y0)
        assert s.wrapper.adapter.attention_calls==len(s.wrapper.adapter.targets)
        for k,v in p.base.model.diffusion_model.state_dict().items():assert torch.equal(v,weights[k])
    finally:s.close()
    assert p.base.model_options=={'transformer_options':{}}


@pytest.mark.parametrize('family',['sdxl','anima'])
@pytest.mark.parametrize('cfg',[0.,.5,1.,2.,7.5])
@pytest.mark.parametrize('separate',[False,True])
def test_native_lifecycle_regular_cfg_and_off_restore(family,cfg,separate):
    p=Processing(family,batch=2);p.cfg_scale=cfg;p.separate=separate
    baseline=p.sample();original_forward=p.base.model.diffusion_model.forward
    original_scripts=p.scripts
    original_state=(p.cfg_scale,p.width,p.height,p.sampler_name,p.negative_prompt)
    arm_request(p,(True,'glasses',2.,2.5,.25,1000.,0.,family),lambda:p.bindings,preset='unrelated')
    got=p.generate_and_save()
    assert torch.isfinite(got).all()
    assert p.sd_model.encode_calls==1 and p.saved==1
    assert p.sd_model.forge_objects.unet is p.base and p.scripts is original_scripts
    assert p.base.model.diffusion_model.forward==original_forward
    assert original_state==(p.cfg_scale,p.width,p.height,p.sampler_name,p.negative_prompt)
    if cfg != 0:assert not torch.equal(got,baseline)
    else:torch.testing.assert_close(got,baseline,atol=1e-6,rtol=1e-5)
    meta=p.extra_generation_params
    assert meta['Forge NAG Adapter']==family
    assert meta['Forge NAG Adapter Selection']==family
    assert meta['Forge NAG Active Calls']==len(p.sigmas)
    assert 'Forge NAG Text Fusion Calls' not in meta
    assert p.observed_wrappers[-1].adapter.negative_context is None
    arm_request(p,(False,),lambda:pytest.fail('OFF loaded backend'))
    assert torch.equal(p.sample(),baseline)
    assert p.sd_model.encode_calls==1 and p.extra_generation_params=={}


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_separate_uncond_does_no_negative_work_and_matches_native(family):
    p=Processing(family,2)
    expected=p.evaluate(p.payload(p.x,.5,[1]))
    s=session(p)
    try:
        actual=p.evaluate(p.payload(p.x,.5,[1]))
        assert torch.equal(actual,expected)
        assert s.wrapper.active_calls==0 and s.wrapper.adapter.attention_calls==0
        assert s.wrapper.adapter._context is None
    finally:s.close()


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_separate_context_lengths_supported(family):
    p=Processing(family,2);p.uncond=torch.randn(2,5,24);p.separate=True
    arm_request(p,(True,'glasses'),lambda:p.bindings)
    assert torch.isfinite(p.sample()).all()


@pytest.mark.parametrize('family',['sdxl','anima'])
@pytest.mark.parametrize('kind',['exception','interrupt','late_clip_change','nonfinite','nonfinite_uncond','no_targets'])
def test_failure_restores_without_saving(family,kind):
    p=Processing(family);original=p.scripts
    if kind=='exception':
        p.after_hook=lambda:(_ for _ in ()).throw(ValueError('test failure'))
        error=ValueError
    elif kind=='interrupt':
        p.after_hook=lambda:(_ for _ in ()).throw(KeyboardInterrupt())
        error=KeyboardInterrupt
    elif kind=='late_clip_change':
        p.scripts.hooks.append(lambda p,**kw:setattr(p.sd_model.text_processing_engine_l,'clip_skip',5))
        error=NAGError
    elif kind=='nonfinite':
        p.context.fill_(float('nan'));error=NAGError
    elif kind=='nonfinite_uncond':
        p.uncond.fill_(float('nan'));p.separate=True;error=NAGError
    else:
        # An additional expected target is never executed; the call must fail.
        def mutate():
            w=p.sd_model.forge_objects.unet.model_options['model_function_wrapper']
            w.adapter.targets=tuple(w.adapter.targets)+('missing',)
        p.after_hook=mutate;error=NAGError
    arm_request(p,(True,'glasses'),lambda:p.bindings)
    with pytest.raises(error):p.generate_and_save()
    assert p.saved==0 and p.scripts is original and p.sd_model.forge_objects.unet is p.base
    assert p.base.model_options=={'transformer_options':{}}
    p.after_hook=None;p.scripts.hooks=[]
    p.context=torch.randn_like(p.context);p.uncond=torch.randn_like(p.uncond)
    arm_request(p,(False,))
    assert torch.isfinite(p.generate_and_save()).all()


@pytest.mark.parametrize('family',['sdxl','anima'])
@pytest.mark.parametrize('mutation',['forward','model_swap','unknown_mask','unknown_patch'])
def test_late_boundary_mutations_rejected(family,mutation):
    p=Processing(family);s=session(p)
    try:
        payload=p.payload(p.x,.5,[0]);model=p.base.model.diffusion_model
        if mutation=='forward':model.forward=lambda *a,**kw:p.x
        elif mutation=='model_swap':p.base.model.diffusion_model=Processing(family).base.model.diffusion_model
        elif mutation=='unknown_mask':payload['c']['transformer_options']['attention_mask']=torch.ones(1)
        else:payload['c']['transformer_options']['patches']={'unknown':lambda:None}
        with pytest.raises(NAGError):p.evaluate(payload)
    finally:s.close()


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_sigma_native_bypass_and_zero_active_failure(family):
    p=Processing(family)
    baseline=p.evaluate(p.payload(p.x,.9,[0]))
    s=session(p,NAGConfig.parse(True,'glasses',sigma_start=.6,sigma_end=.2))
    try:
        assert torch.equal(p.evaluate(p.payload(p.x,.9,[0])),baseline)
        assert s.wrapper.adapter.attention_calls==0
        p.evaluate(p.payload(p.x,.5,[0]));assert s.wrapper.active_calls==1
    finally:s.close()
    arm_request(p,(True,'glasses',2,2.5,.25,.04,.02),lambda:p.bindings)
    with pytest.raises(NAGError,match='No sampled sigma'):p.generate_and_save()
    assert p.saved==0


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_encoder_native_flag_shape_and_positive_pooled_not_replaced(family):
    p=Processing(family,2);cfg=NAGConfig.parse(True,'glasses')
    negative=encode_negative(p,cfg,p.bindings)
    assert negative.shape==(1,77 if family=='sdxl' else 512,24)
    assert torch.equal(negative,p.sd_model.negative)
    prompt=p.sd_model.prompts[-1]
    assert prompt.width==p.width and prompt.height==p.height and prompt.is_negative_prompt
    assert p.context.shape[1]==9 and p.y.shape==(2,16)
    assert not (p.y==999).any()


@pytest.mark.parametrize('index',[0,1])
def test_both_sdxl_clip_chunk_limits_and_four_chunk_boundary(index):
    p=Processing('sdxl');cfg=NAGConfig.parse(True,'glasses')
    p.sd_model.chunks[index]=4;p.sd_model.negative=torch.randn(1,308,24)
    assert encode_negative(p,cfg,p.bindings).shape[1]==308
    p.sd_model.chunks[index]=5
    with pytest.raises(NAGError,match='308'):encode_negative(p,cfg,p.bindings)
    assert p.sd_model.encode_calls==1


@pytest.mark.parametrize('kind',['textual_inversion','implicit_emphasis','encoded_oversize','wrong_width','nonfinite'])
def test_sdxl_encoder_rejects_unsupported_inputs(kind):
    p=Processing('sdxl');cfg=NAGConfig.parse(True,'glasses')
    if kind=='textual_inversion':p.sd_model.fixes=[object()]
    if kind=='implicit_emphasis':cfg=NAGConfig.parse(True,'(glasses)')
    if kind=='encoded_oversize':p.sd_model.negative=torch.randn(1,385,24)
    if kind=='wrong_width':p.sd_model.negative=torch.randn(1,77,23)
    if kind=='nonfinite':p.sd_model.negative.fill_(float('nan'))
    with pytest.raises(NAGError):encode_negative(p,cfg,p.bindings)


@pytest.mark.parametrize('index',[0,1])
def test_both_anima_tokenizer_limits(index):
    p=Processing('anima');p.sd_model.tokens[index]=2049
    with pytest.raises(NAGError,match='2048'):encode_negative(p,NAGConfig.parse(True,'glasses'),p.bindings)
    assert p.sd_model.encode_calls==0


@pytest.mark.parametrize('mutation',['delete','replace','extra'])
def test_sdxl_patch_ownership_not_just_callback_name(mutation):
    p=Processing('sdxl');s=session(p)
    try:
        payload=p.payload(p.x,.5,[0]);options=payload['c']['transformer_options']
        owned=dict(options['patches_replace']['attn2']);key=next(iter(owned))
        if mutation=='delete':del owned[key]
        if mutation=='replace':owned[key]=lambda *a:None
        if mutation=='extra':owned[('input',900,0)]=lambda *a:None
        options['patches_replace']={'attn2':owned}
        with pytest.raises(NAGError):p.evaluate(payload)
    finally:s.close()


def test_sdxl_targets_use_local_block_index_and_no_fixed_layer_count():
    p=Processing('sdxl');targets=enumerate_attn2(p.base.model.diffusion_model,p.bindings.layout_types)
    assert len(targets)==11
    assert ('middle',0,0) in targets and ('middle',0,1) in targets
    model=p.base.model.diffusion_model
    spatial=model.middle_block[1]
    model.middle_block.append(spatial)
    with pytest.raises(NAGError,match='collide'):enumerate_attn2(model,p.bindings.layout_types)


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_third_party_patcher_never_overwritten_during_close(family):
    p=Processing(family);s=session(p)
    foreign=SimpleNamespace()
    p.sd_model.forge_objects.unet=foreign
    s.close()
    assert p.sd_model.forge_objects.unet is foreign


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_model_mismatch_manual_does_not_patch_wrong_model(family):
    p=Processing(family)
    b=replace(p.bindings,engine_type=type('Other',(),{}))
    with pytest.raises(NAGError,match='Choose a compatible adapter'):validate_request(p,b)
    assert p.sd_model.forge_objects.unet is p.base


@pytest.mark.parametrize('family',['sdxl','anima'])
@pytest.mark.parametrize('cfg',[0.,.5,1.,2.,7.5])
def test_actual_native_cfg_function_owns_combination(family,cfg):
    from .native_multi_helpers import source_file,definitions
    ns=definitions(source_file('backend/sampling/sampling_function.py','sampling_methods.py'),{'sampling_function_inner'})
    ns['dynamic_args']=SimpleNamespace(context_handler=None)
    p=Processing(family,2);p.cfg_scale=cfg;s=session(p)
    seen=[]
    def calc(model,cond,uncond,x,timestep,options):
        seen.append(uncond is None)
        if uncond is None:
            return p.evaluate(p.payload(x,.5,[0])),torch.zeros_like(x)
        # Native host commonly reverses batch order; UNCOND precedes COND.
        u,c=p.evaluate(p.payload(x,.5,[1,0])).chunk(2)
        return c,u
    ns['calc_cond_uncond_batch']=calc
    try:
        out,c,u=ns['sampling_function_inner'](p.base.model,p.x,torch.ones(2)*.5,[{}],[{}],cfg,return_full=True)
        assert torch.equal(out,u+(c-u)*cfg)
        assert seen==[cfg==1.]
        assert s.wrapper.active_calls==1
        assert 'sampler_cfg_function' not in p.sd_model.forge_objects.unet.model_options
    finally:s.close()


@pytest.mark.parametrize('family',['sdxl','anima'])
def test_positive_composition_rejected_even_if_host_would_split_calls(family):
    p=Processing(family);p.prompt='girl AND landscape';p.separate=True
    arm_request(p,(True,'glasses'),lambda:p.bindings)
    with pytest.raises(NAGError,match='AND composition'):p.generate_and_save()
    assert p.saved==0 and p.sd_model.encode_calls==0
