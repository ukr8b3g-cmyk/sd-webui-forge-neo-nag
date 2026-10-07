from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch

from forge_neo_nag.config import NAGError, NAGConfig
from forge_neo_nag.host import (arm_request, disarm_request, SamplingSession, encode_negative,
                                validate_request, validate_patcher, memory_reserve)
from .helpers import Processing, make_bindings, TinyDiT, Patcher, TinyKModel

ARGS = (True,"big wings",4.0,2.5,.25,1000.,0.)


def enable(p, args=ARGS, bindings=None):
    bindings = bindings or make_bindings()
    arm_request(p,args,lambda:bindings)
    return bindings


def test_off_is_bit_exact_no_encoding_or_host_imports():
    p=Processing()
    baseline=p.sample()
    normal=p.sample
    arm_request(p,(False,"ignored"),lambda:pytest.fail("OFF loaded the backend"))
    assert p.sample==normal
    assert torch.equal(p.sample(),baseline)
    assert p.sd_model.encode_calls==0
    assert p.extra_generation_params=={}


@pytest.mark.parametrize("args", [(True,""),(True,"wings",0),(True,"wings",4,2.5,0)])
def test_zero_and_empty_use_native_path(args):
    p=Processing()
    baseline=p.sample()
    p.cfg_scale=9  # ignored while NAG is a true no-op
    arm_request(p,args,lambda:pytest.fail("bypass loaded the backend"))
    assert torch.equal(p.sample(),baseline)
    assert p.sd_model.encode_calls==0
    assert p.extra_generation_params["Forge NAG Status"].startswith("bypassed")


@pytest.mark.parametrize("batch", [1,2,3])
def test_full_lifecycle_cache_and_restore(batch):
    p=Processing(batch=batch)
    original_sample=p.sample
    original_runner=p.scripts
    original_forward=p.base.model.diffusion_model.forward
    weights={k:v.clone() for k,v in p.base.model.diffusion_model.state_dict().items()}
    enable(p)
    result=p.generate_and_save()
    assert result.shape==p.x.shape and torch.isfinite(result).all()
    assert p.saved==1 and p.sd_model.encode_calls==1
    assert p.sd_model.forge_objects.unet is p.base
    assert p.scripts is original_runner
    assert p.base.model.diffusion_model.forward==original_forward
    assert p.base.model_options=={"transformer_options":{}}
    assert p.base.extra_preserved_memory_during_sampling==0
    assert p.negative_prompt=="standard negative remains untouched"
    assert p.extra_generation_params["Forge NAG Active Calls"]==3
    assert p.extra_generation_params["Forge NAG Text Fusion Calls"]==1
    assert p.base.model.predictor.calls==["input","time","output"]*3
    assert p.observed_wrappers[0].adapter.negative_context is None
    assert p.observed_wrappers[0].adapter._negative_text is None
    for k,v in p.base.model.diffusion_model.state_dict().items():
        assert torch.equal(v,weights[k])
    disarm_request(p)
    assert p.sample==original_sample


@pytest.mark.parametrize("field,value", [("cfg_scale",float('nan')),("cfg_scale",True),
                                         ("enable_hr",True),("is_hr_pass",True),("txt2img_upscale",True),
                                         ("refiner_checkpoint","other"),("sampler_name","Euler CFG++"),("tiling",True)])
def test_invalid_request_cannot_save(field,value):
    p=Processing()
    setattr(p,field,value)
    enable(p)
    with pytest.raises(NAGError):p.generate_and_save()
    assert p.saved==0 and p.sd_model.encode_calls==0


def test_wrong_model_error_and_other_models_untouched_when_off():
    p=Processing()
    binding=replace(make_bindings(),engine_type=type("OtherEngine",(),{}))
    enable(p,bindings=binding)
    with pytest.raises(NAGError,match="Krea2"):p.generate_and_save()
    assert p.saved==0
    enable(p,(False,),binding)
    p.generate_and_save()
    assert p.saved==1


@pytest.mark.parametrize("conflict", ["model_function_wrapper","sampler_cfg_function","sampler_pre_cfg_function",
                                      "sampler_post_cfg_function","conditioning_modifiers","disable_cfg1_optimization"])
def test_conflicting_sampler_options_stop(conflict):
    p=Processing()
    p.base.model_options[conflict]=True
    enable(p)
    with pytest.raises(NAGError):p.generate_and_save()
    assert p.saved==0 and p.base.model_options[conflict] is True


@pytest.mark.parametrize("conflict", ["patches","patches_replace","block_modifiers","block_inner_modifiers","attention_override"])
def test_conflicting_transformer_options_stop(conflict):
    p=Processing()
    p.base.model_options["transformer_options"][conflict]={"other":1}
    enable(p)
    with pytest.raises(NAGError):p.generate_and_save()
    assert p.saved==0


def test_late_hook_rejection_escapes_host_swallow_loop():
    p=Processing()
    runner=p.scripts
    def conflicting(p,**kwargs):p.sd_model.forge_objects.unet.model_options["sampler_cfg_function"]=True
    runner.hooks.append(conflicting)
    enable(p)
    with pytest.raises(NAGError,match="wrapper"):p.generate_and_save()
    assert runner.errors==[]
    assert p.saved==0 and p.scripts is runner


def test_non_nag_script_failure_keeps_native_runner_behavior():
    p=Processing()
    runner=p.scripts
    def bad_script(p,**kwargs):raise ValueError("unrelated script error")
    runner.hooks.append(bad_script)
    enable(p)
    p.generate_and_save()
    assert len(runner.errors)==1 and p.saved==1


@pytest.mark.parametrize("kind", ["exception","keyboard_interrupt"])
def test_restore_after_failure_or_interrupt(kind):
    p=Processing()
    runner=p.scripts
    exc=ValueError if kind=="exception" else KeyboardInterrupt
    def fail():raise exc("cancelled")
    p.after_hook=fail
    enable(p)
    with pytest.raises(exc):p.generate_and_save()
    assert p.saved==0 and p.scripts is runner and p.sd_model.forge_objects.unet is p.base
    assert "model_function_wrapper" not in p.base.model_options
    p.after_hook=None
    enable(p,(False,))
    p.generate_and_save()
    assert p.saved==1


def test_no_model_calls_cannot_be_misreported_as_nag():
    p=Processing()
    p.skip_model=True
    enable(p)
    with pytest.raises(NAGError,match="not executed"):p.generate_and_save()
    assert p.saved==0 and p.sd_model.forge_objects.unet is p.base


def test_sigma_interval_native_bypass_and_zero_active_error():
    p=Processing()
    enable(p,(True,"wings",4,2.5,.25,.6,.2))
    p.generate_and_save()
    assert p.extra_generation_params["Forge NAG Active Calls"]==1
    p2=Processing()
    enable(p2,(True,"wings",4,2.5,.25,.04,.02))
    with pytest.raises(NAGError,match="No sampled sigma"):p2.generate_and_save()
    assert p2.saved==0 and p2.sd_model.forge_objects.unet is p2.base


def test_reference_detection_at_runtime():
    p=Processing()
    bindings=make_bindings()
    def add_reference():bindings.dynamic_args.ref_latents=[torch.zeros(1)]
    p.after_hook=add_reference
    enable(p,bindings=bindings)
    with pytest.raises(NAGError,match="Reference"):p.generate_and_save()
    assert p.saved==0 and p.sd_model.forge_objects.unet is p.base


def test_encoder_failure_is_not_swallowed():
    p=Processing()
    runner=p.scripts
    def fail(prompt):raise RuntimeError("encoder unavailable")
    p.sd_model.get_learned_conditioning=fail
    enable(p)
    with pytest.raises(RuntimeError,match="encoder unavailable"):p.generate_and_save()
    assert p.saved==0 and p.scripts is runner


def test_negative_context_flag_and_token_limit():
    p=Processing()
    p.sd_model.text_processing_engine_qwen.tokenize=lambda text:list(range(2049))
    enable(p)
    with pytest.raises(NAGError,match="2048"):p.generate_and_save()
    assert p.sd_model.encode_calls==0


def test_reused_request_no_duplicate_wrapper_or_cache():
    p=Processing()
    enable(p)
    first=p.generate_and_save()
    first_wrapper=p.observed_wrappers[-1]
    enable(p)
    second=p.generate_and_save()
    second_wrapper=p.observed_wrappers[-1]
    assert first_wrapper is not second_wrapper
    assert torch.equal(first,second)
    assert p.sd_model.encode_calls==2
    enable(p,(False,))
    assert not any(key.startswith("Forge NAG") for key in p.extra_generation_params)
    p.generate_and_save()
    assert p.sd_model.encode_calls==2


def test_weight_change_between_runs_does_not_reuse_text_fusion_cache():
    p=Processing()
    enable(p)
    first=p.generate_and_save()
    p.base.model.diffusion_model.txtmlp[0].weight.add_(.1)
    second=p.generate_and_save()
    assert not torch.equal(first,second)
    assert p.sd_model.encode_calls==2


def test_reserve_counts_batch_and_resolution():
    model=TinyDiT()
    neg=torch.zeros(1,7,3,16)
    one=memory_reserve(model,torch.zeros(1,4,1,4,6),neg,torch.float32)
    two=memory_reserve(model,torch.zeros(2,4,1,4,6),neg,torch.float32)
    large=memory_reserve(model,torch.zeros(1,4,1,8,12),neg,torch.float32)
    assert 0<one<two and one<large


def test_nonfinite_model_output_stops_and_restores():
    p=Processing()
    p.base.model.diffusion_model.last.proj.weight.fill_(float('nan'))
    runner=p.scripts
    enable(p)
    with pytest.raises(NAGError,match="not finite"):p.generate_and_save()
    assert p.saved==0 and p.scripts is runner and p.sd_model.forge_objects.unet is p.base


def test_shallow_copied_processing_rebinds_sample_to_new_owner():
    import copy
    p=Processing()
    enable(p)
    q=copy.copy(p)
    q.extra_generation_params={}
    q.x=p.x+3
    q.observed_wrappers=[]
    q.saved=0
    enable(q)
    q.generate_and_save()
    assert q.saved==1 and p.saved==0
    assert q.extra_generation_params['Forge NAG Active Calls']==3
    assert p.extra_generation_params=={}
    assert q.observed_wrappers and not p.observed_wrappers


def test_foreign_callable_is_never_unwrapped_as_our_guard():
    p=Processing()
    calls=[]
    original=p.sample
    def foreign(*args,**kwargs):
        calls.append('foreign')
        return original(*args,**kwargs)
    # A foreign callable could carry a similarly named attribute; ownership
    # must come from the private guard's type, not arbitrary copied attributes.
    foreign._forge_nag_original=original
    p.sample=foreign
    enable(p,(False,))
    assert p.sample is foreign
    p.generate_and_save()
    assert calls==['foreign']
