"""Routing/UI/CFG boundary tests that do not require a Forge checkout."""
import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.registry import choose_adapter, ENGINE_TYPES, ADAPTER_CHOICES
from forge_neo_nag.adapters.base import BranchLayout, validate_literal_text
from forge_neo_nag.host import arm_request, SamplingSession, validate_request
from forge_neo_nag.ui import build_ui, DEFAULTS
from .helpers import Processing, make_bindings


@pytest.mark.parametrize("family", list(ENGINE_TYPES))
@pytest.mark.parametrize("preset", [None,"sd","xl","krea","anima","unrecognized"])
def test_auto_uses_model_and_never_forces_preset(family,preset):
    module,name=ENGINE_TYPES[family]
    engine=type(name,(),{"__module__":module})()
    assert choose_adapter(engine,"auto",preset)==family


@pytest.mark.parametrize("choice",["krea2","anima","sdxl","klein","zimage","ernie"])
def test_manual_choice_available_even_if_auto_cannot_detect(choice):
    engine=object()
    with pytest.raises(NAGError,match="manually"):choose_adapter(engine)
    assert choose_adapter(engine,choice,"unrelated")==choice


def test_refiner_not_misdetected_by_sdxl_flag():
    engine=type("StableDiffusionXLRefiner",(),{"__module__":"backend.diffusion_engine.sdxl","is_sdxl":True})()
    with pytest.raises(NAGError):choose_adapter(engine)


@pytest.mark.parametrize("cfg", [0.,.5,1.,2.,7.5,30.])
def test_cfg_left_to_user_and_all_other_generation_settings_untouched(cfg):
    p=Processing();p.cfg_scale=cfg
    state=(p.cfg_scale,p.sampler_name,p.width,p.height,p.negative_prompt)
    b=make_bindings()
    arm_request(p,(True,"glasses"),lambda:b,preset="sd")
    result=p.generate_and_save()
    assert torch.isfinite(result).all()
    assert state==(p.cfg_scale,p.sampler_name,p.width,p.height,p.negative_prompt)
    assert p.extra_generation_params["Forge NAG CFG"]==cfg
    assert p.extra_generation_params["Forge NAG UI Preset"]=="sd"


@pytest.mark.parametrize("labels",[[0,1],[1,0]])
@pytest.mark.parametrize("batch",[1,2])
def test_krea_mixed_native_cfg_preserves_unconditional(labels,batch):
    p=Processing(batch=batch);b=make_bindings()
    cfg=NAGConfig.parse(True,"wings")
    session=SamplingSession(p,cfg,b,torch.randn(1,5,3,16))
    x=torch.cat([p.x]*2)
    context=torch.randn(batch*2,8,3,16)
    sigma=torch.ones(batch*2)*.5
    original=x.clone(),context.clone()
    c={"c_crossattn":context,"transformer_options":{"cond_or_uncond":labels}}
    uncond_i=labels.index(1); part=slice(uncond_i*batch,(uncond_i+1)*batch)
    baseline=p.base.model.apply_model(x[part],sigma[part],c_crossattn=context[part].clone(),transformer_options={"cond_or_uncond":[1]})
    session.install(p,x=p.x)
    try:
        out=session.wrapper(p.base.model.apply_model,{"input":x,"timestep":sigma,"c":c,"cond_or_uncond":labels})
        torch.testing.assert_close(out[part],baseline,atol=0,rtol=0)
        assert torch.equal(x,original[0]) and torch.equal(context,original[1])
        assert session.wrapper.active_calls==1
    finally:session.close()


@pytest.mark.parametrize("labels,batch", [([1,0],4),([0,1],4),([0],2),([1],2)])
def test_branch_row_indices(labels,batch):
    layout=BranchLayout.from_payload({"cond_or_uncond":labels},batch)
    expected=tuple(i for i in range(batch) if labels[i//(batch//len(labels))]==0)
    assert layout.positive_rows==expected


@pytest.mark.parametrize("labels,batch",[([],2),([0,1],3),([False],1),([2],1),([0,0],2),([1,1],2)])
def test_unknown_or_composed_branches_rejected(labels,batch):
    with pytest.raises(NAGError):BranchLayout.from_payload({"cond_or_uncond":labels},batch)


def test_eighth_api_argument_is_optional_and_off_ignores_stale_selection():
    seven=NAGConfig.parse(True,"glasses",2,2.5,.25,1000,0)
    assert seven.adapter=="auto"
    assert NAGConfig.parse(True,"glasses",2,2.5,.25,1000,0,"SDXL").adapter=="sdxl"
    assert not NAGConfig.parse(False,None,"bad",0,None,None,None,"missing").active
    assert NAGConfig.parse(True,"",adapter="anima").metadata()["Forge NAG Adapter Selection"]=="anima"
    with pytest.raises(NAGError):NAGConfig.parse(True,"glasses",adapter="missing")


@pytest.mark.parametrize("text",[r"character \(series\)","glasses, text",r"name \[tag\]"])
def test_literal_escaped_tag_punctuation(text):
    validate_literal_text(text)


@pytest.mark.parametrize("text",["(glasses)","[glasses]","name (series)"])
def test_implicit_emphasis_rejected(text):
    with pytest.raises(NAGError):validate_literal_text(text)


def test_ui_manual_selector_button_does_not_write_forge_controls():
    import gradio as gr
    with gr.Blocks() as demo:controls,fields=build_ui(gr,"ja_JP")
    assert len(controls)==8
    assert controls[-1].value=="auto" and controls[-1].interactive is not False
    assert [v for _,v in controls[-1].choices]==["auto","krea2","anima","sdxl","klein","zimage","ernie"]
    for selection in ("auto","sdxl","anima","krea2","klein","zimage","ernie"):
        params=NAGConfig.parse(True,"glasses",adapter=selection).metadata()
        assert fields[-1][1](params)==selection
    assert fields[-1][1]({})=="auto"
    config=demo.get_config_file()
    ids={c["id"]:c["props"].get("elem_id") for c in config["components"]}
    dependencies=[d for d in config["dependencies"] if d.get("outputs")]
    # Only the explicit SDXL sample-settings button has a write callback.
    assert len(dependencies)==1
    assert [ids[i] for i in dependencies[0]["outputs"]]==[
        "forge_neo_nag_phi","forge_neo_nag_tau","forge_neo_nag_alpha",
        "forge_neo_nag_sigma_start","forge_neo_nag_sigma_end"]


@pytest.mark.parametrize("locale", ["None", "ja_JP"])
@pytest.mark.parametrize("label,expected", [choice for choice in ADAPTER_CHOICES if choice[1] != "auto"])
def test_manual_adapter_display_name_restores_to_internal_id(locale,label,expected):
    import gradio as gr
    with gr.Blocks():
        controls,fields=build_ui(gr,locale)
    params={"Forge NAG Adapter Selection":label}
    assert fields[-1][1](params)==expected


@pytest.mark.parametrize("label,expected", [choice for choice in ADAPTER_CHOICES if choice[1] != "auto"])
def test_manual_adapter_internal_id_roundtrip_stays_stable(label,expected):
    params=NAGConfig.parse(True,"glasses",adapter=expected).metadata()
    assert params["Forge NAG Adapter Selection"]==expected
