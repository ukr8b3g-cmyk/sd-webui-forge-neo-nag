import math
import pytest
import torch

from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.math import guide_attention


def reference(p, n, phi, tau, alpha):
    p, n = p.float(), n.float()
    value = p + phi * (p - n)
    eps = torch.finfo(torch.float32).eps
    ratio = value.abs().sum(-1, keepdim=True).clamp_min(eps) / p.abs().sum(-1, keepdim=True).clamp_min(eps)
    capped = value * (ratio.clamp_max(tau) / ratio)
    return alpha * capped + (1 - alpha) * p


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("phi,tau,alpha", [(4,2.5,.25),(20,.01,1),(1,1,.5),(0,2.5,.25),(4,2.5,0)])
def test_math_matches_reference(dtype, phi, tau, alpha):
    p = torch.randn(2, 9, 64).to(dtype)
    n = torch.randn_like(p)
    oldp, oldn = p.clone(), n.clone()
    result = guide_attention(p, n, phi=phi, tau=tau, alpha=alpha)
    expected = reference(p, n, phi, tau, alpha).to(dtype)
    torch.testing.assert_close(result, expected, rtol=0.004 if dtype != torch.float32 else 1e-6,
                               atol=0.004 if dtype != torch.float32 else 1e-6)
    assert result.dtype == dtype
    assert torch.equal(p, oldp) and torch.equal(n, oldn)


@pytest.mark.parametrize("pval,nval", [(0,0),(0,1),(1,0),(1e-30,-1e-30),(1e4,-1e4)])
def test_zero_and_extreme_norms(pval, nval):
    p = torch.full((2,3,32), float(pval))
    n = torch.full_like(p, float(nval))
    result = guide_attention(p,n,phi=4,tau=2.5,alpha=.25)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, reference(p,n,4,2.5,.25), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("tau", [.01,.5,1,2.5,20])
def test_l1_cap(tau):
    p, n = torch.randn(2,5,32), torch.randn(2,5,32)
    out = guide_attention(p,n,phi=20,tau=tau,alpha=1)
    assert torch.all(out.abs().sum(-1) <= tau*p.abs().sum(-1) + 1e-4)


def test_shape_and_dtype_validation():
    with pytest.raises(ValueError):
        guide_attention(torch.zeros(1,3,4), torch.zeros(1,2,4), phi=4,tau=2.5,alpha=.25)
    with pytest.raises(ValueError):
        guide_attention(torch.zeros(1,3,4), torch.zeros(1,3,4).half(), phi=4,tau=2.5,alpha=.25)


@pytest.mark.parametrize("field,value", [("phi",math.nan),("tau",math.inf),("alpha",-1),("phi",True),
                                         ("tau",0),("phi",21),("sigma_start",-1),("sigma_end",1001),
                                         ("alpha",2),("phi","bad")])
def test_invalid_numeric_config(field,value):
    with pytest.raises(NAGError):
        NAGConfig.parse(True,"wings",**{field:value})


@pytest.mark.parametrize("text", ["<lora:foo:1>","<lyco:foo:1>","[red:blue:0.5]","[red|blue]","(red:1.2)","red AND blue","BREAK"])
def test_reject_unsupported_prompt_syntax(text):
    with pytest.raises(NAGError):
        NAGConfig.parse(True,text)


@pytest.mark.parametrize("text", ["big wings", "青い羽根、文字", 'red, text: "hello"\nsmall objects', "hands (large)"])
def test_plain_text(text):
    assert NAGConfig.parse(True,text).negative == text


@pytest.mark.parametrize("enabled", [1,None,[],"yes"])
def test_boolean_strict(enabled):
    with pytest.raises(NAGError):
        NAGConfig.parse(enabled,"wings")


def test_off_does_not_validate_inactive_controls():
    assert not NAGConfig.parse(False, None, "invalid", 0, math.nan).active


def test_sigma_order():
    with pytest.raises(NAGError):
        NAGConfig.parse(True,"wings",sigma_start=.1,sigma_end=.5)


@pytest.mark.parametrize("args,reason", [((False,),"disabled"),((True," "),"empty negative"),
                                      ((True,"wings",0),"phi=0"),((True,"wings",4,2.5,0),"alpha=0")])
def test_bypass(args,reason):
    cfg = NAGConfig.parse(*args)
    assert not cfg.active and cfg.bypass_reason == reason
