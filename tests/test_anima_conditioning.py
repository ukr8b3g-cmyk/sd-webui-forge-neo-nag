"""Native CPU regressions for Forge's singleton 4D Anima conditioning."""
from types import SimpleNamespace

import pytest
import torch

from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.host import arm_request
from .native_multi_helpers import Processing, definitions, session, source_file


def processing(batch=1, dtype=torch.float32):
    p = Processing('anima', batch, dtype)
    stack = definitions(source_file('modules/prompt_parser.py', 'prompt_parser.py'),
                        {'stack_conds'})['stack_conds']
    # Each native Qwen06Engine output retains [1, tokens, width]. The real
    # Forge stack_conds adds the prompt batch axis without squeezing it.
    p.context = stack([row.unsqueeze(0) for row in p.context])
    p.uncond = stack([row.unsqueeze(0) for row in p.uncond])
    assert p.context.shape == (batch, 1, 9, 24)
    return p


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('labels', [[0], [0, 1], [1, 0]])
def test_native_4d_zero_alpha_matches_forward(dtype, batch, labels):
    p = processing(batch, dtype)
    baseline = p.evaluate(p.payload(p.x, .5, labels))
    cfg = NAGConfig.parse(True, 'glasses', alpha=0., adapter='anima')
    s = session(p, cfg)
    try:
        assert s.wrapper.adapter.negative_context.ndim == 3
        got = p.evaluate(p.payload(p.x, .5, labels))
        assert torch.equal(got, baseline)
        assert s.wrapper.adapter.attention_calls == len(s.wrapper.adapter.targets)
    finally:
        s.close()


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('labels', [[0], [0, 1], [1, 0]])
def test_native_4d_guides_positive_only_without_mutating_input(batch, labels):
    p = processing(batch)
    payload = p.payload(p.x, .5, labels)
    baseline = p.evaluate(payload)
    x0 = payload['input'].clone()
    context0 = payload['c']['c_crossattn'].clone()
    weights = {k: v.clone() for k, v in p.base.model.diffusion_model.state_dict().items()}
    s = session(p)
    try:
        payload = p.payload(p.x, .5, labels)
        got = p.evaluate(payload)
        c = slice(labels.index(0) * batch, (labels.index(0) + 1) * batch)
        assert not torch.equal(got[c], baseline[c])
        if 1 in labels:
            u = slice(labels.index(1) * batch, (labels.index(1) + 1) * batch)
            assert torch.equal(got[u], baseline[u])
        assert torch.isfinite(got).all()
        assert torch.equal(payload['input'], x0)
        assert torch.equal(payload['c']['c_crossattn'], context0)
        assert payload['c']['c_crossattn'].ndim == 4
        assert s.wrapper.adapter.negative_context.shape == (1, 512, 24)
        assert s.wrapper.adapter.attention_calls == len(s.wrapper.adapter.targets)
        for name, value in p.base.model.diffusion_model.state_dict().items():
            assert torch.equal(value, weights[name])
    finally:
        s.close()
    assert p.base.model_options == {'transformer_options': {}}


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('cfg_scale', [1., 7.5])
@pytest.mark.parametrize('separate', [False, True])
def test_native_4d_off_on_off_restores_host(batch, cfg_scale, separate):
    p = processing(batch)
    p.cfg_scale, p.separate = cfg_scale, separate
    baseline = p.sample()
    original_runner = p.scripts
    original_forward = p.base.model.diffusion_model.forward
    original_state = (p.cfg_scale, p.sampler_name, p.negative_prompt)
    arm_request(p, (True, 'glasses', 2., 2.5, .25, 1000., 0., 'anima'), lambda: p.bindings)
    got = p.generate_and_save()
    assert torch.isfinite(got).all() and not torch.equal(got, baseline)
    assert p.saved == 1 and p.sd_model.encode_calls == 1
    assert p.scripts is original_runner and p.sd_model.forge_objects.unet is p.base
    assert p.base.model.diffusion_model.forward == original_forward
    assert original_state == (p.cfg_scale, p.sampler_name, p.negative_prompt)
    assert p.base.model_options == {'transformer_options': {}}
    assert p.extra_generation_params['Forge NAG Adapter'] == 'anima'
    assert p.observed_wrappers[-1].adapter.negative_context is None
    arm_request(p, (False,), lambda: pytest.fail('OFF loaded backend'))
    assert torch.equal(p.sample(), baseline)
    assert p.sd_model.encode_calls == 1 and p.extra_generation_params == {}


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('cfg_scale', [1., 7.5])
def test_native_4d_cfg_function_keeps_native_combination(batch, cfg_scale):
    ns = definitions(source_file('backend/sampling/sampling_function.py', 'sampling_methods.py'),
                     {'sampling_function_inner'})
    ns['dynamic_args'] = SimpleNamespace(context_handler=None)
    p = processing(batch)
    s = session(p)
    seen = []
    def calc(model, cond, uncond, x, timestep, options):
        seen.append(uncond is None)
        if uncond is None:
            return p.evaluate(p.payload(x, .5, [0])), torch.zeros_like(x)
        u, c = p.evaluate(p.payload(x, .5, [1, 0])).chunk(2)
        return c, u
    ns['calc_cond_uncond_batch'] = calc
    try:
        out, c, u = ns['sampling_function_inner'](
            p.base.model, p.x, torch.full((batch,), .5), [{}], [{}], cfg_scale, return_full=True)
        assert torch.equal(out, u + (c - u) * cfg_scale)
        assert seen == [cfg_scale == 1.]
        assert s.wrapper.active_calls == 1
        assert 'sampler_cfg_function' not in p.sd_model.forge_objects.unet.model_options
    finally:
        s.close()


@pytest.mark.parametrize('boundary', ['outer', 'cross'])
@pytest.mark.parametrize('kind', ['rank2', 'rank5', 'nonsingleton', 'batch', 'width', 'missing'])
def test_invalid_context_is_rejected_at_both_boundaries(boundary, kind):
    p = processing()
    s = session(p)
    try:
        context = {
            'rank2': lambda: torch.randn(9, 24),
            'rank5': lambda: torch.randn(1, 1, 1, 9, 24),
            'nonsingleton': lambda: torch.randn(1, 2, 9, 24),
            'batch': lambda: torch.randn(2, 1, 9, 24),
            'width': lambda: torch.randn(1, 1, 9, 23),
            'missing': lambda: None,
        }[kind]()
        adapter = s.wrapper.adapter
        if boundary == 'outer':
            with pytest.raises(NAGError, match='Unexpected Anima conditioning shape'):
                adapter(p.x, torch.ones(1), context)
        else:
            cross = adapter.view.blocks[0].cross_attn
            with pytest.raises(NAGError, match='Unexpected Anima positive context batch'):
                cross(torch.randn(1, 9, 48), context)
    finally:
        s.close()


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('ndim', [3, 4])
def test_context_is_forwarded_to_native_view_without_reshape(batch, ndim):
    p = processing(batch)
    s = session(p)
    try:
        context = p.context if ndim == 4 else p.context.squeeze(1)
        adapter = s.wrapper.adapter
        seen = []
        def native_view(x, timesteps, forwarded, **kwargs):
            seen.append(forwarded)
            return x
        adapter.view = native_view
        assert adapter(p.x, torch.ones(batch), context) is p.x
        assert seen[0] is context
    finally:
        s.close()
