"""Compact UI, saved defaults and per-request Sigma compatibility."""
import json
from types import SimpleNamespace

import pytest

from forge_neo_nag.config import NAGConfig, NAGError
from forge_neo_nag.settings import help_html, legacy_sigma_defaults, resolve_sigma
from forge_neo_nag.ui import DEFAULTS, FIELDS, META_KEYS, STRINGS, build_ui


@pytest.mark.parametrize("locale", ["None", "ja_JP"])
def test_compact_panel_keeps_input_and_runtime_contracts(locale):
    import gradio as gr
    with gr.Blocks() as demo:
        controls, fields = build_ui(gr, locale, settings_defaults=True)
    config = demo.get_config_file()
    components = config['components']
    ids = [c['props'].get('elem_id') for c in components]
    assert ids.index('forge_neo_nag_negative') < ids.index('forge_neo_nag_enabled') < ids.index('forge_neo_nag_adapter') < ids.index('forge_neo_nag_phi')
    assert controls[1].lines == 2
    assert not any(c['type'] in ('markdown', 'button') for c in components)
    assert sum(c['type'] == 'accordion' for c in components) == 1
    assert all(getattr(c, 'info', None) in (None, '') for c in controls)
    assert [c.elem_id for c in controls] == [f'forge_neo_nag_{field}' for field in FIELDS]
    assert tuple(c.value for c in controls) == DEFAULTS
    assert all(c.visible is False and c.do_not_save_to_config for c in controls[5:7])
    # Missing Sigma metadata follows the user's current Settings, not stale UI defaults.
    assert fields[5][1]({}) is None and fields[6][1]({}) is None
    explicit = {META_KEYS[5]: 650., META_KEYS[6]: 30.}
    assert fields[5][1](explicit) == 650. and fields[6][1](explicit) == 30.


def test_settings_defaults_are_live_and_explicit_values_win():
    opts = SimpleNamespace(forge_neo_nag_sigma_start=700., forge_neo_nag_sigma_end=40.)
    assert resolve_sigma(opts, None, None) == (700., 40.)
    opts.forge_neo_nag_sigma_start = 500.
    assert resolve_sigma(opts, None, None) == (500., 40.)
    assert resolve_sigma(opts, 900., 10.) == (900., 10.)
    assert resolve_sigma(opts, 900., None) == (900., 40.)
    assert resolve_sigma(SimpleNamespace(), None, None) == DEFAULTS[5:7]


@pytest.mark.parametrize("locale,lang", [("None", "en"), ("ja_JP", "ja")])
def test_legacy_defaults_migrate_without_writing_file(tmp_path, locale, lang):
    filename = tmp_path / 'old-ui-config.json'
    data = {f'customscript/forge_neo_nag.py/txt2img/{STRINGS[field][lang][0]}/value': value
            for field, value in zip(('sigma_start', 'sigma_end'), (750., 25.))}
    filename.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    before = filename.read_bytes()
    assert legacy_sigma_defaults(filename, locale) == (750., 25.)
    assert legacy_sigma_defaults(filename, 'ja_JP' if lang == 'en' else 'None') == (750., 25.)
    assert filename.read_bytes() == before


@pytest.mark.parametrize("content", ['not JSON', '[]', '{}', '{"customscript/forge_neo_nag.py/txt2img/Sigma Start/value": -1}'])
def test_legacy_invalid_or_empty_config_uses_original_defaults(tmp_path, content):
    filename = tmp_path / 'ui-config.json'
    filename.write_text(content, encoding='utf-8')
    assert legacy_sigma_defaults(filename) == DEFAULTS[5:7]


@pytest.mark.parametrize("locale,lang", [("None", "en"), ("ja_JP", "ja")])
def test_help_contains_all_existing_descriptions(locale, lang):
    import html
    content = help_html(locale)
    for field in FIELDS:
        assert html.escape(STRINGS[field][lang][1]) in content
    assert 'Reference' in content and 'Hires fix' in content
    assert 'Apply settings' in content


def test_invalid_effective_sigma_still_rejects_request():
    opts = SimpleNamespace(forge_neo_nag_sigma_start=10., forge_neo_nag_sigma_end=20.)
    with pytest.raises(NAGError):
        NAGConfig.parse(True, 'wings', sigma_start=resolve_sigma(opts, None, None)[0],
                        sigma_end=resolve_sigma(opts, None, None)[1])
    # Invalid global defaults cannot override valid explicit API/PNG values.
    pair = resolve_sigma(opts, 1000., 0.)
    assert NAGConfig.parse(True, 'wings', sigma_start=pair[0], sigma_end=pair[1]).active
