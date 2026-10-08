import html
import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

from forge_neo_nag.config import NAGConfig
from forge_neo_nag.ui import build_ui, tooltip_metadata, language, STRINGS, FIELDS, DEFAULTS
from forge_neo_nag.registry import ADAPTER_CHOICES


@pytest.mark.parametrize("name,expected", [("None","en"),("en_US","en"),("ja_JP","ja"),("ja_JP.json","ja"),
                                          ("Japanese","ja"),("日本語","ja"),("zh_CN","en")])
def test_language(name,expected):
    assert language(name)==expected


@pytest.mark.parametrize("locale", ["None","ja_JP"])
def test_real_gradio_ui_schema_and_infotext(locale):
    import gradio as gr
    with gr.Blocks() as demo:
        controls, fields=build_ui(gr,locale)
    assert len(controls)==8 and len(fields)==8
    assert tuple(component.value for component in controls)==DEFAULTS
    config=demo.get_config_file()
    ids=[component["props"].get("elem_id") for component in config["components"]]
    for name in FIELDS:
        assert f"forge_neo_nag_{name}" in ids
    params=NAGConfig.parse(True,'red, blue: "text"\n日本語').metadata()
    assert fields[0][1](params) is True
    assert fields[1][1](params)=='red, blue: "text"\n日本語'
    assert fields[0][1]({}) is False
    assert [field[1]({}) for field in fields]==list(DEFAULTS)
    assert controls[1].elem_id not in ("txt2img_neg_prompt","img2img_neg_prompt")


def test_tooltip_metadata_is_html_safe_and_complete():
    data=tooltip_metadata("ja")
    assert '<script' not in data
    match=re.search(r'data-forge-nag-help="([^"]*)"',data)
    decoded=json.loads(html.unescape(match.group(1)))
    assert len(decoded["fields"])==8
    for field in FIELDS:
        assert decoded["fields"][f"forge_neo_nag_{field}"]=={k:STRINGS[field][k][1] for k in ("ja","en")}


def test_png_metadata_roundtrip(tmp_path):
    # Match the reviewed host quote/unquote convention, including Unicode,
    # quotes, commas and newlines; do not flatten or silently rewrite prompts.
    params=NAGConfig.parse(True,'wide wings, color: "blue"\n日本語の文字').metadata()
    quote=lambda v:json.dumps(v,ensure_ascii=False) if any(c in str(v) for c in (',','\n',':')) else str(v)
    text='a bird\nSteps: 8, CFG scale: 1.0, '+', '.join(f'{k}: {quote(v)}' for k,v in params.items())
    metadata=PngImagePlugin.PngInfo()
    metadata.add_text("parameters",text)
    path=tmp_path/'metadata.png'
    Image.new('RGB',(2,2)).save(path,pnginfo=metadata)
    loaded=Image.open(path).info['parameters']
    assert loaded==text
    pattern=r'\s*([\w\s\-\/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
    decoded={}
    for key,value in re.findall(pattern,loaded.split('\n')[-1]):
        decoded[key.strip()]=json.loads(value) if value.startswith('"') else value.strip()
    assert decoded['Forge NAG Negative']==params['Forge NAG Negative']
    assert decoded['Forge NAG']=='True'


def test_extension_loads_without_importing_backend(monkeypatch):
    scripts_mod=SimpleNamespace(Script=type('ScriptBase',(),{}),AlwaysVisible=object())
    modules=ModuleType('modules')
    modules.scripts=scripts_mod
    modules.shared=SimpleNamespace(opts=SimpleNamespace(localization='None'))
    monkeypatch.setitem(sys.modules,'modules',modules)
    path=Path(__file__).resolve().parents[1]/'scripts'/'forge_neo_nag.py'
    spec=importlib.util.spec_from_file_location('_nag_ui_entry_test',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    extension=module.Script()
    assert extension.title()=='Forge Neo NAG'
    assert extension.show(False) is scripts_mod.AlwaysVisible
    assert extension.show(True) is False


@pytest.mark.parametrize("locale", ["None","ja_JP"])
def test_adapter_infotext_restore_accepts_ids_and_display_names(locale):
    import gradio as gr
    with gr.Blocks():
        controls,fields=build_ui(gr,locale)
    restore=fields[-1][1]
    for label,internal in ADAPTER_CHOICES:
        assert restore({"Forge NAG Adapter Selection":internal})==internal
        assert restore({"Forge NAG Adapter Selection":label})==internal
