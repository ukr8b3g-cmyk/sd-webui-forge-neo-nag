"""Isolated real-Gradio browser probe; NOT a full Forge Neo UI/GPU test.

Uses already installed Gradio, Playwright and Chromium. No package installation
or public sharing. Run from the extension directory with PYTHONPATH=.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(output: Path):
    import gradio as gr
    from playwright.sync_api import sync_playwright
    from forge_neo_nag.ui import build_ui

    output.mkdir(parents=True, exist_ok=True)
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown("NAG adapter component test — not a running Forge instance")
        preset=gr.Dropdown(choices=['sd','xl','krea','anima'], value='sd', label='Host preset (test)', elem_id='test_preset')
        cfg=gr.Number(value=6., label='Host CFG (test)', elem_id='test_cfg')
        steps=gr.Number(value=20, label='Host steps (test)', elem_id='test_steps')
        controls,_=build_ui(gr,'ja_JP')
        # Simulate a host-owned preset callback changing its own settings.
        preset.change(lambda name:(3.,42) if name=='anima' else (6.,20),[preset],[cfg,steps],queue=False)
    demo.launch(server_name='127.0.0.1',server_port=7874,share=False,prevent_thread_lock=True,quiet=True)
    results=[]
    try:
        with sync_playwright() as pw:
            executable=shutil.which('chromium')
            browser=pw.chromium.launch(headless=True,executable_path=executable,args=['--no-sandbox'])
            page=browser.new_page(viewport={'width':1200,'height':1300})
            page.goto('http://127.0.0.1:7874',wait_until='networkidle')
            page.get_by_text('Forge Neo NAG',exact=True).click()
            adapter=page.locator('#forge_neo_nag_adapter')
            def select(locator,label):
                locator.locator('input').click()
                page.get_by_role('option',name=label,exact=True).click()
            def number(selector):
                return float(page.locator(selector+' input[type=number]').input_value())
            def adapter_label():
                # Gradio 6 renders selected labels in an input or token span.
                return adapter.inner_text()+' '+adapter.locator('input').input_value()
            select(adapter,'SDXL / Illustrious')
            assert 'SDXL / Illustrious' in adapter_label()
            assert not page.locator('#forge_neo_nag_enabled input').is_checked()
            results.append({'case':'manual_selection_while_off','pass':True})
            page.locator('#forge_neo_nag_negative textarea').fill('glasses, text')
            page.locator('#forge_neo_nag_enabled input').check()
            page.locator('#test_cfg input').fill('7.5')
            page.locator('#test_cfg input').press('Tab')
            assert number('#test_cfg')==7.5
            select(page.locator('#test_preset'),'anima')
            page.wait_for_function("document.querySelector('#test_steps input').value === '42'")
            assert number('#test_cfg')==3.
            assert 'SDXL / Illustrious' in adapter_label()
            assert page.locator('#forge_neo_nag_enabled input').is_checked()
            assert number('#forge_neo_nag_phi')==4.
            results.append({'case':'host_preset_changes_only_host_settings_manual_adapter_retained','pass':True})
            page.locator('#forge_neo_nag_sdxl_test').click()
            page.wait_for_function("document.querySelector('#forge_neo_nag_phi input[type=number]').value === '2'")
            assert number('#forge_neo_nag_tau')==2.5 and number('#forge_neo_nag_alpha')==.25
            assert number('#test_cfg')==3. and number('#test_steps')==42
            assert page.locator('#forge_neo_nag_negative textarea').input_value()=='glasses, text'
            assert 'SDXL / Illustrious' in adapter_label()
            assert page.locator('#forge_neo_nag_enabled input').is_checked()
            results.append({'case':'sdxl_button_changes_only_five_nag_numbers','pass':True})
            select(adapter,'Anima')
            page.locator('#forge_neo_nag_enabled input').uncheck()
            assert 'Anima' in adapter_label() and number('#test_cfg')==3.
            results.append({'case':'adapter_switch_and_off_do_not_change_cfg','pass':True})
            js=(ROOT/'javascript/forge_neo_nag.js').read_text(encoding='utf-8')
            page.evaluate("window.opts={localization:'ja_JP'}")
            page.add_script_tag(content=js)
            page.evaluate('window.ForgeNeoNAGHelp.refresh()')
            assert '手動選択' in adapter.get_attribute('title')
            page.evaluate("window.opts.localization='en'; window.ForgeNeoNAGHelp.refresh()")
            assert 'manually' in adapter.get_attribute('title')
            results.append({'case':'existing_hover_js_handles_new_adapter_in_japanese_and_english','pass':True})
            select(adapter,'Auto')
            assert 'Auto' in adapter_label()
            results.append({'case':'manual_to_auto_switch_available','pass':True})
            page.screenshot(path=str(output/'adapter-component.png'),full_page=True)
            browser.close()
    finally:
        demo.close()
    report={'scope':'Isolated Gradio 6.5.1 components and simulated host settings; NOT Forge runtime or GPU',
            'cases':results,'passed':len(results)}
    (output/'ui-component.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:
        run(args.output)
    except Exception as exc:
        args.output.mkdir(parents=True,exist_ok=True)
        report={"status":"not_completed", "scope":"Isolated Gradio components; not Forge runtime",
                "error":f"{type(exc).__name__}: {exc}"}
        (args.output/'ui-component.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        raise
