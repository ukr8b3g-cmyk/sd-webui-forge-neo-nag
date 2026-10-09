"""Forge Settings integration; no model imports or generation-side effects."""
from __future__ import annotations

import html
import json
from pathlib import Path

from .config import NAGError, _number
from .ui import DEFAULTS, FIELDS, PREFIX, STRINGS, language, sdxl_test_values

SIGMA_KEYS = (f"{PREFIX}_sigma_start", f"{PREFIX}_sigma_end")
SECTION = (PREFIX, "Forge Neo NAG")


def resolve_sigma(opts, start, end):
    """None follows the saved defaults; explicit PNG/API values win."""
    return (getattr(opts, SIGMA_KEYS[0], DEFAULTS[5]) if start is None else start,
            getattr(opts, SIGMA_KEYS[1], DEFAULTS[6]) if end is None else end)


def legacy_sigma_defaults(filename, localization="None"):
    """Read old UI defaults without editing the user's UI config."""
    if not filename:
        return DEFAULTS[5:7]
    try:
        data = json.loads(Path(filename).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return DEFAULTS[5:7]
        lang = language(localization)
        other = "en" if lang == "ja" else "ja"
        values = []
        for index, field in zip((5, 6), ("sigma_start", "sigma_end")):
            value = DEFAULTS[index]
            for locale in (lang, other):
                key = f"customscript/forge_neo_nag.py/txt2img/{STRINGS[field][locale][0]}/value"
                if key in data:
                    value = data[key]
                    break
            values.append(value)
        start = _number("Sigma Start", values[0], 0, 1000)
        end = _number("Sigma End", values[1], 0, 1000)
        return (start, end) if start >= end else DEFAULTS[5:7]
    except (OSError, ValueError, TypeError, NAGError):
        return DEFAULTS[5:7]


def help_html(localization="None"):
    lang = language(localization)
    if lang == "ja":
        heading = "使い方・対応範囲"
        summary = ("Krea2 / Anima / SDXL（Illustrious）/ Klein / Z-Image Turbo / Ernie Image / "
                   "Qwen-Image (2512) のtxt2imgに対応。Reference・Edit・Hires fixは非対応です。"
                   "CFGとPresetはForge標準に任せます。")
        defaults = ("Sigmaは通常生成の既定値です。変更後は上部のApply settingsで保存してください。"
                    "PNG Infoで復元した値やAPIで明示した値は、その生成で優先します。")
        trial = ("SDXL試験設定ボタンは現在のNAG強度・ノルム上限・混合率・Sigmaを変更します。"
                 "Sigmaの既定値も保存する場合はApply settingsを押してください。"
                 "画質の推奨値ではありません。Enable・Adapter・Negative・CFG・Presetは変更しません。")
    else:
        heading = "Usage and supported models"
        summary = ("Krea2 / Anima / SDXL (Illustrious) / Klein / Z-Image Turbo / Ernie Image / "
                   "Qwen-Image (2512) txt2img. Reference, Edit and Hires fix are not supported. "
                   "CFG and presets remain managed by Forge.")
        defaults = ("Sigma values are defaults for normal generation. Save changes with Apply settings above. "
                    "Explicit PNG Info or API values take precedence for that generation.")
        trial = ("The SDXL test button updates the current NAG Scale, Norm Cap, Blend and Sigma values. "
                 "Use Apply settings to also save the Sigma defaults. These are experimental values, "
                 "not a quality recommendation. Enable, Adapter, Negative, CFG and presets are unchanged.")
    fields = "".join(f"<dt><strong>{html.escape(STRINGS[field][lang][0])}</strong></dt>"
                     f"<dd>{html.escape(STRINGS[field][lang][1])}</dd>" for field in FIELDS)
    return (f"<h3>{html.escape(heading)}</h3><p>{html.escape(summary)}</p>"
            f"<p>{html.escape(defaults)}</p><dl>{fields}</dl><p>{html.escape(trial)}</p>")


def register_settings():
    import gradio as gr
    from modules import shared
    from modules.options import OptionHTML, OptionInfo, options_section

    opts = shared.opts
    locale = getattr(opts, "localization", "None")
    lang = language(locale)
    missing = [key not in opts.data for key in SIGMA_KEYS]
    legacy = legacy_sigma_defaults(getattr(shared.cmd_opts, "ui_config_file", None), locale) if any(missing) else DEFAULTS[5:7]

    def trial_button(**kwargs):
        return gr.Button(value="SDXL試験設定を適用" if lang == "ja" else "Apply SDXL test settings",
                         elem_id=f"{PREFIX}_sdxl_test")

    trial = OptionInfo("", component=trial_button)
    trial.do_not_save = True
    options = {
        SIGMA_KEYS[0]: OptionInfo(DEFAULTS[5], STRINGS["sigma_start"][lang][0], component=gr.Number,
                                  component_args={"precision": None, "minimum": 0, "maximum": 1000},
                                  onchange=lambda: _number("Sigma Start", getattr(opts, SIGMA_KEYS[0]), 0, 1000)),
        SIGMA_KEYS[1]: OptionInfo(DEFAULTS[6], STRINGS["sigma_end"][lang][0], component=gr.Number,
                                  component_args={"precision": None, "minimum": 0, "maximum": 1000},
                                  onchange=lambda: _number("Sigma End", getattr(opts, SIGMA_KEYS[1]), 0, 1000)),
        f"{PREFIX}_sdxl_test_action": trial,
        f"{PREFIX}_help": OptionHTML(help_html(locale)),
    }
    for key, info in options_section(SECTION, options).items():
        opts.add_option(key, info)
    for key, migrate, value in zip(SIGMA_KEYS, missing, legacy):
        if migrate:
            opts.data[key] = value


def bind_settings_ui(component, **kwargs):
    # Forge renders all interfaces before the footer, so cross-tab outputs exist.
    if getattr(component, "elem_id", None) != "footer":
        return
    import gradio as gr
    from modules import shared

    context = gr.context.get_blocks_context()
    if context is None or getattr(context, "_forge_nag_settings_bound", False):
        return
    by_id = {getattr(block, "elem_id", None): block for block in context.blocks.values()}
    required = [f"{PREFIX}_{field}" for field in FIELDS]
    required += [f"setting_{key}" for key in SIGMA_KEYS] + ["settings_json", f"{PREFIX}_sdxl_test"]
    if any(elem_id not in by_id for elem_id in required):
        return
    start, end = (by_id[f"{PREFIX}_{field}"] for field in ("sigma_start", "sigma_end"))

    def current_defaults():
        return tuple(resolve_sigma(shared.opts, None, None))

    previous_defaults = gr.State(current_defaults())

    def initialize_defaults():
        return None, None, current_defaults()

    def sync_defaults(previous):
        current = current_defaults()
        if tuple(previous) == current:
            return gr.skip(), gr.skip(), gr.skip()
        return None, None, current

    # Only the normal browser page receives None; API ui() retains numeric defaults.
    context.root_block.load(fn=initialize_defaults, inputs=[], outputs=[start, end, previous_defaults],
                            queue=False, show_progress=False)
    by_id["settings_json"].change(fn=sync_defaults, inputs=[previous_defaults],
                                  outputs=[start, end, previous_defaults], queue=False, show_progress=False)
    # Apply to the current request and fill the Settings editors; saving is explicit.
    outputs = [by_id[f"{PREFIX}_{field}"] for field in ("phi", "tau", "alpha", "sigma_start", "sigma_end")]
    outputs += [by_id[f"setting_{key}"] for key in SIGMA_KEYS]
    by_id[f"{PREFIX}_sdxl_test"].click(fn=lambda: (*sdxl_test_values(), *DEFAULTS[5:7]),
                                     inputs=[], outputs=outputs, queue=False, show_progress=False)
    context._forge_nag_settings_bound = True


def register_callbacks():
    from modules import script_callbacks
    script_callbacks.on_ui_settings(register_settings)
    script_callbacks.on_after_component(bind_settings_ui)
