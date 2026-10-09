"""Bilingual UI strings and scoped tooltip metadata; no model imports."""
from __future__ import annotations

import html
import json
import re

PREFIX = "forge_neo_nag"
FIELDS = ("enabled", "negative", "phi", "tau", "alpha", "sigma_start", "sigma_end", "adapter")
DEFAULTS = (False, "", 4.0, 2.5, 0.25, 1000.0, 0.0, "Auto")
META_KEYS = ("Forge NAG", "Forge NAG Negative", "Forge NAG Phi", "Forge NAG Tau", "Forge NAG Alpha", "Forge NAG Sigma Start", "Forge NAG Sigma End", "Forge NAG Adapter Selection")

STRINGS = {
    "adapter": {
        "en": ("Adapter", "Auto uses the loaded model. Select Krea2, Anima, SDXL, Klein, Z-Image, Ernie or Qwen-Image manually to override Auto. UI Preset never locks this selection. A structurally incompatible model reports an error; no silent fallback or model switch."),
        "ja": ("アダプター", "Autoは読み込まれたモデルから判定します。Krea2・Anima・SDXL・Klein・Z-Image・Ernie・Qwen-Imageを手動選択して変更できます。UI Presetでは固定しません。構造が合わない場合は理由を表示し、黙って切り替えたりモデルを変更したりしません。"),
    },
    "enabled": {
        "en": ("Enable NAG", "Krea2, Anima, SDXL, Klein, Z-Image, Ernie and Qwen-Image txt2img. NAG ON/OFF and CFG are your choice. Normal CFG and its negative branch are preserved; no preset or generation setting is changed. OFF, empty text, Phi=0 or Alpha=0 uses the normal path."),
        "ja": ("NAGを有効化", "Krea2・Anima・SDXL・Klein・Z-Image・Ernie・Qwen-Imageのtxt2imgに対応します。CFGが1以外でもON/OFFを選べます。標準CFGとNegativeの経路を維持し、Presetや生成設定は変更しません。OFF・空欄・Phi=0・Alpha=0では通常経路です。"),
    },
    "negative": {
        "en": ("NAG Negative Prompt", "Describe what to suppress, for example: big wings. Separate from the standard negative box, which remains controlled by Forge. Plain text only; no weights, schedules or LoRA tags. SDXL: up to four CLIP chunks (308 positions). Anima: both tokenizers up to 2048. Krea2/Klein/Z-Image: 2048 including the Qwen template. Ernie: up to 2048 Ministral tokens. Qwen-Image: up to 2048 Qwen2.5-VL tokens."),
        "ja": ("NAG Negative Prompt（抑制したい内容）", "例：big wings のように抑制したいものを書きます。標準Negative欄とは独立です。通常の文章のみ。重み・スケジュール・LoRAタグは非対応です。SDXLは最大4チャンク（308位置）、Animaは両Tokenizerとも2048、Krea2・Klein・Z-ImageはQwenテンプレート込み2048、Ernieは最大2048 Ministral、Qwen-Imageは最大2048 Qwen2.5-VLトークンです。"),
    },
    "phi": {
        "en": ("NAG Scale / Phi", "Attention extrapolation strength. Existing default: 4.0. The SDXL test button sets 2.0, not a validated quality recommendation. A higher value can suppress more strongly but can change composition or degrade quality. Zero bypasses NAG."),
        "ja": ("NAG強度 / Phi", "Attentionの差を強める係数です。既存初期値4.0。SDXL試験設定ボタンでは2.0になりますが、画質の推奨値ではありません。上げすぎると構図や画質が変化する場合があります。0でNAGをバイパスします。"),
    },
    "tau": {
        "en": ("Norm Cap / Tau", "Caps the extrapolated attention's L1 norm relative to the positive attention before blending. Default 2.5; this is not a CFG scale."),
        "ja": ("ノルム上限 / Tau", "ブレンド前のAttentionのL1ノルムをPositive側に対する比率で制限します。初期値2.5。CFGとは別の値です。"),
    },
    "alpha": {
        "en": ("Blend / Alpha", "Blends normalized guided attention with the positive attention. Default 0.25; range 0–1. Zero is an exact normal-path bypass."),
        "ja": ("混合率 / Alpha", "正規化したNAGのAttentionをPositive側へ混ぜる割合です。初期値0.25、範囲0〜1。0では通常経路へ戻ります。"),
    },
    "sigma_start": {
        "en": ("Sigma Start", "Inclusive upper sampling-sigma boundary, before the host converts sigma to model time. Default 1000. Start must be at least End; this is not a step count or percentage."),
        "ja": ("開始Sigma", "モデル時刻への変換前の、サンプリングSigmaの上限です。境界を含みます。初期値1000。終了Sigma以上にします。ステップ数や割合ではありません。"),
    },
    "sigma_end": {
        "en": ("Sigma End", "Inclusive lower sampling-sigma boundary. Default 0. Outside the interval the original model runs. A run with no active NAG calls is rejected, not saved as an NAG result."),
        "ja": ("終了Sigma", "サンプリングSigmaの下限です。境界を含みます。初期値0。範囲外は元のモデル経路を使います。一度もNAGが適用されなかった生成は成功扱いで保存しません。"),
    },
}


def language(localization) -> str:
    return "ja" if re.search(r"^(ja|jp)([-_.]|$)|japanese|日本語", str(localization).lower()) else "en"


def tooltip_metadata(lang: str) -> str:
    data = {"language": lang, "fields": {
        f"{PREFIX}_{field}": {locale: STRINGS[field][locale][1] for locale in ("en", "ja")}
        for field in FIELDS
    }}
    return '<div data-forge-nag-help="' + html.escape(json.dumps(data, ensure_ascii=False), quote=True) + '" hidden></div>'


def _paste_value(index: int, params, *, settings_defaults=False):
    if settings_defaults and index in (5, 6) and META_KEYS[index] not in params:
        return None  # Follow Settings when this image has no explicit Sigma.
    value = params.get(META_KEYS[index], DEFAULTS[index])
    if index == 0:
        # Loading a non-NAG image must not leave a previously enabled NAG on.
        return value is True or str(value).lower() == "true"
    if index == 7:
        from .registry import adapter_display_name
        from .config import NAGError
        try:
            # PNG metadata continues to store IDs; the visible Dropdown needs
            # the corresponding label to work with Forge UI defaults.
            return adapter_display_name(value)
        except NAGError:
            return DEFAULTS[7]
    return value


def sdxl_test_values():
    # Does not alter Enabled, Adapter, Negative, CFG or any Forge preset.
    return 2.0, 2.5, 0.25, 1000.0, 0.0


def build_ui(gr, localization="None", *, settings_defaults=False):
    lang = language(localization)
    text = lambda field: STRINGS[field][lang]
    with gr.Accordion("Forge Neo NAG", open=False, elem_id=f"{PREFIX}_txt2img"):
        negative = gr.Textbox(value="", label=text("negative")[0],
                              placeholder="big wings", lines=2, max_lines=6, elem_id=f"{PREFIX}_negative")
        with gr.Row(variant="compact"):
            enabled = gr.Checkbox(value=False, label=text("enabled")[0],
                                  elem_id=f"{PREFIX}_enabled", scale=0, min_width=140)
            from .registry import ADAPTER_CHOICES
            # Keep display labels as actual values for Forge UiLoadsave.
            adapter = gr.Dropdown(
                choices=[label for label, _ in ADAPTER_CHOICES],
                value=DEFAULTS[7], label=text("adapter")[0],
                elem_id=f"{PREFIX}_adapter", interactive=True, scale=0, min_width=200,
            )
            phi = gr.Slider(minimum=0, maximum=20, step=0.1, value=4.0,
                            label=text("phi")[0], elem_id=f"{PREFIX}_phi", scale=1, min_width=180)
            tau = gr.Slider(minimum=0.01, maximum=20, step=0.01, value=2.5,
                            label=text("tau")[0], elem_id=f"{PREFIX}_tau", scale=1, min_width=180)
            alpha = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25,
                              label=text("alpha")[0], elem_id=f"{PREFIX}_alpha", scale=1, min_width=180)
        # Preserve the eight API arguments and per-image PNG Info bindings.
        start = gr.Number(value=DEFAULTS[5], precision=None, visible=False,
                          label=text("sigma_start")[0], elem_id=f"{PREFIX}_sigma_start")
        end = gr.Number(value=DEFAULTS[6], precision=None, visible=False,
                        label=text("sigma_end")[0], elem_id=f"{PREFIX}_sigma_end")
        start.do_not_save_to_config = end.do_not_save_to_config = True
        gr.HTML(tooltip_metadata(lang))
    controls = [enabled, negative, phi, tau, alpha, start, end, adapter]
    fields = [(control, functools_partial(index, settings_defaults=settings_defaults))
              for index, control in enumerate(controls)]
    return controls, fields


def functools_partial(index, *, settings_defaults=False):
    from functools import partial
    return partial(_paste_value, index, settings_defaults=settings_defaults)
