"""Bilingual UI strings and scoped tooltip metadata; no model imports."""
from __future__ import annotations

import html
import json
import re

PREFIX = "forge_neo_nag"
FIELDS = ("enabled", "negative", "phi", "tau", "alpha", "sigma_start", "sigma_end")
DEFAULTS = (False, "", 4.0, 2.5, 0.25, 1000.0, 0.0)
META_KEYS = ("Forge NAG", "Forge NAG Negative", "Forge NAG Phi", "Forge NAG Tau", "Forge NAG Alpha", "Forge NAG Sigma Start", "Forge NAG Sigma End")

STRINGS = {
    "enabled": {
        "en": ("Enable NAG", "V1: Krea2 txt2img at CFG=1 only. OFF, an empty NAG prompt, Phi=0 or Alpha=0 uses the normal path. No automatic CFG changes."),
        "ja": ("NAGを有効化", "V1はKrea2のtxt2img・CFG=1専用です。OFF・NAG欄が空・Phi=0・Alpha=0では通常経路を使います。CFGは自動変更しません。"),
    },
    "negative": {
        "en": ("NAG Negative Prompt", "Describe what to suppress, for example: big wings. This is separate from the greyed-out standard negative box. Plain text only; no weights, schedules or LoRA tags. Maximum 2048 tokens including the template."),
        "ja": ("NAG Negative Prompt（抑制したい内容）", "例：big wings のように抑制したいものを書きます。グレーアウトする標準Negative欄とは別です。通常の文章のみ。重み・スケジュール・LoRAタグは非対応です。テンプレート込みで最大2048トークンです。"),
    },
    "phi": {
        "en": ("NAG Scale / Phi", "Attention extrapolation strength. Start at 4.0. A higher value can suppress more strongly but can change composition or degrade quality. Zero bypasses NAG."),
        "ja": ("NAG強度 / Phi", "Attentionの差を強める係数です。まず4.0から試します。上げすぎると構図や画質が変化する場合があります。0でNAGをバイパスします。"),
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


def _paste_value(index: int, params):
    value = params.get(META_KEYS[index], DEFAULTS[index])
    if index == 0:
        # Loading a non-NAG image must not leave a previously enabled NAG on.
        return value is True or str(value).lower() == "true"
    return value


def build_ui(gr, localization="None"):
    lang = language(localization)
    text = lambda field: STRINGS[field][lang]
    with gr.Accordion("Forge Neo NAG — Krea2", open=False, elem_id=f"{PREFIX}_txt2img"):
        gr.Markdown(
            "**V1：Krea2 / txt2img / CFG=1**。標準Negative欄は変更しません。Reference・Edit・Hires fixは非対応です。"
            if lang == "ja" else
            "**V1: Krea2 / txt2img / CFG=1**. The standard negative box is unchanged. Reference, Edit and Hires fix are not supported."
        )
        enabled = gr.Checkbox(value=False, label=text("enabled")[0], info=text("enabled")[1], elem_id=f"{PREFIX}_enabled")
        negative = gr.Textbox(value="", label=text("negative")[0], info=text("negative")[1],
                              placeholder="big wings", lines=3, elem_id=f"{PREFIX}_negative")
        with gr.Row():
            phi = gr.Slider(minimum=0, maximum=20, step=0.1, value=4.0,
                            label=text("phi")[0], info=text("phi")[1], elem_id=f"{PREFIX}_phi")
            tau = gr.Slider(minimum=0.01, maximum=20, step=0.05, value=2.5,
                            label=text("tau")[0], info=text("tau")[1], elem_id=f"{PREFIX}_tau")
            alpha = gr.Slider(minimum=0, maximum=1, step=0.01, value=0.25,
                              label=text("alpha")[0], info=text("alpha")[1], elem_id=f"{PREFIX}_alpha")
        with gr.Accordion("詳細設定 / Advanced", open=False):
            start = gr.Number(value=1000.0, precision=None,
                              label=text("sigma_start")[0], info=text("sigma_start")[1], elem_id=f"{PREFIX}_sigma_start")
            end = gr.Number(value=0.0, precision=None,
                            label=text("sigma_end")[0], info=text("sigma_end")[1], elem_id=f"{PREFIX}_sigma_end")
        gr.HTML(tooltip_metadata(lang))
    controls = [enabled, negative, phi, tau, alpha, start, end]
    fields = [(control, functools_partial(index)) for index, control in enumerate(controls)]
    return controls, fields


def functools_partial(index):
    from functools import partial
    return partial(_paste_value, index)
