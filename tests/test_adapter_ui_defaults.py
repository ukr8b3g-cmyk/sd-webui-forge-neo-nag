"""Forge Neo UI defaults regression: Gradio labels versus internal adapter IDs.

Forge UiLoadsave.radio_choices() returns the *first* item of each Gradio
choice pair, and UiLoadsave.add_component() only accepts saved values present
in that list. These tests reproduce that exact contract with real Gradio
Dropdowns, separately from NAG's PNG Info and API serialization.

Forge source: modules/ui_loadsave.py at
Haoming02/sd-webui-forge-classic@899cf6955d67da768bc0fb59ba3dce6025287ccd
"""
from __future__ import annotations

import pytest

from forge_neo_nag.config import NAGConfig
from forge_neo_nag.registry import (
    ADAPTER_CHOICES, adapter_display_name, normalize_choice,
)
from forge_neo_nag.ui import DEFAULTS, META_KEYS, build_ui


def _forge_radio_choices(component):
    # Keep aligned with native Forge UiLoadsave.radio_choices().
    return [choice[0] if isinstance(choice, tuple) else choice
            for choice in getattr(component, "choices", [])]


@pytest.mark.parametrize("locale", ["None", "ja_JP"])
@pytest.mark.parametrize("display,internal", ADAPTER_CHOICES[1:])
def test_forge_ui_defaults_manual_adapter_save_and_reload(locale, display, internal):
    """14 cases: selected displayed value survives the same-language restart."""
    import gradio as gr

    with gr.Blocks():
        controls, fields = build_ui(gr, locale)
    selector = controls[-1]
    assert selector.value == "Auto"
    assert _forge_radio_choices(selector) == [
        label for label, _ in ADAPTER_CHOICES
    ]
    assert selector.preprocess(display) == display
    assert display != internal  # The old version persisted the wrong form.

    # Forge's UI-defaults Apply serializes the component's submitted value.
    saved_default = selector.preprocess(display)
    with gr.Blocks():
        restarted, restarted_fields = build_ui(gr, locale)
    reload_selector = restarted[-1]

    # UiLoadsave.add_component accepts values found in radio_choices().
    if saved_default in _forge_radio_choices(reload_selector):
        reload_selector.value = saved_default
    assert reload_selector.value == display
    assert normalize_choice(reload_selector.value) == internal
    assert NAGConfig.parse(
        True, "glasses", adapter=reload_selector.value
    ).adapter == internal

    # PNG Info continues to serialize a stable ID, not the display label.
    metadata = NAGConfig.parse(
        True, "glasses", adapter=reload_selector.value
    ).metadata()
    assert metadata[META_KEYS[-1]] == internal
    restored_label = restarted_fields[-1][1](metadata)
    assert restored_label == display
    assert reload_selector.preprocess(restored_label) == display
    assert normalize_choice(restored_label) == internal


@pytest.mark.parametrize("display,internal", ADAPTER_CHOICES)
def test_api_and_infotext_accept_both_adapter_representations(display, internal):
    # API can continue using old seven arguments, or an eighth internal ID.
    assert NAGConfig.parse(
        True, "glasses", 4., 2.5, .25, 1000., 0., internal
    ).adapter == internal
    assert NAGConfig.parse(
        True, "glasses", 4., 2.5, .25, 1000., 0., display
    ).adapter == internal
    assert adapter_display_name(internal) == display
    assert adapter_display_name(display) == display


@pytest.mark.parametrize("locale", ["None", "ja_JP"])
def test_auto_default_and_invalid_png_choice_stay_safe(locale):
    import gradio as gr
    with gr.Blocks():
        controls, fields = build_ui(gr, locale)
    assert DEFAULTS[-1] == "Auto"
    assert controls[-1].value == "Auto"
    assert controls[-1].preprocess("Auto") == "Auto"
    assert fields[-1][1]({}) == "Auto"
    assert fields[-1][1]({META_KEYS[-1]: "unrecognized"}) == "Auto"
    assert NAGConfig.parse(True, "glasses", adapter="Auto").adapter == "auto"


def test_display_labels_remain_unique():
    labels = [label for label, _ in ADAPTER_CHOICES]
    internal = [adapter for _, adapter in ADAPTER_CHOICES]
    assert len(labels) == len(set(labels))
    assert len(internal) == len(set(internal))
