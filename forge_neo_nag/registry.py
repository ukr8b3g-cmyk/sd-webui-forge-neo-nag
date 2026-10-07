"""Adapter choice is user-controlled; Forge UI presets are never written.

Auto identifies the loaded engine. A manual choice bypasses preset hints, not
structural compatibility checks. Imports of a model family remain lazy.
"""
from __future__ import annotations

from .config import NAGError

ADAPTER_IDS = ("auto", "krea2", "anima", "sdxl", "klein")
ENGINE_TYPES = {
    "krea2": ("backend.diffusion_engine.krea", "Krea2"),
    "anima": ("backend.diffusion_engine.anima", "Anima"),
    "sdxl": ("backend.diffusion_engine.sdxl", "StableDiffusionXL"),
    "klein": ("backend.diffusion_engine.flux2", "Flux2"),
}


def normalize_choice(value) -> str:
    if not isinstance(value, str):
        raise NAGError("NAG Adapter must be Auto, Krea2, Anima, SDXL or Klein. / アダプターを選択してください。")
    value = value.strip().lower()
    if value not in ADAPTER_IDS:
        raise NAGError(f"Unknown NAG adapter '{value}'. Select Auto, Krea2, Anima, SDXL or Klein.")
    return value


def choose_adapter(engine, requested="auto", preset=None) -> str:
    """Pure selection. Preset is informational, never a rejection criterion.

    Qualified names find a candidate without importing every backend. The host
    subsequently imports the selected native class and checks actual isinstance
    and model-layout compatibility; names alone never authorize a patch.
    """
    selected = normalize_choice(requested)
    if selected != "auto":
        return selected
    mro = {(cls.__module__, cls.__name__) for cls in type(engine).__mro__}
    matches = [key for key, qualname in ENGINE_TYPES.items() if qualname in mro]
    if len(matches) != 1:
        raise NAGError(
            "NAG Auto could not identify this model. Choose a compatible adapter manually "
            "or disable NAG. / 自動判定できません。対応するアダプターを手動選択するかNAGをOFFにしてください。"
        )
    return matches[0]
