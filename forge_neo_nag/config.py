"""Strict, host-independent configuration for UI and always-on API requests."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any


class NAGError(RuntimeError):
    """A request cannot safely execute the requested NAG operation."""


def boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in ("true", "false"):
        return value.lower() == "true"
    raise NAGError("Enable NAG must be true or false. / Enable NAGは真偽値で指定してください。")


def _number(name: str, value: Any, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise NAGError(f"{name} must be a number, not a boolean.")
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise NAGError(f"{name} must be a finite number in [{low}, {high}].") from exc
    if not math.isfinite(result) or not low <= result <= high:
        raise NAGError(f"{name} must be a finite number in [{low}, {high}].")
    return result


@dataclass(frozen=True)
class NAGConfig:
    enabled: bool = False
    negative: str = ""
    phi: float = 4.0
    tau: float = 2.5
    alpha: float = 0.25
    sigma_start: float = 1000.0
    sigma_end: float = 0.0
    adapter: str = "auto"

    @classmethod
    def parse(cls, enabled=False, negative="", phi=4.0, tau=2.5, alpha=0.25,
              sigma_start=1000.0, sigma_end=0.0, adapter="auto") -> "NAGConfig":
        enabled = boolean(enabled)
        # OFF must not inspect a model or validate inactive stored controls.
        if not enabled:
            return cls()
        if not isinstance(negative, str):
            raise NAGError("NAG Negative Prompt must be text. / NAG Negative Promptは文字列で指定してください。")
        if len(negative) > 32768:
            raise NAGError("NAG Negative Prompt exceeds 32768 characters; it was not truncated.")
        from .registry import normalize_choice
        # Inactive/bypassed controls must not force model detection.
        selected = "auto"
        result = cls(enabled, negative.strip(), _number("Phi", phi, 0, 20),
                     _number("Tau", tau, 0.01, 20), _number("Alpha", alpha, 0, 1),
                     _number("Sigma Start", sigma_start, 0, 1000),
                     _number("Sigma End", sigma_end, 0, 1000), selected)
        if result.sigma_start < result.sigma_end:
            raise NAGError("Sigma Start must be >= Sigma End. / 開始Sigmaは終了Sigma以上にしてください。")
        from dataclasses import replace
        try:
            selected = normalize_choice(adapter)
        except NAGError:
            if result.active:
                raise
            selected = "auto"
        result = replace(result, adapter=selected)
        if result.active:
            validate_negative_text(result.negative)
        return result

    @property
    def active(self) -> bool:
        return self.enabled and bool(self.negative) and self.phi > 0 and self.alpha > 0

    @property
    def bypass_reason(self) -> str:
        if not self.enabled:
            return "disabled"
        if not self.negative:
            return "empty negative"
        if self.phi == 0:
            return "phi=0"
        if self.alpha == 0:
            return "alpha=0"
        return ""

    def metadata(self, model=None) -> dict[str, Any]:
        from . import __version__
        data = {
            "Forge NAG": self.enabled,
            "Forge NAG Version": __version__,
            "Forge NAG Adapter Selection": self.adapter,
            "Forge NAG Negative": self.negative,
            "Forge NAG Phi": self.phi,
            "Forge NAG Tau": self.tau,
            "Forge NAG Alpha": self.alpha,
            "Forge NAG Sigma Start": self.sigma_start,
            "Forge NAG Sigma End": self.sigma_end,
        }
        if model is not None:
            data["Forge NAG Model"] = model
        return data


def validate_negative_text(text: str) -> None:
    """Reject known WebUI syntax rather than silently treating it as a control."""
    patterns = (
        r"<\s*(?:lora|lyco|hypernet)\s*:",
        r"\[(?:[^\[\]]*:[^\[\]]*:[\d.]+|[^\[\]]*\|[^\[\]]*)\]",
        r"\([^()]*:[+-]?(?:\d+(?:\.\d*)?|\.\d+)\)",
        r"(?<!\w)(?:AND|BREAK)(?!\w)",
    )
    if any(re.search(pattern, text) for pattern in patterns):
        raise NAGError(
            "NAG Negative accepts plain text, not LoRA tags, prompt schedules, "
            "weights, AND or BREAK. Put LoRA tags in the positive prompt. / "
            "NAG欄は通常の文章のみです。LoRAはPositive欄へ指定してください。"
        )
