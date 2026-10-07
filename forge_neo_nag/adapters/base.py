"""Request-local cross-attention state, branch routing and strict boundaries."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from ..config import NAGConfig, NAGError
from ..math import guide_attention


class Adapter(Protocol):
    name: str
    negative_context: torch.Tensor | None

    def clear(self) -> None: ...


def has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, torch.Tensor):
        return value.numel() > 0
    return bool(value)


def check_unmodified(module, label, methods=("forward",)):
    if hasattr(module, "_orig_mod"):
        raise NAGError(f"Compiled {label} is not supported by NAG.")
    if (any(name in vars(module) for name in methods)
            or getattr(module, "_forward_hooks", {})
            or getattr(module, "_forward_pre_hooks", {})):
        raise NAGError(f"Another extension has patched {label}; NAG refuses to bypass it.")


def context_tensor(value, *, width, name, max_tokens=None):
    if (not isinstance(value, torch.Tensor) or value.ndim != 3
            or value.shape[0] != 1 or value.shape[1] < 1 or value.shape[2] != width):
        raise NAGError(f"Unexpected {name} negative conditioning shape; expected [1, tokens, {width}].")
    if max_tokens is not None and value.shape[1] > max_tokens:
        raise NAGError(f"{name} NAG negative exceeds {max_tokens} encoded tokens; it was not truncated.")
    value = value.detach().to(device="cpu").contiguous()
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise NAGError(f"{name} negative conditioning contains non-finite or non-floating values.")
    return value


def validate_literal_text(text):
    """Allow escaped tag punctuation, not implicit CLIP/Qwen emphasis."""
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char in "()[]":
            raise NAGError(
                "NAG Negative accepts literal text/tags, not emphasis. Escape literal parentheses "
                "as \\( and \\). / 強調構文は非対応です。タグ内の括弧はエスケープしてください。"
            )


@dataclass(frozen=True)
class BranchLayout:
    labels: tuple[int, ...]
    chunk_size: int
    positive_rows: tuple[int, ...]

    @classmethod
    def from_payload(cls, payload, batch):
        options = payload.get("c", {}).get("transformer_options", {})
        a = payload.get("cond_or_uncond")
        b = options.get("cond_or_uncond")
        if a is not None and b is not None and list(a) != list(b):
            raise NAGError("Conflicting CFG branch labels in payload and transformer_options.")
        labels = a if a is not None else b
        if labels is None:
            # Forge's native sampling contract always labels conditional rows.
            raise NAGError("CFG branch labels are missing; refusing to guide an unknown branch.")
        if (not isinstance(labels, (list, tuple)) or not labels
                or any(type(x) is not int or x not in (0, 1) for x in labels)
                or batch <= 0 or batch % len(labels)):
            raise NAGError("Unexpected CFG batch layout; only native conditional/unconditional rows are supported.")
        if labels.count(0) > 1 or labels.count(1) > 1:
            raise NAGError("Multiple composed/region conditions are not supported by NAG yet.")
        size = batch // len(labels)
        rows = tuple(row for i, label in enumerate(labels) if label == 0
                     for row in range(i * size, (i + 1) * size))
        return cls(tuple(labels), size, rows)


class CrossAttentionAdapter:
    """One session, one denoiser call at a time. No shared model mutation."""
    def __init__(self, model, negative_context, config: NAGConfig):
        self.model = model
        self.negative_context = negative_context
        self.config = config
        self.attention_calls = 0
        self.targets = ()
        self._context_key = None
        self._context = None
        self._rows = None
        self._batch = None
        self._indices = None
        self._seen = {}

    def to(self, *args, **kwargs):
        return self

    def __deepcopy__(self, memo):
        return self

    def begin_call(self, rows, batch, device):
        if self._rows is not None:
            raise NAGError("Nested NAG model calls are not supported.")
        self._rows = tuple(rows)
        self._batch = batch
        self._indices = (None if len(rows) == batch else
                         torch.tensor(rows, dtype=torch.long, device=device))
        self._seen = {}

    def end_call(self, *, completed=False):
        seen = self._seen
        self._rows = self._indices = self._batch = None
        self._seen = {}
        if completed and (set(seen) != set(self.targets) or any(n != 1 for n in seen.values())):
            raise NAGError("NAG did not execute every expected Cross-Attention exactly once; generation rejected.")

    def selected(self, value):
        if self._rows is None or not self._rows or value.shape[0] != self._batch:
            raise NAGError("NAG Cross-Attention was called outside its owned conditional batch.")
        return value if self._indices is None else value.index_select(0, self._indices)

    def negative_for(self, like):
        if self.negative_context is None:
            raise NAGError("NAG sampling cache has already been released.")
        key = (like.device, like.dtype)
        if key != self._context_key:
            self._context = self.negative_context.to(device=like.device, dtype=like.dtype)
            self._context_key = key
        return self._context.expand(like.shape[0], -1, -1)

    def mix(self, positive, negative, key):
        result = guide_attention(self.selected(positive), negative,
                                 phi=self.config.phi, tau=self.config.tau,
                                 alpha=self.config.alpha)
        self.attention_calls += 1
        self._seen[key] = self._seen.get(key, 0) + 1
        if self._indices is None:
            return result
        # Never alter the native unconditional rows, even when UNCOND comes first.
        output = positive.clone()
        output.index_copy_(0, self._indices, result)
        return output

    def clear(self):
        self.negative_context = self._context = self._context_key = None
        self._rows = self._indices = self._batch = None
        self._seen = {}
