# Ernie Image validation plan — 2026-10-08

Implementation adds reduced CPU contracts for Auto/manual routing, native model-layout validation, native
Ministral conditioning, the 2048-token extension safety boundary, different positive/negative text lengths,
alpha=0 native parity, hybrid image RoPE equality, one positive full-sequence projection plus one negative
text-only projection, native unconditional preservation in both CFG branch orders, Positive AND rejection and
Ministral conditioning-stamp invalidation.

The repository publishing workflow compiles committed Python files and checks JavaScript syntax when the package
entry file changes. This does not replace pytest or real Forge/GPU validation.

Remaining gates: pretrained Ernie model generation, BF16/FP16/FP8 or mixed quantization, real LoRA combinations,
visual suppression, OFF/ON/OFF image parity, performance, VRAM/RAM, cancellation and long-run stability.
