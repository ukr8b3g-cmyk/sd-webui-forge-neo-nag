# Klein / Flux.2 validation — 2026-10-08

## Result

Development CPU suite after adding Klein: **315 passed, 0 failed** with the existing v0.2 reduced native-source fixtures enabled. The same run includes 8 new Klein-specific tests plus registry/UI regressions. No checkpoint or GPU generation was executed.

Environment for this validation is the ChatGPT Linux container. Existing v0.2 native UNet/Anima/KModel probes use their previously captured Forge definitions. Klein uses a reduced random CPU Flux2 fixture modeled directly on the reviewed Forge Neo `831d242` contract, including the actual Flux2 global-modulation path, four RoPE axes, `txt_ids_dims=[3]`, and SiLU-gated SingleStream MLP layout.

## New Klein checks

- Auto and manual `klein` adapter routing.
- Engine/model/block structure validation.
- Qwen negative flag and 2048-token rejection without truncation.
- Different positive and negative text lengths.
- Alpha=0 adapter math reproduces the native reduced model path.
- Nonzero negative context changes finite output while input tensors and weights remain unchanged.
- DoubleStream image Q/K/V are projected once; negative branch projects text only.
- SingleStream performs one positive full-sequence projection plus one negative-text-only projection.
- Native unconditional branch remains byte-equal in combined CFG calls for both `[COND, UNCOND]` and `[UNCOND, COND]` ordering.
- Positive `AND`, active references, and pre-existing Flux patches are rejected.
- Full existing Krea2/Anima/SDXL regression suite remains green.

## Source review facts

Forge Neo `neo` at `831d242d4cf45a2ba1a64c2087ef0a17934a9daf` currently identifies Klein as `Flux2`; the transformer implementation is `IntegratedFluxTransformer2DModel`. The 4B config contains 5 DoubleStream and 20 SingleStream blocks, while 9B contains 8 and 24. Detection config sets `global_modulation=True`, `mlp_silu_act=True`, four RoPE axes, and `txt_ids_dims=[3]`. These facts were checked from the live GitHub source during this implementation.

## Remaining gates

Not tested: real Forge Neo startup/UI interaction, Klein 4B/9B checkpoints, FP16/BF16/FP8 or mixed quantization on GPU, real LoRA combinations, visual suppression, OFF/ON/OFF image parity, speed, VRAM/RAM, cancellation, or long-run stability. Those must remain separate from CPU implementation status.
