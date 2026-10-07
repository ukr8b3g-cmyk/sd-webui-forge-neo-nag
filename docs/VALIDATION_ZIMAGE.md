# Z-Image Turbo / ZIT validation — 2026-10-08

## Result

CPU suite after adding Z-Image Turbo: **332 passed, 0 failed** with the existing reduced native-source fixtures enabled via `FORGE_NAG_TEST_SOURCE=/mnt/data/nag_work/native`.

The run includes the full existing Krea2/Anima/SDXL/Klein regression suite plus 10 Z-Image-specific reduced CPU tests. No checkpoint, CUDA kernel or image generation was executed.

## Z-Image checks

- Auto and manual `zimage` adapter routing.
- Native NextDiT/JointTransformerBlock/JointAttention contract validation.
- Qwen negative flag and 2048-token rejection without truncation.
- Different positive and negative text lengths.
- Alpha=0 equality with the reduced native Z-Image path.
- Nonzero negative context produces finite changed output.
- Positive full-sequence Q/K/V is projected once per main block; the extra branch projects negative text only.
- Native image Noise Refiner is not duplicated for the negative branch.
- Positive image RoPE is reused by the negative image branch while negative text uses its own positions.
- `pad_tokens_multiple` handling for text/image sequence padding.
- Negative text advances through native attention output, residual and FFN across all main blocks.
- Native unconditional output is preserved exactly for both `[COND, UNCOND]` and `[UNCOND, COND]` host ordering.
- Positive `AND`, active references, explicit attention masks, foreign transformer patches and non-native attention replacement are rejected.
- Existing Krea2/Anima/SDXL/Klein regressions remain green.

## Source review facts

Forge Neo `neo` at `831d242d4cf45a2ba1a64c2087ef0a17934a9daf` uses `ZImage -> NextDiT -> Qwen34BEngine`. The reviewed official transformer configuration has 30 main blocks, two context refiners, two noise refiners, dim 3840, 30 equal Q/K/V heads, patch size 2, QK normalization, RoPE axes 32/48/48 and `time_scale=1000`.

The reduced Z-Image fixture used in CPU tests preserves those method boundaries and positional/padding logic while using small random tensor widths. It is not a pretrained model or a byte-identical Forge checkout.

## Remaining gates

Not tested: real Forge Neo startup/UI interaction, official or merged Z-Image Turbo checkpoints, BF16/FP32/allowed FP16, FP8/mixed quantization on GPU, real LoRA combinations, visual suppression, OFF/ON/OFF image parity, speed, peak VRAM/RAM, cancellation/recovery, or long-run stability. Nunchaku Z-Image remains explicitly unsupported in this initial adapter.
