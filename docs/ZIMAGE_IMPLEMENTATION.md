# Z-Image Turbo / ZIT NAG implementation — development gate

Date: 2026-10-08 JST. This document describes the Z-Image Turbo implementation layered on top of Forge Neo NAG v0.3.0. It is not a release note and does not claim pretrained-model GPU validation.

## Reviewed Forge contract

Implementation target: `Haoming02/sd-webui-forge-classic`, `neo` commit `831d242d4cf45a2ba1a64c2087ef0a17934a9daf`.

Reviewed blobs:

- `backend/nn/lumina.py` — `a08ab9e44dbd3b03d303861273b8a5173d09afcc`
- `backend/diffusion_engine/zimage.py` — `4150ed917f769cce41165183fb6bc60c29b49a9c`
- `backend/text_processing/z_image_engine.py` — `c3285411795a2b42a4af40e390ed876b6ba5ea93`
- `backend/modules/k_model.py` — `0f09636d1e9919ef54c1a47c6c034ab08a922cc4`

Forge identifies Z-Image Turbo as the `ZImage` engine backed by `NextDiT`. The current official configuration is patch size 2, latent channels 16, dim 3840, 30 heads, 30 main blocks, two context-refiner and two noise-refiner blocks, QK normalization, three-axis RoPE, `time_scale=1000`, and Z-Image modulation.

## Adapter boundary

`registry.py` adds `zimage -> backend.diffusion_engine.zimage.ZImage`. Auto selection uses the loaded engine type, not UI Preset. Manual `Z-Image Turbo (ZIT)` remains available and structural checks still reject a mismatched model.

The adapter reuses native `NextDiT`, its context/noise refiners, native Linear/LoRA/quantization layers, native RMSNorm/RoPE operations and the currently selected Forge attention backend. It does not assign to shared model methods or mutate the model tree.

Nunchaku Z-Image is intentionally outside the initial contract. Forge changes the runtime model class and replaces attention/feed-forward modules for Nunchaku, so the exact-native model/type checks reject it rather than silently applying an incompatible adapter.

## Negative conditioning

The loaded `Qwen34BEngine` is reused with `is_negative_prompt=True`. Input is capped at 2048 Qwen tokens including the native template and is never silently truncated. The encoded negative tensor is kept on CPU between model calls and projected with the same native `cap_embedder` used by the positive condition.

Known prompt weighting/schedule syntax, AND/BREAK and LoRA tags remain unsupported in the independent NAG negative box.

## Shared image refiner

The native positive path executes `cap_embedder` and `patchify_and_embed` once. That preserves the positive Context Refiner, image Noise Refiner, image padding and image positional IDs exactly as Forge implements them.

The negative branch executes only its own text `cap_embedder` and the two native Context Refiner blocks. It does not execute the image Noise Refiner again. Therefore both branches enter the 30 main blocks with the same refined image state.

## Hybrid RoPE

Z-Image derives image axis-0 positions from the padded positive text length. Recomputing a complete negative position sequence from a different negative-text length would change image RoPE and violate the shared-image NAG assumption.

The adapter therefore uses:

- native positive text + native positive image RoPE for the positive branch;
- negative-text RoPE derived from the negative padded length;
- the already-computed **positive image RoPE/Q/K/V** for the image portion of the negative branch.

No separate negative image position IDs or image Q/K/V projections are created.

## Main JointTransformerBlock processing

For each native main block:

1. the positive full `[text | image]` sequence is normalized/modulated and passed through native `qkv` once;
2. the negative **text-only** state is normalized/modulated and passed through the same native `qkv` layer once;
3. native RMSNorm + RoPE is applied to positive full Q/K and negative text Q/K;
4. negative Joint Attention is assembled as `[negative_text Q/K/V | shared positive image Q/K/V]`;
5. NAG is applied only to raw image attention rows;
6. positive text raw attention is concatenated with guided image raw attention and passed through native `attention.out` once for the positive sequence;
7. negative text raw attention alone is passed through `attention.out` and the native FFN so the negative text state advances independently to the next block.

The final native layer/unpatchify sees only the guided positive sequence. A persistent negative image residual stream is never created.

## CFG and unsupported features

Z-Image uses the same positive-only host-wrapper strategy as Krea2/Klein. For native mixed CFG calls the wrapper splits labelled branches, leaves the unconditional branch byte-native, applies Z-Image NAG only to the conditional branch, restores row order and leaves final CFG combination to Forge. CFG is never rewritten.

Initial support is txt2img/no-reference. Reference/Edit, Hires, img2img, ControlNet, CFG++, token merging, Positive AND, explicit attention masks, pre-existing transformer patches, compiled models and Nunchaku Z-Image are rejected. Static LoRA and native quantized Linear layers are structurally reused but require real GPU validation.
