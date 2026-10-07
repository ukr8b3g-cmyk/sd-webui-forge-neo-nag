# Klein / Flux.2 NAG implementation — development gate

Date: 2026-10-08 JST. This document describes the Klein-only implementation layered on top of Forge Neo NAG v0.2.0. It is not a release note and does not claim pretrained-model GPU validation.

## Reviewed Forge contract

Implementation target: `Haoming02/sd-webui-forge-classic`, `neo` commit `831d242d4cf45a2ba1a64c2087ef0a17934a9daf`.

Reviewed blobs:

- `backend/diffusion_engine/flux2.py` — `1ff13a1544917526b834ba417248aefdc07651e1`
- `backend/nn/flux.py` — `fba3c246553d54759f126d995ff64ada5ff8d9b3`
- `backend/text_processing/klein_engine.py` — `ca6f91d343b1f2362a34ff0410780eaadbac37dd`
- `backend/modules/k_model.py` — `0f09636d1e9919ef54c1a47c6c034ab08a922cc4`

Forge detects Flux.2 Klein with global modulation, `mlp_silu_act=True`, four RoPE axes and `txt_ids_dims=[3]`. The official Forge configs use 5 DoubleStream + 20 SingleStream blocks for Klein 4B and 8 + 24 for Klein 9B.

## Adapter boundary

`registry.py` adds `klein -> backend.diffusion_engine.flux2.Flux2`. Auto selection uses the loaded engine type, not UI Preset. Manual `Klein / Flux.2` remains available and structural checks still reject a mismatched model.

The adapter reuses the loaded native `IntegratedFluxTransformer2DModel`, native Linear/LoRA/quantization layers and native `backend.nn.flux.attention`. It does not replace shared model methods. A request-local `transformer_options` copy installs:

1. one owned `post_input` observer to build the negative text projection and its RoPE from the native image IDs;
2. owned `dit` replacements for every DoubleStream block;
3. owned `dit` replacements for every SingleStream block.

Any pre-existing Flux `patches` or `patches_replace` is rejected rather than merged silently.

## Negative conditioning

The existing Klein `Qwen3_4B_8B_Engine` is reused with `is_negative_prompt=True`. Raw Qwen hidden states are not invented or loaded separately. The result must be one finite `[tokens, context_width]` tensor and is stored on CPU between sampling calls. Both raw and encoded lengths are capped at 2048 without truncation. The encoder's native minimum padding remains intact.

## DoubleStream blocks

For each block, positive image Q/K/V are projected exactly once. Positive and negative text Q/K/V are projected independently with the native `txt_attn` layers. The two attention sequences are:

- positive: `[positive_text | image]`
- negative: `[negative_text | same_image]`

Each sequence uses its own text RoPE, while image IDs are identical in both PE tensors. NAG is applied only to the image attention rows. Positive text follows its native result; negative text advances through native text projection, residual and MLP for use by the next block.

## SingleStream blocks

The positive full sequence is projected once. Only negative text is passed through the additional `linear1`; image Q/K/V are reused from the positive projection. Attention is evaluated for `[positive_text | image]` and `[negative_text | same_image]`; only image attention rows are guided. Native positive MLP activations are retained. Negative text receives its own attention/MLP update. Both standard SiLU-gated Flux2 MLP and the Yak MLP branch are preserved.

The native final layer receives only the guided positive image state. No negative image residual stream is maintained.

## CFG and unsupported features

Klein uses the same positive-only wrapper strategy as Krea2. When Forge supplies a native mixed CFG call, the wrapper splits by `cond_or_uncond`, runs the original unconditional branch unchanged, runs NAG only on the conditional branch, restores row order, and leaves final `u + cfg * (c-u)` ownership to Forge. CFG is never rewritten.

Initial Klein support remains txt2img/no-reference. Reference/Edit, ControlNet, Hires, img2img, CFG++, token merging, positive `AND`, existing Flux block/input replacement patches and compiled models are rejected. Static LoRA and native quantized Linear layers are structurally reused but still require real GPU validation.
