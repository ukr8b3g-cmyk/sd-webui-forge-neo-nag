# Qwen-Image NAG validation (Phase A)

## CPU gate

A reduced Forge-contract test model reproduces native Qwen-Image Attention and
TransformerBlock projection/modulation/Residual/MLP calls, 5D single-frame
patching, native text/image rotary position layout, time embeddings, and final
output shape. It does **not** use pretrained Qwen weights.

Tests include: alpha=0 native parity at batch 1/2 and unequal positive/negative
text lengths, 3-axis negative text RoPE derived from native positive positions,
shared rotated image Q/K/V at both actual attention inputs, per-block negative
text advance, native encoder negative flag and 2048-token limit, late runtime
block/attention patch rejection, unsupported modes, and resource cleanup.

Host-level checks also cover Auto/manual adapter routing, rejection for unrelated
Qwen-Image-2.1 engine, preserving CFG and native unconditional branch, and
existing-adapter regressions.

## Release boundary

Not verified here: pretrained Qwen-Image-2512 GPU inference, LoRA combinations,
FP8/mixed quantization, VRAM, speed, visual efficacy, image-quality outcomes,
and Qwen-Image-Edit-2511 reference/edit pipelines. A CPU PASS does not imply
these are working. Edit-2511 with references remains explicitly unsupported in
this release; no silent fallback into altered reference conditioning is allowed.
