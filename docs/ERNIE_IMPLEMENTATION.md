# Ernie Image NAG implementation — development gate

Date: 2026-10-08 JST. Target Forge Neo `neo` head reviewed: `90c2766d4630f3629d4cc7bd7afad587b6ca690a`.
Ernie blobs are unchanged from the earlier review: engine `6ec48d7f`, model `b71e6ef6`, text engine `e0a4d820`.

## Contract

Ernie uses `ErnieImage` + `ErnieImageModel` and native `Ministral3Engine`. The adapter is
positive-only: Forge keeps native CFG ownership and the ordinary unconditional branch is never guided.

The outer native `ErnieImageModel.forward` is reused through a request-local read-only view. Only the
36-style main block list is substituted with block views. Native image/text projection, timestep projection,
time embedding, AdaLN modulation, position construction, final norm and final projection remain in Forge's
own forward path.

Each block projects the positive `[image | text]` sequence once. Negative text is projected separately;
positive image Q/K/V, including already-applied image RoPE, is reused for the negative joint attention.
NAG mixes image attention rows only. Positive and negative text states advance independently through native
attention output projection, residuals and MLP.

Ernie image position axis 0 depends on positive text length. The negative branch therefore uses hybrid RoPE:
negative text gets its own text positions while image Q/K/V and image RoPE are copied from the positive branch.

## Initial boundaries

txt2img only. Reference/Edit, Hires, img2img, ControlNet, CFG++, token merging, Positive AND, compiled
models and foreign attention/guidance patches are rejected. Static LoRA/native quantized Linear layers are
structurally reused but require real GPU validation.

The NAG negative uses native Ministral3 conditioning with a conservative extension safety limit of 2048 tokens;
this is not claimed as the model's architectural maximum.
