# Multi-adapter implementation contract — 0.2.0-dev

## Accepted override

The user retains control of CFG, presets, generation settings and NAG ON/OFF.
Adapter selection is an optional eighth always-on argument. Auto identifies the
loaded engine; manual selection overrides Auto, not structural checks. Preset
is read only for metadata. There are no preset/CFG change callbacks or writes.
This supersedes the CFG=1-only and no-manual-selector portions of the 0.1 plan.

## Host and adapter boundary

`registry.py` chooses one family lazily. `HostBindings` retains the original
seven positional fields for the existing CPU harness; model-specific bindings
are appended. The processing-instance sample guard remains outside the host
ScriptRunner's swallowed exceptions. Negative encoding uses the loaded native
text engine with `is_negative_prompt=True`. The original post-LoRA Patcher and
text/model stamp are checked again after other pre-sampling hooks.

Only a Patcher clone receives wrappers and SDXL attention replacements. Native
`KModel.apply_model` preserves input/sigma/predictor/output transformations.
No global forward replacement, model copy, dequantization or new attention backend
is introduced. Krea2's original adapter implementation and `math.py` are unchanged.

`BranchLayout` uses native COND=0 / UNCOND=1 labels, including reversed [1,0],
separate calls and batch>1. Conflicting or missing labels are errors. SDXL and
Anima keep the native mixed batch; only conditional attention rows are mixed.
Krea2 retains its positive-only adapter: the wrapper splits a combined call by
labels, runs the original native unconditional branch, and restores the original
row order. The host continues to perform `u + cfg * (c - u)`; the extension never
installs a CFG-combination function or disables the CFG=1 optimization.

Sigma is checked once per call. Out-of-range or unconditional-only calls use the
native path without negative projections. SDXL removes only identity-verified
owned replacement entries from a copied call-local dictionary. Model outputs
are checked for finite values on each wrapper call, including native uncond.
Model/forward substitution, late foreign patches, missing expected attention
calls or zero total active calls stop the request before save.

Normal, error and interrupt exits restore only owned Patcher/ScriptRunner
references. Foreign replacements are not overwritten. Embeddings and cached
states are cleared. No state is reused across requests or LoRA changes.

## SDXL / Illustrious

The native StableDiffusionXL engine and IntegratedUNet2DConditionModel are checked;
`is_sdxl` or the checkpoint name is insufficient. Four-channel base UNet only;
Refiner, inpaint/video and RF SDXL are outside this initial support contract.

Each native SpatialTransformer is enumerated by input/middle/output block.
The third replacement key is its LOCAL `block_index`, not the global
`transformer_index`. Duplicate keys/targets or unreachable attn2 modules fail.
The existing setter `set_model_attn2_replace` installs one callback per target.
The callback reuses native Q and positive K/V; native Linear layers create the
negative K/V. It returns guided raw attention. Native BasicTransformerBlock calls
`to_out` once and performs its unchanged residual/FF processing.

Both CLIP-L and CLIP-G receive preflight tokenization. Actual 77-position chunks,
not a naive token-count approximation, enforce the four-chunk maximum. TI fixes
are rejected. Native learned conditioning supplies negative `crossattn` only;
negative `vector` is not substituted. The original positive and unconditional
pooled/size/crop vectors remain attached to their native CFG rows.

## Anima

The native Anima engine, DiT, Block and SelfCrossAttention layout are checked.
Native Qwen/T5 preprocessing is reused; raw Qwen hidden states are not passed to
the DiT. Native padding, including the minimum 512 positions, is retained.

Request-local read-only views delegate native Anima.forward and Block.forward.
Only cross_attn is redirected. Self-attention, time embeddings, positional
embeddings, gates, MLP, final layer and FP16/FP32 residual policy are native.
Native normalized image Q is shared between positive and negative cross-attention.
Negative K uses the native k_norm, V the native v_norm. Output projection/dropout
are executed once. Latents are the native five-dimensional single-frame layout.

## Resources and metadata

Only negative embeddings are cached for SDXL/Anima; no all-layer K/V cache.
Estimated additional activations are reserved, including potential CFG batching;
this is not a guarantee of peak memory or OOM avoidance. Normalization is the
unchanged shared FP32 L1 NAG formula, then cast back to computation dtype.

Metadata records selection, actual adapter/model, user CFG, implementation ID,
active model calls and actual attention calls. Krea2-only text-fusion counters
are not fabricated for other families. OFF adds no encoding, patch or NAG
metadata. Blank text / Phi=0 / Alpha=0 are recorded bypasses without host loading.

## Reviewed sources

Extension baseline:
https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-nag/tree/9453000fd3799e81b86e1af584c3d2f807071bcf

Forge Neo source contract:
https://github.com/Haoming02/sd-webui-forge-classic/tree/558e26c3d662238e74285545ca98a139efa695a6

Relevant files: backend/nn/unet.py, backend/nn/anima.py,
backend/diffusion_engine/sdxl.py, backend/diffusion_engine/anima.py,
backend/text_processing/sd_engine.py, backend/text_processing/anima_engine.py,
backend/modules/k_model.py, backend/patcher/base.py,
backend/sampling/sampling_function.py, modules_forge/main_entry.py.

No upstream model source is shipped in this update. Existing LICENSE and NOTICE
are retained in the user's original extension. The new adapters call native
implementations instead of shipping copied UNet/Anima forward code.
