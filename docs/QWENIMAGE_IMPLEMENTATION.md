# Qwen-Image 2512 NAG implementation

Reviewed against Forge Neo `neo` at `27f770cb5dbfb6e9ee54f48fa2867346d5ac5d11`.

## Target and limitations

The initial adapter ID is `qwenimage`, using the loaded
`backend.diffusion_engine.qwen.QwenImage` engine and native
`backend.nn.qwen.QwenImageTransformer2DModel` implementation.
This is **not** the separate `QwenImage21` / Qwen-Image 2.1 family.

The implementation targets **txt2img without image references** using the standard
Qwen-Image architecture, including Qwen-Image-2512. No generation parameter or
Forge UI preset is rewritten. Native CFG / unconditional and standard Negative
remain controlled by Forge; NAG has its own plain-text negative. The current
initial limit is 2048 tokenizer tokens including the Qwen2.5-VL template.

## Model-specific attention contract

Native image projection: `to_q/to_k/to_v`; native text projection:
`add_q_proj/add_k_proj/add_v_proj`. The adapter projects image once per block,
projects positive text once, and projects negative text separately. Image Q/K/V
are **the identical RoPE-applied tensors** in both positive and negative joint
attention calls, preserving native FP8/LoRA-compatible layers and backend selection.
Only image attention rows are guided using existing `guide_attention()`.
Positive text and negative text each advance through native output projection,
residual, and MLP operations. The image state is a single shared stream.

The outer native `QwenImageTransformer2DModel.forward` is executed through a
request-local view, preserving native latent patching, image/ref token IDs,
timestep embedding, final normalization, output projection, and cropping.
The view intercepts native positive rotary ID construction, checks the native
text position contract (three identical advancing axes), and builds negative text
IDs from the **captured native text start**, not a guess from prompt length.
Negative text and image Q/K/V use their appropriate native RoPE frequencies.

The original model and its modules are never monkey-patched. Preflight and
each active call verify structural ownership and absence of competing patches.
The request-local negative tensor and temporary states are released in `finally`.

## Qwen-Image-Edit-2511 (separate Phase B, not enabled)

Qwen-Image-Edit-2511 shares the basic architecture but uses image references
in its native pipeline and has official transformer config `zero_cond_t=true`.
Current Forge `backend/nn/qwen.py` does not expose an explicit `zero_cond_t`
constructor parameter. This adapter deliberately rejects Forge edit/reference
requests rather than silently guide reference tokens or change native conditioning.
The Edit phase requires further reference-token masking, image-conditioned
negative-context decisions, and real GPU parity before enabling.

Official model configs:
- https://huggingface.co/Qwen/Qwen-Image-2512/blob/main/transformer/config.json
- https://huggingface.co/Qwen/Qwen-Image-Edit-2511/blob/main/transformer/config.json

Forge source:
- https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/backend/nn/qwen.py
- https://github.com/Haoming02/sd-webui-forge-classic/blob/neo/backend/diffusion_engine/qwen.py
