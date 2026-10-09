# Changelog

## 0.5.1 — 2026-10-09

- Fix manual Adapter selection resetting to Auto after saving/reloading Forge Neo UI defaults: Dropdown choice values now match Forge UiLoadsave's display-label validation.
- Keep stable adapter IDs for NAG configuration, 7-/8-argument API calls and PNG Info metadata; convert IDs/legacy labels to visible Dropdown choices during PNG Info restore.
- Add real Gradio/Forge radio-choice contract regression tests for all seven manual adapters in Japanese and English, plus API, metadata and invalid-choice fallback checks.
- Previously saved UI-default internal IDs may require re-selection and saving once; user settings files are not automatically modified.
- No changes to Forge Neo core, model adapters, CFG or sampling math. Qwen-Image-Edit-2511 remains on hold.

## 0.5.0 — 2026-10-08

- Add Qwen-Image joint-attention NAG adapter for the native Forge `QwenImage` model family, including Qwen-Image-2512 txt2img without references.
- Auto/manual adapter routing uses the loaded `backend.diffusion_engine.qwen.QwenImage` engine; Qwen-Image-2.1 remains a separate unsupported model.
- Reuse native Qwen2.5-VL 7B negative conditioning, native Qwen-Image forward, 3-axis RoPE, time modulation and projection layers; share image Q/K/V and guide image rows only.
- Keep Forge native CFG, standard Negative, preset and sampler settings; reject Edit-2511/reference mode until Phase B native-edit and GPU parity checks.
- Add reduced native-contract tests, CFG mixed-branch/unconditional checks, an independent CPU regression workflow, and implementation/validation documentation.
- CPU tests do not validate pretrained GPU, FP8, real LoRA, visual suppression or Qwen-Image-Edit-2511.

## 0.4.2 — 2026-10-08

- Release the post-install Z-Image / Ernie Block/Attention mutation guard already committed to main (F01).
- Replace reduced CPU RoPE test doubles with native-style Ernie split-half and Z-Image interleaved rotations (F02).
- Add attention-input image Q/K/V sharing checks and late-mutation regression cases (F02).
- Correct bilingual README multi-adapter, CFG and eight-control usage text, plus stale supported-model labels (F03).
- No model weights, guidance math, preset, sampler or CFG ownership changes.
- Full pytest rerun and pretrained-model GPU validation remain pending.

## 0.4.1 — 2026-10-08

- Reject Forge Sparse Attention's `optimized_attention_override` before and during NAG sampling.
- Fix manual Adapter save/restore by normalizing display labels and stable internal IDs through one registry mapping.
- Add regression coverage for Sparse Attention conflicts and all six manual adapters across English/Japanese UI restore cases.
- No model architecture, CFG math, sampler, scheduler or preset behavior is changed by this maintenance release.

## 0.4.0 — 2026-10-08

- Added Z-Image Turbo (ZIT) adapter with Hybrid RoPE and shared image Q/K/V.
- Fixed the ZIT noise-refiner contract to pass the computed `adaln_input`, matching Forge `NextDiT.forward`.
- Added Ernie Image adapter with native Ministral3 conditioning, joint-attention NAG and Hybrid RoPE.
- Extended Auto/manual adapter routing and UI to ZIT and Ernie.
- Preserved native Forge CFG ownership, standard Negative branch and user generation settings.
- Pretrained-model GPU validation for ZIT/Ernie remains pending; the publishing workflow performs syntax/package verification, not full GPU or image-quality validation.

## 0.3.0 — 2026-10-08

- Added Klein / Flux.2 adapter and Auto routing.
- Added Klein DoubleStream/SingleStream NAG path while preserving native CFG.
- Added Klein implementation/validation docs and CPU regression coverage.
- CPU suite after Klein: 315 passed; real pretrained-model GPU validation remains pending.

## 0.2.0 — 2026-10-08

- Added Anima and SDXL / Illustrious adapters.
- Added manual adapter selection and removed the CFG=1-only restriction.
- Preserved Forge preset/generation settings and native CFG ownership.

## 0.1.0 — 2026-10-07

Initial implementation; CPU-validated, pretrained-model GPU validation pending.

- Independent Krea2 NAG negative prompt in a Forge Neo always-on extension.
- Shared-image attention-space NAG with FP32 L1 normalization.
- Shared image Q/K/V projection and sampling-local negative text-fusion cache.
- Request-local sampling guard and predictor-preserving KModel delegation.
- Explicit unsupported-mode/conflict rejection and stale-state cleanup.
- English/Japanese controls, scoped hover help, README and generation metadata restoration.
- CPU regression tests, native-source probe and a real-environment acceptance checklist.

No additional model adapters are advertised by this initial version.
