# Changelog

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
