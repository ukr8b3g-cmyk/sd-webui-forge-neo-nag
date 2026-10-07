# Changelog

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
