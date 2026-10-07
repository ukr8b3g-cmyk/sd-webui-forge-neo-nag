# Implementation / 実装仕様

## Scope

V1 targets native Forge Neo Krea2 text-to-image at CFG=1. The shared numerical core
and model adapter are separated for future models, but no Wan, LTX or Qwen adapter
is registered or advertised as supported. New models require their own attention,
conditioning, reference-layout and verification work.

Reviewed host commit: `Haoming02/sd-webui-forge-classic@bd1d015950b62f1c88a6d95f0b9ea1e0dfc69aa3`.

## Call path

```text
Forge Script.before_process
  └─ arm this processing object's sample callable (not the global class)
       └─ request guard at p.sample
            ├─ parse and validate active parameters
            ├─ check model / CFG / unsupported modes
            ├─ encode one NAG negative with is_negative_prompt=True
            └─ temporarily delegate this request's ScriptRunner
                 ├─ run existing process_before_every_sampling hooks normally
                 └─ validate the final LoRA-applied Patcher OUTSIDE their catch loop
                      ├─ clone Patcher options, not model weights
                      ├─ reserve estimated extra activation memory
                      └─ install this request's model_function_wrapper
                           └─ reuse actual KModel.apply_model on an attribute view
                                ├─ native predictor.calculate_input
                                ├─ native predictor.timestep / dtype conversion
                                ├─ Krea2Adapter attention-space NAG
                                └─ native predictor.calculate_denoised
       ├─ reject missing / zero-active NAG execution
       ├─ add truthful generation metadata
       └─ finally: restore request runner/Patcher and clear negative caches
```

### Why not just clone and assign `model.forward`?

Forge Patcher clones share the same underlying model. Assigning a different forward
on that object can affect other generations even when the Patcher was cloned.
This extension does not assign to the shared model, its forward, block forwards,
weights, or a global attention function. A read-only attribute view substitutes the
DiT callable only while executing the existing `KModel.apply_model` method.
Its original predictor conversions are not duplicated or approximated.

### Why a request-local sampling guard?

The reviewed Forge ScriptRunner catches exceptions from ordinary extension hooks
and logs them. Merely raising an error there may leave generation running without
NAG. The request-local runner view delegates existing callbacks first, then installs
NAG outside that catch loop. Errors propagate to the sampling caller. An additional
postcondition rejects a sampling path that ignored the wrapper entirely, before
returning its latent to the standard save pipeline.

The view and Patcher options are restored in `finally`, including exceptions and
interrupts. Only this extension's installed Patcher is restored; unrelated model
weights or global settings are not reset. OFF does not install the guard or encode
extra text. Shallow-copied processing objects rebind the native sample method to the
new owner; a foreign callable is never unwrapped merely for carrying a copied marker.

These safeguards do not claim compatibility with arbitrary third-party functions
that independently replace the sampling pipeline or save images as side effects.
Known conflicting wrappers and transformer patches are rejected rather than bypassed.

## Mathematics

For positive and negative image attention results `P` and `N`, sharing identical
image queries and image K/V:

```text
E = P + phi * (P - N)
scale = min(1, tau * max(L1(P), eps) / max(L1(E), eps))
R = alpha * (E * scale) + (1 - alpha) * P
```

Norms are per image token over the final feature axis. Extrapolation, norms and blend
use FP32, then cast back to the attention output dtype. `eps` is float32 epsilon.
The scale expression is algebraically equivalent to the upstream ratio/cap formula.
The attention result is modified before the sigmoid gate and output projection,
not as a post-sampling CFG combination. Model weights retain their host precision.

### Shared image projection

Each block starts with one current image state. Project `[positive text | image]`
once and separately project only negative text. Normalize and rotate these Q/K
projections using the host operations. The image Q/K/V slices from the positive
projection are reused when assembling `[negative text | image]`. The negative
attention result updates its own text state; it does not create a second image state.

This saves the negative branch's duplicate image projection and image gate work.
Both attention calls and all per-block negative-text updates still run. It is not
a skipped-layer, skipped-step, approximate-attention or distilled model shortcut.

### Text fusion cache

Krea2's initial `txtfusion + txtmlp` does not depend on diffusion time in the reviewed
native implementation. The negative initial text state is therefore cached once
per sampling session and device/dtype. Each model call starts with a fresh clone;
block updates cannot mutate the cached initial state. The raw negative conditioning
is also protected from the host text fusion's in-place operations.

Caches are cleared at session end. Positive prompt schedules retain the host path;
no positive cache is introduced. Only static LoRA is in V1 scope. Time-dependent
third-party layer patches and other guidance modifications are not supported.

### Precision and memory boundaries

The code calls existing `wq/wk/wv/gate/wo`, MLP and text-fusion modules directly, so
it does not detach LoRA or materialize a separate full model. It calls the host
attention backend rather than forcing a new kernel.

The negative-text projection has a different matrix shape from the unoptimized
full projection. Real GPU/quantized kernels may therefore have small numerical
or performance differences; CPU equivalence is not a GPU bit-equality guarantee.

A conservative extra-activation estimate is passed to
`add_extra_preserved_memory_during_sampling`. It scales with batch, image-token
count, feature width and text context. It is not a measured peak or an OOM guarantee.
Sigma gating is checked once per denoiser call, not per block. A finite-output check
runs once for each active call; non-finite results are not silently repaired.

## Input, state and persistence contracts

- One independent plain-text NAG negative for the entire batch; no implicit Style,
  LoRA-tag, wildcard, emphasis or schedule expansion.
- UI/API numerical validation is finite and bounded. CFG is required to be exactly
  1.0 for active NAG, and is never silently changed.
- Sigma bounds are inclusive sampling sigmas, before predictor time conversion.
  Mixed in-range/out-of-range rows in one model call are rejected by V1.
- Empty negative, Phi=0 and Alpha=0 bypass NAG. Enabled bypasses are marked explicitly.
- Normal OFF requests leave no stale `Forge NAG ...` metadata.
- Active metadata is added only after a call actually used NAG. Unexecuted or
  never-active wrappers cannot be reported as applied.
- PNG Info without NAG metadata turns the extension OFF rather than keeping a
  previously enabled value. Multiline/Unicode negative text is preserved through
  the host's quoted generation-info format.
- Native hover help modifies `title` attributes only inside the extension section.
  It does not toggle disabled states, edit standard prompts or inject user text as HTML.

## 日本語要約

V1は本体ファイルの書換えやグローバルなforward差替えを行わず、生成要求の範囲内に
処理の接続を限定します。Forgeがフック内の例外を記録だけして続行する問題に対して、
NAGの最終検証をその例外処理の外へ置いています。モデルへの適用回数も確認し、
「NAGが無視された普通の画像」をNAG成功として返さない設計です。

最適化は画像Q/K/Vの共有と生成中限定のNegativeテキスト変換キャッシュです。
Negative側のテキスト状態更新、追加Attention、FP32正規化は省略しません。
推論の入口・出口のSigma変換はForge本来の`KModel.apply_model`を使用します。
実モデルの見た目、FP8動作、速度、VRAMの評価はCPU試験とは別に必要です。
