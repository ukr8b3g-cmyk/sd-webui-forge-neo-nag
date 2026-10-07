# GPU / native UI acceptance checklist

**Not executed in the implementation environment. / 実装環境では未実施です。**

Do not promote this build from CPU-validated to GPU-validated based only on the
unit-test count. Use the user's existing working Forge Neo environment and models.
No dependency updates or new model downloads are necessary for this gate.

## Record the baseline / 基準を固定

Record Forge commit, extension version, checkpoint hash, encoder, LoRA names and
strengths, storage/computation precision, attention backend, sampler, scheduler,
steps, seed, resolution, batch size, wall time, peak VRAM and saved generation info.
Disable Reference/Edit, Hires fix, refiner, CFG++, ControlNet, tiling, token merging
and other guidance patches. Keep the ordinary negative box unused at CFG=1.

日本語：普段動いているKrea2設定をそのまま基準にし、比較ごとにモデルやサンプラーを
同時変更しないでください。最初は小さい解像度・バッチ1で確認し、その後いつもの解像度へ進めます。

## Minimal sequence

| Case | Change from baseline | Acceptance |
|---|---|---|
| A | NAG OFF | Ordinary generation completes; no NAG metadata or encoding. |
| B | Enable ON, same negative, Alpha=0 | Same latent/image as A under a deterministic backend; no extra negative encoding. |
| C | Enable ON, Phi=4 / Tau=2.5 / Alpha=.25 | At least one active NAG call; finite output; unwanted concept can be evaluated visually. |
| D | NAG OFF again | Returns to A's ordinary behavior; no stale wrapper, prompt or NAG cache. |
| E | Repeat C, then interrupt a later run | Reproducible settings; next OFF run succeeds; memory does not grow from retained NAG caches. |

Compare C against A on a few fixed seeds with an easily visible concept. Record both
suppression and collateral changes to composition, identity, color and detail.
Do not use a changed image alone as proof that the requested concept was suppressed.
Do not choose only successful seeds after seeing the result.

日本語：画像が変わっただけでは抑制成功とはしません。指定した対象が弱まったか、
必要な部分まで壊れていないかを別々に評価します。効果の弱さを強度の無制限な引上げで
埋め合わせないでください。

## Additional compatibility checks

Run the same small sequence with the actual FP8/quantization mode and static Turbo
LoRA used in practice. Then test batch size 2 when memory permits. Compare warm-run
time and peak VRAM against OFF; the extra attention need not be free or faster.
No speedup ratio was promised by the CPU tests.

Confirm that CFG=2 with active NAG, Reference/Edit, Hires fix and another guidance
wrapper produce clear errors without a falsely labelled normal image. Confirm that
an empty NAG interval is rejected, while empty text/Phi=0/Alpha=0 is an explicit bypass.

## Actual UI and persistence

Open the extension in the real Forge UI in English and Japanese. Hover all seven
controls. The ordinary negative field must remain greyed out at CFG=1, while the
NAG negative field remains usable. Generate a PNG with a multiline/Unicode NAG
negative, restore it through PNG Info, and compare every NAG parameter. Pasting a
normal image without NAG metadata must disable NAG.

The implementation environment tested the tooltip JavaScript in a Chromium DOM
fixture, not the user's Forge page. A separate live Gradio browser attempt was
blocked by environment navigation policy; that policy was not changed.
