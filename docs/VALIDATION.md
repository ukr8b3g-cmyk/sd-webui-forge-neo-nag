# Validation report / 検証結果

Date: **2026-10-07**

**Implementation and CPU verification complete. Native Forge UI, pretrained-model
GPU generation, FP8 GPU operation, image quality, runtime speed and peak VRAM remain
unverified.** This report records implementation verification before publication;
see the [v0.1.0 release notes](releases/v0.1.0.md) for the publishing scope.

## Source baseline

The target repository was empty when inspected through the connected GitHub API.
The worktree was initialized for `ukr8b3g-cmyk/sd-webui-forge-neo-nag`, branch `main`,
without creating a commit. Container Git cloning could not resolve GitHub; source
inspection used the GitHub connector. Existing remote files were not overwritten.

Forge Neo reviewed commit:
`bd1d015950b62f1c88a6d95f0b9ea1e0dfc69aa3`.

Two retrieved Forge source files were materialized for an external CPU probe and
verified against their Git blob IDs. They are **not bundled** with this extension:

| Source | Verified Git blob |
|---|---|
| `backend/nn/krea.py` | `eb5d997633fe0018e1ed5275865c4b7809f9372f` |
| `backend/modules/k_model.py` | `0f09636d1e9919ef54c1a47c6c034ab08a922cc4` |

## Executed checks

| Check | Result | Scope |
|---|---:|---|
| Python regression suite | **154 passed, 0 failed** | Numerical core, configuration, adapter, lifecycle, restoration, rejection, metadata and UI construction. |
| Native-definition CPU probe | **24 passed, 0 failed** | Actual retrieved Forge class/method definitions, reduced random models, CPU backend substitutes. |
| Hover-help browser fixture | **16 passed, 0 failed** | Production JavaScript in Chromium, English/Japanese help, scope isolation and repeat refresh. |
| JavaScript syntax | **PASS** | `node --check javascript/forge_neo_nag.js`. |
| Python compilation | **PASS** | Runtime, script entry point, tests and probe. |
| Live Gradio browser navigation | **BLOCKED** | Environment browser policy blocked loopback navigation; no policy bypass attempted. |
| User's native Forge UI | **NOT RUN** | No access to the user's running Windows application. |
| Pretrained Krea2 / Qwen3-VL GPU generation | **NOT RUN** | The execution environment has CPU PyTorch and no CUDA device or pretrained weights. |

Environment: Python 3.13, PyTorch `2.10.0+cpu`, available Gradio `6.5.1` for isolated
UI construction. This is not a claim that the user's Forge uses that Gradio version.
No dependencies were installed or updated for this work.

Raw records: [pytest](results/pytest.txt), [native CPU probe](results/native-cpu.json),
[tooltip fixture](results/tooltip-browser.json).

### What the numerical results establish

The L1 NAG equations were compared with the upstream ratio/cap formulation, including
zero/small norms and FP32/FP16/BF16 CPU values. Shared-image Q/K/V blocks were compared
with an unoptimized two-full-projection reference. Different text lengths, grouped
KV heads, batches, odd spatial dimensions, cropping and single-frame 5D latents were exercised.

The native-definition probe checked the actual Forge SingleStreamDiT block and
text-fusion methods under reduced random weights, plus the actual `KModel.apply_model`
method with a nonidentity test predictor. Normalized input, model-time conversion
and output denoising conversion all remained in the native method's path.
The largest recorded absolute discrepancy in the native numerical comparisons was
**5.960464477539063e-08**. This is not an image-level ComfyUI/Forge equivalence result.

The negative text changes the model computation. Its initial fused state is cached
once per sampling session, while per-block negative text continues advancing. Input
conditioning and cached tensors are not mutated. Caches are not carried into the
next request or reused after a weight change between requests.

### What the lifecycle checks establish

OFF requests are bit-identical in the CPU contract harness and do not load NAG host
bindings or encode extra negative text. Alpha=0, Phi=0 and empty text take the normal
path. Failure, interruption and normal completion restore the request runner/Patcher
and release NAG cache references. Shallow-copied processing objects correctly rebind
the native sample method to the new request; foreign callables are not mistakenly unwrapped.

Invalid CFG, unsupported models/modes, references, object/attention/guidance patches,
nonfinite output and sampling paths that never execute NAG cannot return a successful
NAG-labelled result in the tested standard-save contract. Errors in the final NAG
pre-sampling hook escape the host-style exception-swallowing callback loop.

### Remaining acceptance gate

Run [GPU_CHECKLIST.md](GPU_CHECKLIST.md) in the actual Forge environment before calling
this build stable for that environment. In particular, actual Qwen3-VL encoding,
FP8/quantized operators, Turbo LoRA combinations, attention kernels, cancellation
memory behavior, image quality and GPU performance have not been established by
these tests. Hover help on the live Forge screen and native PNG Info interaction
also need real-environment confirmation.

## 日本語の結論

**実装とCPU検証は完了しています。ただし、実GPUと実際のForge画面での合格判定はまだです。**
154件の回帰試験、取得したForgeのクラス／メソッドを使った小型CPU試験24件、
ChromiumのDOMフィクスチャによる日英ヘルプ試験16件が通過しました。
ブラウザーからローカルGradioページを開く試験は環境のポリシーにより阻止され、成功扱いにしていません。

CPUでNAG計算が反映されることと、実モデルで狙った対象を十分に抑制できることは別です。
FP8での動作、画質、速度、VRAM、実UIは未確認です。この報告は公開前の実装検証記録です。
公開内容は[v0.1.0リリース説明](releases/v0.1.0.md)を参照してください。
ユーザー環境への配置・再起動・生成キューへの投入は行っていません。
