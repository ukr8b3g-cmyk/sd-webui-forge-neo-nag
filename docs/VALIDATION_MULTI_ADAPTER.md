# Multi-adapter validation — 2026-10-08

## Results

- CPU pytest: **300 passed, 0 failed**.
- Syntax compilation: completed for the changed package, script and test files.
- Real Gradio Blocks construction, eight argument/default ordering, optional
  adapter compatibility, paste restoration, bilingual help metadata and exact
  button output targets: passed in pytest.
- Browser interaction probe: **not completed**. Chromium refused navigation to
  the local component server with `ERR_BLOCKED_BY_ADMINISTRATOR`. No screenshot
  or successful real-screen interaction is claimed. The server was closed.
- Real Forge runtime, Windows/user installation, pretrained-model GPU generation,
  FP8/quantized GPU execution, real LoRA combinations, image quality, speed,
  peak VRAM/RAM and GPU interrupt recovery: **not tested**.

## Environment and scope

Linux / Python 3.13.5 / PyTorch 2.10.0+cpu / Gradio 6.5.1. CUDA is unavailable.
No new dependencies or model weights were installed. GitHub supplied the
reviewed code through the connector; network restrictions prevented a full git
clone. Changed original extension files were reconstructed from those responses
and verified against their original Git blob hashes before editing.

Native-model probes use locally transcribed/extracted Forge definitions, loaded
through AST, with reduced random weights and explicit CPU SDPA/RoPE/offload test
operators. This is NOT a complete, byte-identical Forge checkout or an end-to-end
Forge model-loader test. The full-file UNet byte hash was not established as an
upstream match. These probes verify the reviewed method contracts and adapter
routing, not native CUDA operators, real CLIP/Qwen loading or generated images.
The deployed extension does not contain these copied model definitions.

The optional tests can be rerun against an actual local Forge source checkout:

```powershell
$env:FORGE_NAG_TEST_SOURCE = 'D:\path\to\ForgeNeo'
$env:PYTHONPATH = '.'
python -m pytest -q
```

Run inside the extension with its existing test dependencies. Without the source
variable, native-source probes are explicitly skipped, not counted as passes.
Nothing in this command loads checkpoints or queues a GPU generation.

## Coverage

The original Krea2 math/projection/cache/host/boundary suite is retained. The old
CFG=2 rejection case was removed because that contract was explicitly changed;
positive acceptance and branch-preservation tests were added. UI tests now expect
eight fields instead of seven, with the first seven unchanged.

Additional tests cover:

1. Auto model identification independent of all preset values; manual overrides;
   unknown models/Refiner; eighth API argument/default Auto; OFF with stale fields;
   metadata/paste restoration; no callbacks writing preset/CFG/Enabled/selection.
2. CFG values 0, 0.5, 1, 2 and 7.5; reversed [1,0] and [0,1] combined branches;
   batch 1/2; separate branches and unequal text lengths; uncond exactly native in
   deterministic same-size CPU calls; native CFG function retains final ownership.
3. Reduced SDXL UNet (11 cross-attention targets with multi-depth local indexes)
   and Anima DiT (2 blocks), float32/float16/bfloat16 zero-alpha comparisons;
   one output projection; finite nonzero changes; unchanged conditional vectors,
   inputs, weights and shared forward; late model/patch/mask rejection.
4. CLIP-L/G four-chunk boundaries, TI/shape/nonfinite rejection; separate Qwen/T5
   2048-token limits and native preprocessed padding; actual negative flag.
5. Normal/exception/KeyboardInterrupt exits; late Clip skip changes; missing target
   execution; missing/foreign SDXL callbacks; no active Sigma rejection; third-party
   Patcher preservation; OFF -> ON -> OFF output restoration and no cached negative.

## Remaining gates

Use the user's existing model, preset, CFG, sampler, steps, size and LoRA settings.
The extension must not set them for a test. Record them so a diagnostic comparison
can be interpreted. Check each intended pretrained model/precision separately:
NAG OFF/ON/OFF, independent adapter choice at CFG=1 and the user's regular CFG,
standard Negative retained, save/PNG Info restore, cancellation and a following
OFF run. Assess suppression and face/composition/color changes separately.
No image quality/default-strength recommendation or performance claim is made.
