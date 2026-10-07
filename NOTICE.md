# License and provenance

Copyright (c) 2026 ukr8b3g-cmyk and contributors.

This Forge Neo extension is distributed under **AGPL-3.0-only**; see [LICENSE](LICENSE).

The attention-guidance equations and the two-text/one-image Krea2 block design
are adapted from **iljung1106/ComfyUI-Krea2-NAG**, copyright (c) 2026 iljung1106,
licensed under MIT. Its complete notice is retained in
[licenses/ComfyUI-Krea2-NAG-MIT.txt](licenses/ComfyUI-Krea2-NAG-MIT.txt).
The Forge integration, lifecycle guard, shared-image projection optimization,
UI, documentation and tests are implemented for this extension.
This is a community port, not an official Krea or Forge release.

References reviewed on 2026-10-07:

- [ComfyUI-Krea2-NAG](https://github.com/iljung1106/ComfyUI-Krea2-NAG)
  - `nag_math.py` blob: `7f91e49165d126cd34d70f8191791dd4d24ee3f8`
  - `krea2_nag.py` blob: `1db6f6044aa2049e59dd37f72ac1dd066e4e0e2f`
  - MIT license blob: `8f703fc946930fb38abc9a3a7b9e5c371f6aac5d`
- [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo),
  reviewed commit `bd1d015950b62f1c88a6d95f0b9ea1e0dfc69aa3`.
  Forge's own source is AGPL-3.0. No Forge source files or pretrained weights
  are bundled in this extension. The optional CPU probe reads the user's
  existing source files without modifying them.
- Chen et al., [Normalized Attention Guidance: Universal Negative Guidance for Diffusion Model](https://arxiv.org/abs/2505.21179).

The root license does not relicense upstream dependencies or model weights.
Their own notices and model terms continue to apply.
