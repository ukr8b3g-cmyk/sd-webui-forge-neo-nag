"""Always-on Forge Neo extension entry point. No changes to Forge source files."""
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import gradio as gr
from modules import scripts, shared

from forge_neo_nag.ui import build_ui
from forge_neo_nag.settings import register_callbacks, resolve_sigma

register_callbacks()


class Script(scripts.Script):
    sorting_priority = 55
    create_group = False

    def __init__(self):
        super().__init__()
        self._ui_context = None
        self._ui_controls = None
        self._ui_infotext_fields = None
        self.on_after_component(self._after_negative_prompt, elem_id="txt2img_neg_prompt_row")

    def title(self):
        # Stable even when the visible control labels are Japanese; API key.
        return "Forge Neo NAG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def _build_page_ui(self, context):
        controls, fields = build_ui(gr, getattr(shared.opts, "localization", "None"), settings_defaults=True)
        self._ui_context = context
        self._ui_controls = controls
        self.infotext_fields = self._ui_infotext_fields = fields
        return controls

    def _after_negative_prompt(self, params):
        if self.is_img2img or params.component.elem_id != "txt2img_neg_prompt_row":
            return
        current = scripts.scripts_current
        if current is None or self not in current.scripts:
            return
        context = gr.context.get_blocks_context()
        if context is None:
            return
        if self._ui_context is context and self._ui_controls is not None:
            return
        self._build_page_ui(context)

    def ui(self, is_img2img):
        context = gr.context.get_blocks_context()
        if (not is_img2img and context is not None
                and scripts.scripts_current is scripts.scripts_txt2img):
            # Reuse only this page's inputs; also covers the old-position fallback.
            if self._ui_context is not context or self._ui_controls is None:
                self._build_page_ui(context)
            self.infotext_fields = self._ui_infotext_fields
            return self._ui_controls

        # Forge's API builds a separate Blocks context and calls ui() twice.
        # Preserve its original defaults and the page's PNG Info bindings.
        controls, _ = build_ui(gr, getattr(shared.opts, "localization", "None"))
        return controls

    def before_process(self, p, enabled=False, negative="", phi=4.0, tau=2.5,
                       alpha=0.25, sigma_start=1000.0, sigma_end=0.0, adapter="auto"):
        from forge_neo_nag.host import arm_request
        sigma_start, sigma_end = resolve_sigma(shared.opts, sigma_start, sigma_end)
        arm_request(p, (enabled, negative, phi, tau, alpha, sigma_start, sigma_end, adapter),
                    preset=getattr(shared.opts, "forge_preset", None))

    def postprocess(self, p, processed, *args):
        from forge_neo_nag.host import disarm_request
        disarm_request(p)
