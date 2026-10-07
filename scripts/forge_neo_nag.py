"""Always-on Forge Neo extension entry point. No changes to Forge source files."""
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

import gradio as gr
from modules import scripts, shared

from forge_neo_nag.ui import build_ui


class Script(scripts.Script):
    sorting_priority = 55

    def title(self):
        # Stable even when the visible control labels are Japanese; API key.
        return "Forge Neo NAG"

    def show(self, is_img2img):
        return scripts.AlwaysVisible if not is_img2img else False

    def ui(self, is_img2img):
        controls, self.infotext_fields = build_ui(gr, getattr(shared.opts, "localization", "None"))
        return controls

    def before_process(self, p, enabled=False, negative="", phi=4.0, tau=2.5,
                       alpha=0.25, sigma_start=1000.0, sigma_end=0.0):
        from forge_neo_nag.host import arm_request
        arm_request(p, (enabled, negative, phi, tau, alpha, sigma_start, sigma_end))

    def postprocess(self, p, processed, *args):
        from forge_neo_nag.host import disarm_request
        disarm_request(p)
