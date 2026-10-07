/* Scoped native hover help. Never changes prompt values or disabled states. */
(() => {
    "use strict";
    const isJapanese = (value) => /^(ja|jp)([-_.]|$)|japanese|日本語/i.test(String(value || ""));
    function refresh() {
        const app = typeof gradioApp === "function" ? gradioApp() : document;
        const root = app.querySelector("#forge_neo_nag_txt2img");
        if (!root) return;
        const metadata = root.querySelector("[data-forge-nag-help]");
        if (!metadata) return;
        let config;
        try {
            config = JSON.parse(metadata.getAttribute("data-forge-nag-help"));
        } catch (_) {
            return;
        }
        const localization = typeof opts !== "undefined" ? opts.localization : undefined;
        const lang = localization === undefined ? config.language : (isJapanese(localization) ? "ja" : "en");
        for (const [id, help] of Object.entries(config.fields || {})) {
            // Metadata contains developer-defined IDs, not prompt text.
            if (!/^forge_neo_nag_[a-z_]+$/.test(id)) continue;
            const block = root.querySelector(`#${id}`);
            if (!block) continue;
            const title = help[lang] || help.en;
            if (typeof title !== "string") continue;
            for (const element of [block, ...block.querySelectorAll("label, input, textarea, button")]) {
                if (element.getAttribute("title") !== title) element.setAttribute("title", title);
            }
        }
    }
    if (typeof onUiLoaded === "function") onUiLoaded(refresh);
    if (typeof onUiUpdate === "function") onUiUpdate(refresh);
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", refresh, {once: true});
    else refresh();
    window.ForgeNeoNAGHelp = Object.freeze({refresh});
})();
