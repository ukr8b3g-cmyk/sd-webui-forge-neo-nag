# Forge Neo NAG

<img width="900" height="440" alt="{9A932AEB-0587-439C-9733-EF51F05D8AAE}" src="https://github.com/user-attachments/assets/a20fc273-58cc-46d1-a7e4-279bcf7f5d5b" />


**Normalized Attention Guidance for Forge Neo. Multi-adapter support for Krea2, Anima, SDXL / Illustrious, Klein / Flux.2, Z-Image Turbo, Ernie Image and Qwen-Image (2512 / base).**

[日本語](#日本語) · [English](#english) · [Download / ダウンロード](https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-nag/releases) · [Validation / 検証](docs/VALIDATION_MULTI_ADAPTER.md) · [Implementation](docs/MULTI_ADAPTER_IMPLEMENTATION.md)

> **v0.5.1 — fixes manual Adapter UI-default saving and reloading in English and Japanese; model features remain unchanged.**
> ZIT includes the native `adaln_input` refiner-time contract fix. Pretrained-model GPU validation for ZIT/Ernie is still pending.
> This is a Forge Neo WebUI **extension**, not a ComfyUI custom node.
> NAG does not rewrite Forge presets, CFG, sampler, scheduler, steps, resolution, Clip skip, LoRA, or the standard Negative Prompt.
>
> **Current v0.5.1 behavior:** Adapter = `Auto / Krea2 / Anima / SDXL (Illustrious) / Klein (Flux.2) / Z-Image Turbo (ZIT) / Ernie Image / Qwen-Image (2512 / base)`. Preset and CFG remain user-controlled. Native Edit-2511 reference generation is not enabled.

## 日本語

### v0.5.1 の要点

- 対応アダプター: **Krea2 / Anima / SDXL (Illustrious) / Klein (Flux.2) / Z-Image Turbo (ZIT) / Ernie Image / Qwen-Image (2512 / base)**。
- アダプターは`Auto`または手動選択。Presetと一致しない場合もユーザーが自分で修正できます。
- **v0.5.1修正**：手動Adapter選択をForgeのUI既定値として保存した後、再起動するとAutoに戻る問題を修正しました。UI既定値は表示名で保存し、API・生成メタデータは従来の内部IDを維持します。
- **CFG=1固定は撤廃**。通常CFGと標準Negative枝はForge Neo本来の処理を維持します。
- Preset、CFG、sampler、scheduler、steps、解像度、Clip skip、LoRAをNAG側から自動変更しません。
- `txt2img / 参照画像なし`が現在の共通対応範囲です。Hires fix、img2img、Refiner、ControlNet、CFG++などは対象外です。
- Qwen-Image（2512／基本系列）専用のJoint AttentionアダプターとQwen2.5-VL 7BによるNAG Negative入力を追加しました。**Qwen-Image-2.1は別モデルとして非対応、Edit-2511の参照・画像編集はPhase B待ち**です。実モデルGPU、FP8、実LoRA、画質、速度、VRAMは未確認です。

共通仕様は[実装契約](docs/MULTI_ADAPTER_IMPLEMENTATION.md)と[検証記録](docs/VALIDATION_MULTI_ADAPTER.md)です。
モデル固有の実装はKlein、Z-Image、Ernie、Qwen-Imageの各文書を参照してください。
[Qwen-Image実装と制限](docs/QWENIMAGE_IMPLEMENTATION.md)・[Qwen-Image CPU検証](docs/VALIDATION_QWENIMAGE.md)をご確認ください。

### 対応・非対応モデル一覧（v0.5.1）

**判定の意味**：「実装対象」は専用Adapterがあるモデル系列、「対応見込み」は同じNative Engine／Transformer／Attention構造なら動作する可能性がある派生モデルです。**個別チェックポイントの実GPU生成、画質、NAGの抑制効果、FP8・LoRA互換性まで検証済みという意味ではありません。** モデル名や手動Adapter選択だけで互換性は確定せず、構造が不一致ならNAGは停止します。

| Adapter | 実装対象（系列） | 対応見込みのモデル・派生（個別GPU未検証） |
|---|---|---|
| **Krea2** | Krea-2-Raw、Krea-2-Turbo | 同一Krea2構造のファインチューニング・派生 |
| **Anima** | Anima Base v1.0、Anima Aesthetic v1.0／v1.1、Anima Turbo v1.0／v1.1 | Anima Preview系など同一構造の派生 |
| **SDXL / Illustrious** | Stable Diffusion XL 1.0、Illustrious XL v1.x／v2.x | NoobAI-XL、Animagine XL 3.1／4.0、Pony Diffusion V6 XL、WAI-Illustrious、Juggernaut XL、RealVisXL、その他通常SDXL互換モデル |
| **Klein / Flux.2** | FLUX.2-klein-4B、FLUX.2-klein-9B | FLUX.2-klein-base-4B／base-9B、同一構造の派生 |
| **Z-Image Turbo (ZIT)** | Z-Image-Turbo | Z-Image（非蒸留Base）、同一構造の派生 |
| **Ernie Image** | ERNIE-Image（8B） | 同じ`ErnieImage`構造の派生 |
| **Qwen-Image (2512 / base)** | 初代Qwen-Image、Qwen-Image-2512 | 同じ`QwenImage`構造のtxt2img派生（参照画像なし） |

**非対応モデル・構造（v0.5.1）**

| モデル・系列 | 理由・状態 |
|---|---|
| **Qwen-Image-2.1** | 別エンジン`QwenImage21`／別Transformer。Adapterなし |
| **Qwen-Image-Edit-2511** | 画像参照・編集、`zero_cond_t`経路は未対応。**実装案は保留中** |
| **FLUX.1-dev／FLUX.1-schnell** | FLUX.2 Kleinとは異なる構造 |
| **FLUX.2-dev** | Klein系とは異なる構造 |
| **Stable Diffusion 1.5／2.x** | SDXL Adapterでは非対応 |
| **Stable Diffusion 3／3.5** | Adapter未実装 |
| **SDXL Refiner専用／Inpaint専用／Rectified-Flow SDXL** | 現行SDXL Adapterの対応範囲外 |
| **Z-Image-Edit／編集用Z-Image系** | 画像編集・参照入力経路は未実装 |
| **Nunchaku版Z-Image** | Nativeモデル型・Attention構造が異なるため非対応 |

**利用条件**：Forge Neoで適合するNativeエンジンとして読み込める**txt2img（参照画像なし）**に限ります。img2img、Hires fix、Reference／Edit、Refiner切替、ControlNet、CFG++、動画、Tiling、Token merging、競合するAttention／Guidanceパッチ、未対応のコンパイル済みモデルは対象外です。NAGはCFG・Sampler・Scheduler・Presetを変更しません。実GPU確認は[GPU確認手順](docs/GPU_CHECKLIST.md)を参照してください。

### 何ができるか

CFGを変更せずに、読み込み中の対応モデルのAttention内部へ独立したNAG Negative条件を追加します。
標準の「Negative Prompt」とは別に、**NAG Negative Prompt専用入力欄**を追加します。
標準Negative欄のグレーアウト、通常CFG、他モデルの生成処理は変更しません。

これは通常のCFG Negativeの再有効化ではありません。同じ画像側Q/K/Vに対し、
PositiveとNAG Negativeのテキスト条件を変えてAttentionを計算し、画像側の結果だけを
L1正規化して混合します。画像の状態は1本のまま、PositiveとNegativeのテキスト状態は
それぞれ各ブロックで更新します。

**抑制は必ず成功するものではありません。** 強度やプロンプトによって構図・色・画質が
変わる場合があります。特に「指を必ず直す」「特定の物体を必ず消す」といった保証はありません。
実モデルの視覚評価は[GPU確認手順](docs/GPU_CHECKLIST.md)で行います。

### インストール

使用するモデルが通常生成できているForge Neoを前提にします。追加モデルや追加の実行時依存関係を
この拡張からインストールすることはありません。

ZIPを展開し、次の配置になるようにForge Neoの`extensions`へ入れてください。
二重フォルダーになっていないことを確認し、WebUIプロセスを再起動します。

```text
Forge Neo/
└─ extensions/
   └─ sd-webui-forge-neo-nag/
      ├─ scripts/forge_neo_nag.py
      ├─ forge_neo_nag/
      └─ javascript/forge_neo_nag.js
```

公開ソースから導入する場合は、Forge Neoの`extensions`内で実行します。

```bash
git clone https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-nag.git
```

ComfyUIの`custom_nodes`には入れません。Forge本体のファイルを上書きする必要もありません。

### 基本操作

1. txt2imgで対応モデルを読み込み、通常のPositive Promptと自分のCFG値を設定します。
2. `Forge Neo NAG`を開き、`Adapter=Auto`または互換性のあるモデルを手動選択します。
3. `Enable NAG / NAGを有効化`をONにして、**NAG Negative Prompt**へ抑制対象を入力します。まずは`big wings`や`sunglasses`などで比較してください。

**Adapter既定値の保存**：v0.5.1では、ForgeのUI既定値管理で選択したAdapterを保存・再読み込みできます。以前のバージョンで内部ID（例：`qwenimage`）をUI設定ファイルへ保存していた場合は、アップデート後にAdapterを選択し直して既定値を再保存してください。NAGは既存のUI設定ファイルを自動変更しません。PNG Info・APIの内部IDは引き続き利用できます。

サンプラー・ステップ・解像度・CFG・モデル・LoRAは、通常生成できる設定を維持してください。
NAG用の追加Text Encoderは不要です。各モデルの既存Text Encoderを再利用します。
LoRAタグは通常のPositive Promptへ指定し、NAG欄へは入れません。

各項目にマウスを置くと説明を表示します。ラベルと補助説明はUI作成時のForgeの言語設定に従い、
日本語設定では日本語、それ以外では英語になります。マウスオーバーヘルプも同じ言語設定に対応します。
言語設定変更後はUIを再読込してください。

| 項目 | 初期値 | 内容 |
|---|---:|---|
| Enable NAG | OFF | 有効化。OFFではNegativeの追加エンコードも行いません。 |
| Adapter | Auto | 読み込まれたEngineからAuto判定。手動選択可。Forge Presetは変更しません。 |
| NAG Negative Prompt | 空欄 | 抑制対象の通常テキスト。標準Negative欄とは独立です。 |
| NAG Scale / Phi | 4.0 | Attention差分を強める係数。0でバイパスします。 |
| Norm Cap / Tau | 2.5 | 正規化時のL1ノルム上限比率です。CFGとは別の値です。 |
| Blend / Alpha | 0.25 | 正規化した結果の混合率。0でバイパスします。 |
| Sigma Start | 1000 | 適用するサンプリングSigmaの上限。境界を含みます。 |
| Sigma End | 0 | 適用するサンプリングSigmaの下限。境界を含みます。 |

Sigmaはステップ数や進捗率ではなく、Forgeがモデル時刻へ変換する前の値です。
`Start >= End`が必要です。範囲外では元のモデル経路を使用し、一度もNAGが適用されなかった生成は
NAG成功として保存せずエラーにします。空欄・Phi=0・Alpha=0は通常経路へバイパスし、
有効化されていた場合はメタデータに`bypassed`と記録します。

### V1の範囲

共通初期対応は**Krea2 / Anima / SDXL / Klein / Z-Image / Ernie / Qwen-Imageのtxt2img（参照画像なし）**です。CFGはユーザーが設定し、通常CFGの合成はForge側で行います。
通常の静的LoRAとForge既存の量子化Linear層を再利用する構造ですが、**実GPUでのFP8・量子化・LoRAの
組合せ試験は未実施**です。コード対応と実機確認済みを区別してください。

Reference/Edit、img2img、Hires fix、Refiner切替、動画、ControlNet、CFG++、Tiling、
Token merging、torch.compileされたKrea2、他のAttention/Guidanceパッチとの併用はV1では非対応です。
有効なNAG要求で検出した場合、設定を勝手に変えたり、NAGを無視して通常画像を返したりせず停止します。

NAG欄は通常テキスト専用です。重み構文・プロンプトスケジュール・`AND`・`BREAK`・LoRAタグには対応しません。
既知の非対応構文はエラーにし、黙って解釈を変えません。最大32768文字です。トークン上限はモデル別です（SDXLはCLIP最大4チャンク、Animaは各Tokenizer最大2048、Krea2/Klein/ZIT/Ernie/Qwen-Imageは最大2048）。
上限を超えても切り捨てずエラーにします。標準プロンプトのStyleやWildcardをNAG欄へ自動適用しません。

### 最適化と再現性

画像側Q/K/Vとゲートの不要な再計算を省き、Negativeの`txtfusion + txtmlp`を1回のサンプリング中だけ再利用します。
キャッシュは次の生成へ持ち越しません。各ブロックのNegativeテキスト状態更新は省略しません。
NAGの正規化はFP32、結果は計算用dtypeへ戻します。モデル全体のFP32化・重みの複製・新しいAttention実装への強制切替は行いません。

追加Attentionは必要なので、**NAG OFFより必ず速い、VRAMが増えない、という意味ではありません。**
実速度とメモリ増加量は未測定です。追加アクティベーションの概算をForgeのメモリ管理へ予約しますが、OOM防止の保証ではありません。

画像の生成情報へNAG Negative、全パラメータ、バージョン、実際の適用回数を保存します。
PNG Infoからの復元に対応し、NAG情報がない画像を読み込んだ場合はNAGをOFFへ戻します。
設定の意味が標準Negativeと混ざらないよう、`Forge NAG ...`という独立キーを使用します。

### API・問題が起きた場合

[API例](#api-example)のalways-on script名は、表示言語に関係なく`Forge Neo NAG`です。
NAGをOFFにすると通常経路へ戻ります。問題報告にはForgeのコミット、モデルと精度、LoRA、Attention backend、
解像度、バッチサイズ、生成情報、コンソールのエラーを添えてください。
この拡張は既存モデルの`forward`をグローバルに書き換えず、正常終了・例外・中断で生成用の接続とキャッシュを解放します。

---

## English

### v0.5.1 summary

- Adapters: **Krea2 / Anima / SDXL (Illustrious) / Klein (Flux.2) / Z-Image Turbo (ZIT) / Ernie Image / Qwen-Image (2512 / base)**.
- Adapter selection can be `Auto` or manual. **v0.5.1 fixes manual Adapter persistence through Forge UI defaults in English and Japanese** while leaving API and image metadata IDs unchanged.
- **CFG=1 is no longer required**; Forge's native standard Negative branch and CFG composition are preserved.
- NAG does not change presets, CFG, sampler, scheduler, steps, resolution, Clip skip or LoRA.
- Current common scope: `txt2img`, no reference images. Hires fix, img2img, Refiner, ControlNet and CFG++ remain unsupported.
- Qwen-Image (including 2512) adds a native Qwen2.5-VL text-encoder NAG path with shared image Q/K/V. Qwen-Image-2.1 and Edit-2511 reference/image editing are not supported in this adapter. Pretrained-model GPU behavior, FP8, real LoRA combinations, image quality, speed and VRAM remain unverified.

The following instructions describe all seven adapters; model-specific details are documented separately. The common contract is
[Multi-adapter implementation](docs/MULTI_ADAPTER_IMPLEMENTATION.md), with model-specific details in the Klein, Z-Image and Ernie implementation documents under `docs/`.

### Supported and unsupported models (v0.5.1)

**Interpretation:** An “implemented family” has a dedicated Adapter. “Potential candidates” are same-family variants that *may* work only if Forge Neo loads a compatible native Engine, Transformer and Attention layout. **This is not per-checkpoint GPU, image-quality, NAG effectiveness, FP8 or LoRA validation.** Checkpoint names and manual selection do not bypass structural checks; incompatible models fail closed.

| Adapter | Implemented model family | Potential candidates (individual GPU validation pending) |
|---|---|---|
| **Krea2** | Krea-2-Raw; Krea-2-Turbo | Fine-tunes retaining the native Krea2 layout |
| **Anima** | Anima Base v1.0; Anima Aesthetic v1.0/v1.1; Anima Turbo v1.0/v1.1 | Anima Preview and other same-layout variants |
| **SDXL / Illustrious** | Stable Diffusion XL 1.0; Illustrious XL v1.x/v2.x | NoobAI-XL; Animagine XL 3.1/4.0; Pony Diffusion V6 XL; WAI-Illustrious; Juggernaut XL; RealVisXL; other standard SDXL models |
| **Klein / Flux.2** | FLUX.2-klein-4B; FLUX.2-klein-9B | FLUX.2-klein-base-4B/base-9B; compatible variants |
| **Z-Image Turbo (ZIT)** | Z-Image-Turbo | Z-Image (undistilled base); compatible variants |
| **Ernie Image** | ERNIE-Image (8B) | Variants retaining the native `ErnieImage` layout |
| **Qwen-Image (2512 / base)** | Original Qwen-Image; Qwen-Image-2512 | Same-layout Qwen-Image txt2img variants without references |

**Unsupported models and architectures in v0.5.1**

| Model/family | Reason/status |
|---|---|
| **Qwen-Image-2.1** | Separate `QwenImage21` engine and Transformer; no Adapter |
| **Qwen-Image-Edit-2511** | Reference/edit and `zero_cond_t` path unsupported; **development on hold** |
| **FLUX.1-dev / FLUX.1-schnell** | Not the FLUX.2 Klein architecture |
| **FLUX.2-dev** | Not the Klein architecture |
| **Stable Diffusion 1.5 / 2.x** | Not supported by the SDXL Adapter |
| **Stable Diffusion 3 / 3.5** | No Adapter |
| **SDXL Refiner / inpainting / rectified-flow SDXL** | Outside the supported SDXL model/layout contract |
| **Z-Image-Edit / image-editing Z-Image variants** | Reference/edit pipeline unsupported |
| **Nunchaku Z-Image** | Different native model and Attention layout |

**Common requirements:** Native Forge-compatible **txt2img without references**. No img2img, Hires fix, Reference/Edit, refiner switches, ControlNet, CFG++, video, tiling, token merging, conflicting Attention/Guidance patches or unsupported compiled models. User CFG, sampler, scheduler and preset remain unchanged. See the [GPU checklist](docs/GPU_CHECKLIST.md).

### Purpose

Keep **your chosen CFG value** and add guidance inside supported models' attention through a separate
**NAG Negative Prompt** box. The ordinary negative prompt, its greyed-out state,
standard CFG, and ordinary generation with NAG disabled remain unchanged.

This is attention-space NAG, not a forced unconditional model pass or a renamed CFG
formula. Both attention branches use the same image Q/K/V; only their text context
differs. Image attention is extrapolated, L1-normalized and blended. One shared image
state and two independently advancing text states continue through the transformer.

**Concept suppression is not guaranteed.** Prompt choice and strength can also change
composition, colors or fidelity. CPU correctness tests do not establish visual quality
with pretrained models. Follow the [GPU checklist](docs/GPU_CHECKLIST.md) before treating
a particular setup as validated.

### Installation and use

Start with Forge Neo already able to generate images with the selected supported model. Adapter selections saved in Forge UI defaults now persist after restart. If an older UI settings file stored internal IDs (e.g. `qwenimage`) instead of display labels, select the adapter again and re-save UI defaults once after upgrading. The extension does not rewrite that file; PNG Info and API internal IDs remain compatible. Extract this
extension into `extensions/sd-webui-forge-neo-nag/`, or clone the published repository
there using the command above. Restart the WebUI process. Do not place it in
ComfyUI's `custom_nodes`, and do not overwrite Forge's source files.

No additional runtime packages or model weights are installed by this extension.
It reuses the loaded model's existing text encoder, layers and native attention backend.

Select a supported model in **txt2img**, keep your chosen CFG, open **Forge Neo NAG**,
select **Auto** or a compatible adapter, enable NAG and enter an unwanted concept in **NAG Negative Prompt**. Start with the defaults
`Phi=4.0`, `Tau=2.5`, `Alpha=0.25`. Keep your otherwise-working model, LoRA, sampler,
step count and resolution unchanged for the first comparison. Put LoRA tags in the
ordinary positive prompt, not in the NAG field.

All eight controls have explanatory text and mouse-over help. Japanese Forge
localization selects Japanese labels/help; otherwise English is used. Reload the UI
after changing localization. The script's API name and metadata keys remain English.

| Control | Default | Meaning |
|---|---:|---|
| Enable NAG | OFF | OFF does not add negative encoding or model patches. |
| Adapter | Auto | Uses the loaded engine; manual override never changes Forge presets. |
| NAG Negative Prompt | Empty | Plain text describing what to suppress; separate from the standard box. |
| Phi | 4.0 | Attention extrapolation strength; zero bypasses NAG. |
| Tau | 2.5 | Relative L1 norm cap before blending, not CFG. |
| Alpha | 0.25 | Guided-attention blend fraction; zero bypasses NAG. |
| Sigma Start | 1000 | Inclusive upper **sampling sigma**, before conversion to model time. |
| Sigma End | 0 | Inclusive lower sampling sigma. |

Start must be at least End. Sigma is not a step count or percentage. Out-of-range
calls use the original model. A run with zero active NAG calls is rejected rather
than saved as an NAG result. Empty text, Phi=0 or Alpha=0 bypasses the normal NAG
path; enabled-but-bypassed requests are explicitly recorded as `bypassed`.

### V1 support and limits

Initial common support covers **Krea2 / Anima / SDXL / Klein / Z-Image / Ernie, txt2img, no reference images**, with user-controlled CFG. A batch shares one
NAG negative prompt. Static LoRA and native quantized Linear modules are reused,
not replaced. **GPU combinations involving FP8, quantization and LoRA have not yet
been tested.** Structural compatibility is not a claim of verified GPU operation.

Reference/Edit, img2img, Hires fix, refiner switching, video, ControlNet, CFG++,
tiling, token merging, compiled Krea2 and other attention/guidance patches are not
supported by V1. Detected incompatible active requests stop; settings are not silently
changed, and ordinary generation is not silently substituted for requested NAG.

The NAG box accepts plain text only. Known weights, schedules, `AND`, `BREAK` and LoRA
tags are rejected. The limit is 32768 characters, with model-specific token limits (SDXL: up to four CLIP chunks; Anima: 2048 per tokenizer; Krea2/Klein/ZIT/Ernie/Qwen-Image: up to 2048). Input is never silently truncated. Styles/wildcards from standard prompts
are not automatically applied to this independent field.

### Optimization, state and metadata

Image Q/K/V projections and image gating are not redundantly recomputed for the
negative branch. Negative `txtfusion + txtmlp` is cached only within one sampling
run. Per-block negative text updates still execute. Caches are not reused across
requests or LoRA changes. Normalization uses FP32 and returns to the original
computation dtype. Model weights are neither duplicated nor globally converted.

Additional attention work remains: **no claim is made that NAG is faster than OFF,
uses no extra VRAM, or cannot run out of memory**. An additional-activation estimate
is reserved through Forge's memory manager. GPU time and peak memory need measurement.

NAG text, parameters, extension version, implementation identifier and actual active
model-call count are stored as independent `Forge NAG ...` generation-info fields.
PNG Info restoration is supported. Pasting an image without NAG metadata resets NAG
to OFF instead of retaining a stale enabled state.

The shared model's global `forward` is not monkey-patched. The request's sampling
connection, Patcher options and temporary caches are cleaned up on normal return,
errors and interruption. See [implementation details](docs/IMPLEMENTATION.md).

### API example

Use your existing working API payload for a supported model, keeping its model/precision/LoRA choices.
The always-on script name is `Forge Neo NAG`, regardless of UI language.
Argument order is **enabled, negative, phi, tau, alpha, sigma_start, sigma_end**, followed by an optional eighth adapter ID (default `auto`).

```json
{
  "prompt": "A small bird flying over a green field",
  "negative_prompt": "",
  "cfg_scale": 1.0,
  "seed": 123456,
  "steps": 8,
  "width": 512,
  "height": 512,
  "sampler_name": "Euler",
  "scheduler": "Simple",
  "alwayson_scripts": {
    "Forge Neo NAG": {
      "args": [true, "large wings", 4.0, 2.5, 0.25, 1000.0, 0.0]
    }
  }
}
```

The illustrative sampling settings are not a performance recommendation; retain the
settings that already work with your selected supported model. For problems, report the Forge
commit, model/precision, LoRA, attention backend, resolution, batch size, generation
metadata and full console error. Disable NAG to return to ordinary generation.

### Tests and provenance

```bash
python -m pytest -q
node --check javascript/forge_neo_nag.js
```

Tests require an existing development environment with pytest, PyTorch, einops,
Pillow and Gradio; the extension does not install them. The optional
[CPU probe](tools/native_cpu_probe.py) reads local Forge source files, instantiates
reduced random CPU models and verifies the native method boundaries. It does not
load pretrained weights or replace a GPU test.

[ComfyUI-Krea2-NAG](https://github.com/iljung1106/ComfyUI-Krea2-NAG) is the reference
implementation for the port. [The NAG paper](https://arxiv.org/abs/2505.21179) describes
the method. Forge host integration and UI are implemented for this extension.
See [NOTICE](NOTICE.md) for source identifiers and credits and [LICENSE](LICENSE)
for AGPL-3.0-only terms. The upstream NAG MIT notice is preserved.
