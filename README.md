# Forge Neo NAG

**Normalized Attention Guidance for Forge Neo. Extensible model adapters; V1 supports Krea2 only.**

[日本語](#日本語) · [English](#english) · [Download / ダウンロード](https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-nag/releases) · [Validation / 検証](docs/VALIDATION.md) · [Implementation](docs/IMPLEMENTATION.md)

> **v0.1.0 — CPU-validated implementation, GPU validation pending.**
> This is a Forge Neo WebUI **extension**, not a ComfyUI custom node.
> Do not interpret passing CPU tests as verified image quality, GPU speed, or FP8 compatibility.

## 日本語

### 何ができるか

CFGを **1.0のまま** にして、Krea2のAttention内部へNAGによるNegative条件を追加します。
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

Krea2が通常生成できているForge Neoを前提にします。追加モデルや追加の実行時依存関係を
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

1. txt2imgでKrea2を選択し、通常のPositive Promptを入力します。CFGを`1.0`にします。
2. `Forge Neo NAG — Krea2`を開き、`Enable NAG / NAGを有効化`をONにします。
3. **NAG Negative Prompt**へ抑制したい対象を入力して生成します。まずは`big wings`や
   `sunglasses`のように、結果を見分けやすい短い対象から比較してください。

サンプラー・ステップ・解像度・モデル・LoRAは、まず手元で動いているKrea2設定を維持してください。
NAG用のLoRAタグや追加エンコーダは不要です。使用中のKrea2用Qwen3-VLエンコーダを再利用します。
LoRAタグは通常のPositive Promptへ指定し、NAG欄へは入れません。

各項目にマウスを置くと説明を表示します。ラベルと補助説明はUI作成時のForgeの言語設定に従い、
日本語設定では日本語、それ以外では英語になります。マウスオーバーヘルプも同じ言語設定に対応します。
言語設定変更後はUIを再読込してください。

| 項目 | 初期値 | 内容 |
|---|---:|---|
| Enable NAG | OFF | 有効化。OFFではNegativeの追加エンコードも行いません。 |
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

実装対象は**Krea2 / txt2img / CFG=1 / 参照画像なし**です。同じNAG Negativeをバッチ内の全画像へ適用します。
通常の静的LoRAとForge既存の量子化Linear層を再利用する構造ですが、**実GPUでのFP8・量子化・LoRAの
組合せ試験は未実施**です。コード対応と実機確認済みを区別してください。

Reference/Edit、img2img、Hires fix、Refiner切替、動画、ControlNet、CFG++、Tiling、
Token merging、torch.compileされたKrea2、他のAttention/Guidanceパッチとの併用はV1では非対応です。
有効なNAG要求で検出した場合、設定を勝手に変えたり、NAGを無視して通常画像を返したりせず停止します。

NAG欄は通常テキスト専用です。重み構文・プロンプトスケジュール・`AND`・`BREAK`・LoRAタグには対応しません。
既知の非対応構文はエラーにし、黙って解釈を変えません。最大32768文字かつテンプレート込み2048トークンです。
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

### Purpose

Keep **CFG at 1.0** and add negative guidance inside Krea2 attention through a dedicated
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

Start with Forge Neo already able to generate Krea2 images normally. Extract this
extension into `extensions/sd-webui-forge-neo-nag/`, or clone the published repository
there using the command above. Restart the WebUI process. Do not place it in
ComfyUI's `custom_nodes`, and do not overwrite Forge's source files.

No additional runtime packages or model weights are installed by this extension.
It uses Forge's existing Krea2 Qwen3-VL text encoder, loaded layers and attention backend.

Select Krea2 in **txt2img**, set **CFG=1.0**, open **Forge Neo NAG — Krea2**, enable NAG,
and enter a short unwanted concept in **NAG Negative Prompt**. Start with the defaults
`Phi=4.0`, `Tau=2.5`, `Alpha=0.25`. Keep your otherwise-working model, LoRA, sampler,
step count and resolution unchanged for the first comparison. Put LoRA tags in the
ordinary positive prompt, not in the NAG field.

All seven controls have explanatory text and mouse-over help. Japanese Forge
localization selects Japanese labels/help; otherwise English is used. Reload the UI
after changing localization. The script's API name and metadata keys remain English.

| Control | Default | Meaning |
|---|---:|---|
| Enable NAG | OFF | OFF does not add negative encoding or model patches. |
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

V1 implements **native Krea2, txt2img, CFG=1, no reference images**. A batch shares one
NAG negative prompt. Static LoRA and native quantized Linear modules are reused,
not replaced. **GPU combinations involving FP8, quantization and LoRA have not yet
been tested.** Structural compatibility is not a claim of verified GPU operation.

Reference/Edit, img2img, Hires fix, refiner switching, video, ControlNet, CFG++,
tiling, token merging, compiled Krea2 and other attention/guidance patches are not
supported by V1. Detected incompatible active requests stop; settings are not silently
changed, and ordinary generation is not silently substituted for requested NAG.

The NAG box accepts plain text only. Known weights, schedules, `AND`, `BREAK` and LoRA
tags are rejected. Limits are 32768 characters and 2048 tokens including the encoder
template. Input is never silently truncated. Styles/wildcards from standard prompts
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

Use your existing working Krea2 API payload, keeping its model/precision/LoRA choices.
The always-on script name is `Forge Neo NAG`, regardless of UI language.
Argument order is **enabled, negative, phi, tau, alpha, sigma_start, sigma_end**.

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
settings that already work with your Krea2 model. For problems, report the Forge
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
