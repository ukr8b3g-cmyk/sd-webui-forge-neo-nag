# Forge Neo NAG — Multi-adapter development update

2026-10-08 / version `0.2.0-dev`

対象: https://github.com/ukr8b3g-cmyk/sd-webui-forge-neo-nag
基準: `9453000fd3799e81b86e1af584c3d2f807071bcf`

## 今回の変更

Anima と SDXL / Illustrious のアダプターを追加しました。新しい「アダプター」欄は
`Auto / Krea2 / Anima / SDXL / Illustrious` の4選択肢です（最後は1つのSDXL選択肢）。
Auto は実モデルから判定します。手動選択は常に優先され、Preset変更やNAGのON/OFFでは
上書きしません。構造が合わないモデルに別アダプターを強制接続することはせず、
選択を変更するかNAGをOFFにするようエラーで案内します。

CFG=1という制限は撤廃しました。CFG・Preset・sampler・scheduler・steps・解像度・
Clip skip・LoRA・標準Negativeの操作はForge標準とユーザーに任せます。
NAGがそれらを揃えたり、Presetを変更したりする処理はありません。
通常CFGのPositive枝だけをNAGで補正し、標準Negative枝と最終CFG合成は維持します。
CFG=1ではForge本来の負枝省略を維持し、CFG=0では通常の式によりPositive側の寄与は0になります。
CFGが大きい場合、NAGの変化も最終CFGで強まるため、効果や画質は別途確認してください。

本更新は従来の0.1設計書にある「CFGが厳密に1.0」「手動アダプター欄なし」を上書きします。
旧README・旧検証結果・旧Release文書の「V1 / 0.1.0」は旧版の記録です。
今回の内容は本書と `docs/MULTI_ADAPTER_IMPLEMENTATION.md` を参照してください。

## 操作とAPI

NAGの7項目は従来の位置と意味を維持します。UI上でアダプターを手動変更できます。
「SDXL試験設定を適用」ボタンはPhi=2、Tau=2.5、Alpha=0.25、Sigma Start=1000、End=0だけを
変更し、Enabled・アダプター・Negative本文・CFGには触れません。画質の推奨値ではありません。

APIの8番目の引数は省略可能なアダプターIDです。従来の7引数はAutoとして動作します。

```json
{"alwayson_scripts":{"Forge Neo NAG":{"args":[true,"glasses",2.0,2.5,0.25,1000.0,0.0,"sdxl"]}}}
```

IDは `auto`, `krea2`, `anima`, `sdxl`。モデルをロードする引数ではありません。
通常の生成要求のmodel・prompt・CFG等は既存のユーザー設定を使ってください。
PNG Infoでは選択値と実適用アダプターを別々に記録します。
旧画像に選択値がなければAuto、NAG情報がなければOFFへ復元します。

## 残る対応範囲

txt2img・参照画像なし。静的LoRA適用済みのLinearと既存Attentionを再利用します。
Hires fix、img2img、Refiner、ControlNet、IP-Adapter、CFG++、Tiling、Token merging、
領域/Positive AND合成、動的LoRA、追加Attention/Guidanceパッチ、torch.compileは対象外です。
SD1.5、SDXL Refiner、Rectified-Flow SDXLへの対応は追加していません。

専用NAG Negative欄は静的な文章・タグ列です。重み、スケジュール、AND、BREAK、LoRAタグには
対応しません。SDXL/Animaでは暗黙強調も拒否します。タグ内の括弧は `\(` `\)` として入力します。
SDXLはCLIP-L/Gそれぞれ4チャンク/308位置まで、AnimaはQwen/T5の双方2048トークンまで、
Krea2は既存のテンプレート込み2048まで。Textual InversionはSDXLのNAG欄では拒否します。
上限を超えた内容を黙って切り詰めません。

## 確認状態

CPU自動試験300件成功。小型ランダムモデル、Forge定義の抜粋を使う接続試験、
通常CFG、OFF復元、境界、API、実Gradioの構成/コールバック定義を検査しました。
完全なForgeのロード・学習済みモデル・GPU生成を実行した結果ではありません。

ブラウザーでの独立コンポーネント確認は、この環境のブラウザーポリシーで
ローカルページへの移動が拒否され、未完了です。お使いのForge実画面・Windows環境への反映、
FP8/GPU・実モデルLoRA・抑制効果・速度/VRAMは未確認です。詳細は
`docs/VALIDATION_MULTI_ADAPTER.md` と `docs/results/multi-adapter/` を参照してください。

追加の実行時依存関係・モデル・Forge本体変更はありません。
GitHubへのコミット・プッシュ・Release公開は行っていません。
