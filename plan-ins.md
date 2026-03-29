# DEIMv2 インスタンスセグメンテーション拡張

## 実装方針
- 既存の DEIMv2 検出器の上に、query-based の mask branch を追加する。
- shared mask feature は `HybridEncoder` の最高解像度出力を使う。
- 各 query 用の mask embedding は、最終 decoder query state から生成する。
- matching は box/class ベースのまま維持し、mask loss は match した query のうち body に対応するものにのみ適用する。
- 40 クラス検出は維持し、instance segmentation の教師と評価は `classid=0` の body だけを対象にする。
- `category_id=0` の bbox に `segmentation` が無い場合は、空マスクではなく未ラベルとして扱う。

## 主なコード変更
- `engine/deim/deim_decoder.py`
  - 最終 decoder query と shared mask feature から `pred_masks` を生成する処理を追加した。
- `engine/deim/deim_criterion.py`
  - sigmoid BCE と soft Dice による mask supervision を追加した。
  - `mask_valid=True` の body target にのみ mask loss を適用するようにした。
- `engine/deim/postprocessor.py`
  - mask 対応の後処理と、元画像サイズへのリサイズ処理を追加した。
- `engine/data/dataset/coco_dataset.py`
  - body 以外の annotation に `segmentation` が無くても読み込めるようにした。
  - `mask_valid` を target に追加し、body かつ有効 mask を持つ annotation のみ `True` にした。
  - `segm_eval_valid` を target に追加し、segm 評価対象に含める annotation のみ `True` にした。
- `engine/data/dataset/coco_eval.py`
  - `segm` 評価を `classid=0` のみに制限した。
  - validation では mask 付き body annotation のみを含む `segm` 用 GT を別に構築するようにした。
- `engine/data/dataloader.py`
  - batch multi-scale collation 時の mask resize 経路を実装した。
- `engine/solver/det_solver.py`
  - segmentation AP が存在する場合は、それを優先して best checkpoint を保存するようにした。
  - segmentation の eval artifact 保存と評価表示を追加した。
  - ここで使う segm 指標は body-only 評価である。

## 追加した設定
- `configs/dataset/wholebody40_instance.yml`
  - `train_dataloader.dataset.ann_file` は `train_ins.json` を参照し、body mask supervision を学習に載せる。
  - validation では `ann_file: val.json` を bbox GT に使い、`segm_ann_file: val_ins.json` を segm GT に使う。
  - `return_masks: True` と `['bbox', 'segm']` の評価を有効にする。
  - `mask_category_ids: [0]` を追加し、body-only mask supervision / segm evaluation を指定する。
  - `segm_eval_category_ids: [0]` と `segm_ignore_missing_masks: True` を追加し、sparse body mask validation を許容する。
- `configs/deimv2/deimv2_dinov3_x_wholebody40_ins.yml`
  - DINOv3-X + wholebody40 向けの instance segmentation 学習設定。
  - batch size 1、低めの学習率、MixUp 無効、CopyBlend 無効、Mosaic 無効を前提にする。
  - `DEIMTransformer.mask_feature_level` で、shared mask feature に使う `HybridEncoder.outs` の解像度を切り替えられる。
 - `configs/deimv2/deimv2_dinov3_s_wholebody40_ins.yml`
 - `configs/deimv2/deimv2_hgnetv2_n_wholebody40_ins.yml`
 - `configs/deimv2/deimv2_hgnetv2_pico_wholebody40_ins.yml`
 - `configs/deimv2/deimv2_hgnetv2_femto_wholebody40_ins.yml`
 - `configs/deimv2/deimv2_hgnetv2_atto_wholebody40_ins.yml`
  - wholebody40 instance segmentation の対応済み model family として追加した。
  - dataset / evaluator / sparse body mask 方針は `wholebody40_instance.yml` を共通利用する。
  - DINOv3-S と HGNetv2-N は `['mal', 'boxes', 'local', 'masks']`、HGNetv2 pico/femto/atto は `['mal', 'boxes', 'masks']` を使う。

## この設定で学習したモデルの入出力
- 対象は `configs/deimv2/deimv2_dinov3_x_wholebody40_ins.yml` で学習した DEIMv2 instance segmentation モデル。
- 検出クラス数は 40 クラス想定のまま維持し、mask supervision と `segm` 評価だけを body (`classid=0`) に限定する。

### shared mask feature 解像度切替
- shared mask feature は `DEIMTransformer.mask_feature_level` で選択する。
- この値は `HybridEncoder.forward()` が返す `outs` の index を意味する。
- `HybridEncoder.outs` は高解像度から低解像度の順で並ぶ。
- `configs/deimv2/deimv2_dinov3_x_wholebody40_ins.yml` では次の対応になる。
  - `0`: `outs[0]`、stride 8、既定値
  - `1`: `outs[1]`、stride 16
  - `2`: `outs[2]`、stride 32
- `mask_feature_level` を指定しない場合は `0` が使われ、現行実装と同じく最高解像度 feature が shared mask feature になる。
- 高解像度を選ぶほど mask の空間精度は上がりやすいが、一般に feature map が大きくなるため VRAM と計算量は増えやすい。
- 低解像度を選ぶほど VRAM と計算量は抑えやすいが、細かい輪郭表現は落ちやすい。

### 学習時の入力
- モデル入力画像は通常の DEIMv2 と同じく、`train_dataloader` / `val_dataloader` の transform を通した `Tensor[C, H, W]` を batch 化したもの。
- target は 1 画像ごとに以下の辞書を持つ。
  - `boxes`: `Tensor[N, 4]`
    - 正規化済みの `cx, cy, w, h`。
  - `labels`: `Tensor[N]`
    - 40 クラスの class id。
  - `masks`: `Tensor[N, Hm, Wm]`
    - annotation から生成した instance mask。
    - body 以外、および body でも `segmentation` 未提供の annotation には空 mask が入る。
  - `mask_valid`: `Tensor[N]` の bool
    - body かつ有効 `segmentation` を持つ annotation のみ `True`。
    - mask loss の計算対象を示す。
  - `segm_eval_valid`: `Tensor[N]` の bool
    - `segm` 評価対象に含める annotation のみ `True`。
    - 現状は body かつ有効 `segmentation` を持つ annotation のみ `True`。
  - `area`, `iscrowd`, `image_id`, `orig_size`
- body bbox に `segmentation` が無い場合、その annotation は bbox/class 学習には使われるが、mask loss には使われない。

### 学習時のモデル出力
- 学習中の main output は少なくとも以下を持つ。
  - `pred_logits`: `Tensor[B, Q, C]`
    - 各 query の class logit。
  - `pred_boxes`: `Tensor[B, Q, 4]`
    - 各 query の正規化 bbox (`cx, cy, w, h`)。
  - `pred_masks`: `Tensor[B, Q, Hmask, Wmask]`
    - 各 query の mask logit。
    - `Hmask, Wmask` は `mask_feature_level` で選んだ shared mask feature の解像度で、元画像サイズそのものではない。
- `aux_outputs` / `enc_aux_outputs` / `dn_outputs` / `pre_outputs` は box/class 系の補助出力を持つが、現状 `pred_masks` は main output のみ。
- criterion は `pred_boxes` / `pred_logits` で 40 クラス検出を学習し、`pred_masks` には `mask_valid=True` の matched target のみで BCE + Dice を掛ける。

### 推論時の後処理前出力
- モデル本体の生出力は学習時と同じく `pred_logits`, `pred_boxes`, `pred_masks`。
- `pred_masks` は logit のままで返り、閾値化は後処理側で行う。
- body 以外の class に対する `pred_masks` もテンソルとしては出るが、学習保証外であり品質は保証しない。

### 推論時の後処理後出力
- `PostProcessor` 後の 1 画像あたりの出力は以下。
  - `labels`: `Tensor[K]`
    - top-k 検出結果の class id。
  - `scores`: `Tensor[K]`
    - 各検出の score。
  - `boxes`: `Tensor[K, 4]`
    - 元画像座標系の `x1, y1, x2, y2`。
  - `masks`: `Tensor[K, 1, H, W]`
    - 元画像サイズへ resize 済みの mask。
    - evaluator 互換の形状。
- `bbox` 推論結果は全 40 クラスに対して有効。
- `segm` として意味を持つのは `labels == 0` の body prediction のみ。

### 評価指標との対応
- `bbox` AP:
  - validation の `val.json` を使い、全 40 クラス、全 annotation を対象に計算する。
- `segm` AP:
  - validation の `val_ins.json` を使い、`classid=0` のうち、mask 付き body annotation の subset のみを対象に計算する。
- したがって、この設定で学習したモデルは「40 クラス検出モデル + body-only instance segmentation モデル」と解釈するのが正しい。

### `train_ins.json` / `val_ins.json` の使い分け
- instance segmentation 学習では、学習用 annotation は `train_ins.json` を使う。
- validation は GT を分離する。
  - `bbox` 評価: `val.json`
  - `segm` 評価: `val_ins.json`
- この分離により、`val_ins.json` に含まれる non-body / unmatched body の `area: 0` が `bbox` の area-based 指標を汚染しない。
- `val_ins.json` は body mask 付き subset の segm 評価専用 GT として扱う。

## 前提条件
- body の正解 instance mask は COCO polygon 形式で用意されている前提とする。
- body 以外の class の annotation には `segmentation` が無くてもよい。
- body の bbox annotation に `segmentation` が無い場合も正常とし、その annotation は未ラベル扱いとする。
- bbox から作った矩形の疑似 mask は教師として使わない。
- export や deployment の変更は今回の対象外とする。
- `wholebody40` では `classid=0` が body である前提とする。

## アノテーションサンプル
- `CocoDetection` は `images`, `annotations`, `categories` を持つ COCO JSON を前提にする。
- body-only 学習では、body annotation にのみ `segmentation` が必要で、body 以外は省略できる。
- sparse mask 学習では、body annotation であっても `segmentation` が無いケースを許容する。
- 現状実装で最も安全なのは、body の `segmentation` を polygon 配列で持つ形式。
- 下の例は、body class を `category_id: 0` とし、人物の全身を複数輪郭で表現するパターン。
- `iscrowd: 0` の annotation が通常学習対象になる。

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "000001.jpg",
      "width": 1280,
      "height": 720
    }
  ],
  "annotations": [
    {
      "id": 1001,
      "image_id": 1,
      "category_id": 0,
      "bbox": [468.0, 96.0, 202.0, 518.0],
      "area": 58340.0,
      "segmentation": [
        [
          546.0, 96.0,
          582.0, 104.0,
          604.0, 132.0,
          610.0, 174.0,
          598.0, 206.0,
          626.0, 248.0,
          642.0, 308.0,
          632.0, 372.0,
          606.0, 406.0,
          586.0, 386.0,
          580.0, 320.0,
          564.0, 268.0,
          544.0, 252.0,
          520.0, 266.0,
          504.0, 316.0,
          500.0, 384.0,
          482.0, 402.0,
          470.0, 362.0,
          478.0, 294.0,
          492.0, 238.0,
          520.0, 202.0,
          510.0, 170.0,
          516.0, 132.0
        ],
        [
          514.0, 404.0,
          548.0, 400.0,
          560.0, 450.0,
          548.0, 540.0,
          526.0, 614.0,
          498.0, 610.0,
          492.0, 544.0,
          500.0, 468.0
        ],
        [
          566.0, 402.0,
          602.0, 406.0,
          626.0, 470.0,
          642.0, 550.0,
          670.0, 612.0,
          646.0, 614.0,
          612.0, 572.0,
          590.0, 506.0,
          576.0, 450.0
        ]
      ],
      "iscrowd": 0
    },
    {
      "id": 1002,
      "image_id": 1,
      "category_id": 0,
      "bbox": [702.0, 118.0, 184.0, 498.0],
      "area": 52218.0,
      "iscrowd": 0
    },
    {
      "id": 1003,
      "image_id": 1,
      "category_id": 7,
      "bbox": [820.0, 240.0, 96.0, 132.0],
      "area": 12672.0,
      "iscrowd": 0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "body"
    },
    {
      "id": 7,
      "name": "head"
    }
  ]
}
```

- 学習時に最低限必要なのは `image_id`, `category_id`, `bbox`, `area`, `iscrowd`。
- `segmentation` は body annotation に付いていれば mask 学習・segm 評価に使われる。
- body annotation に `segmentation` が無い場合、その bbox は検出学習には使われるが、mask loss と segm 評価からは除外される。
- body 以外では `segmentation` を省略できる。
- `bbox` は COCO 標準の `[x, y, width, height]`。
- body annotation の `segmentation` は、物体輪郭ごとの polygon をフラットな座標列で持つ。
- 複数輪郭の body は、上の例のように「上半身」「左脚」「右脚」のような分離した polygon 群として表現できる。
- `category_id` は `categories[].id` と一致させる。
- `mask_valid` はコード側で自動生成される内部 target であり、annotation JSON に直接書く必要はない。
- `segm_eval_valid` もコード側で自動生成される内部 target であり、annotation JSON に直接書く必要はない。
- `wholebody40` 用に使う場合も JSON 構造は同じで、`category_id` と `categories` だけ対象クラス集合に合わせて置き換える。
- 別データセットから body mask をマージする場合は、bbox annotation を基準に保持し、対応付けできた body annotation にだけ `segmentation` を付与する想定とする。

## 検証範囲
- 今回は静的なコード整合性確認までを対象にする。
- VRAM 制約があるため、学習や評価の end-to-end 実行は十分な GPU 環境が確保できてから行う。

## body-only mask supervision の注意点
- 40 クラス全体の bbox detection は維持される。
- mask 教師と segm 評価は `classid=0` の body のみを対象にする。
- body 以外の class に対する `pred_masks` は学習保証外であり、評価対象外とする。
- body が存在しない画像や batch では、mask loss は 0 になり、detection loss のみで学習が継続する。
- body bbox が存在しても `segmentation` が 1 件も無い画像や batch では、その body bbox は未ラベル扱いとなり、mask loss は 0 のまま検出学習だけが進む。
- `bbox` AP は全 annotation で計算し、`segm` AP は mask 付き body annotation の subset 上で計算する。

## 補足: auxiliary / denoising / encoder auxiliary への mask loss 拡張
- 現状実装では、mask loss は最終 main output の matched query のみに適用している。
- 将来的に `auxiliary`、`denoising`、`encoder auxiliary` にも mask loss を広げることは可能だが、それぞれ影響が異なる。

### auxiliary decoder outputs に mask loss を入れる場合
- 各 decoder layer の中間 query にも mask supervision が入るため、mask の初期収束が速くなる可能性がある。
- 最終層だけに mask 学習を集中させる構造よりも、勾配が各層に分散しやすくなる。
- 一方で、各層分の `pred_masks` を保持して backward する必要があるため、VRAM 使用量と計算量は増える。
- 導入する場合は、main mask loss より小さめの重みで auxiliary mask loss を与える前提が安全。

### denoising outputs に mask loss を入れる場合
- ノイズ付き query に対しても mask supervision を与えることで、mask head の頑健性が上がる可能性がある。
- ただし denoising branch は本来 box/class の学習安定化が主目的なので、mask まで強く縛ると学習干渉を起こす恐れがある。
- 特に粗い query やノイズの強い query に対して mask loss を強く掛けると、box/class の収束を悪化させるリスクがある。
- 導入する場合は、通常の main mask loss よりかなり弱い重みで扱う前提が望ましい。

### encoder auxiliary outputs に mask loss を入れる場合
- encoder auxiliary は decoder よりも粗い proposal 段階なので、この時点で mask supervision を掛けても効果は読みづらい。
- 位置や意味がまだ十分に整っていない特徴に対して輪郭予測まで要求するため、改善幅の割に不安定化しやすい。
- 実装コストも高く、現状の `enc_aux_outputs` は box/logit 前提なので、mask 用の query feature 配線を追加する必要がある。
- 費用対効果は低めで、優先度は最も低い。

### 導入優先度の推奨
- 1. まず `auxiliary decoder outputs` に限定して mask loss を追加する。
- 2. その次に `denoising outputs` を弱い重みで試す。
- 3. `encoder auxiliary outputs` は最後に検討するか、見送る前提でよい。

### 期待される変化
- 増えやすいもの
  - VRAM 使用量
  - backward 時間
  - loss weight 調整の難しさ
  - 実装複雑度
- 改善が期待できるもの
  - mask の初期収束
  - 中間層の mask 表現学習
  - 条件次第での mask AP の微増
