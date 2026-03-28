# DEIMv2 インスタンスセグメンテーション改善計画

## 目的

- DEIMv2 の body instance segmentation 精度を、推論 I/O を実質変えずに引き上げる。
- `human-instance-segmentation` から、境界とインスタンス分離に効く要素だけを移植する。
- wholebody40 の instance 系 config で再利用できる共通基盤として実装する。

## 意図的に移植しないもの

- full-image pre-trained UNet guidance
- hierarchical UNet の本体アーキテクチャ
- distillation
- マルチスケール segmentation の再設計
- 参照元の外付け auxiliary-task wrapper

## 採用する設計

- 既存の DEIMv2 の `pred_masks` 経路は維持する。
- [`engine/deim/deim_decoder.py`](/home/b920405/git/DEIMv2/engine/deim/deim_decoder.py) に train 時専用の補助 mask head を追加する。
  - `pred_mask_contours`
  - `pred_mask_distances`
- [`engine/deim/deim_criterion.py`](/home/b920405/git/DEIMv2/engine/deim/deim_criterion.py) に追加 mask loss を実装する。
  - `loss_mask_boundary`
  - `loss_mask_contour`
  - `loss_mask_distance`
- 追加 loss はすべて `mask_valid=True` の matched target のみに適用する。
- postprocess と eval の出力は変えない。補助予測は train 時のみ使う。

## 追加した設定インターフェース

### `DEIMTransformer`

- `use_contour_aux_head: bool = False`
- `use_distance_aux_head: bool = False`
- `aux_mask_feature_level: int | None = None`

#### `DEIMTransformer` パラメータ詳細

- `use_contour_aux_head`
  - decoder 内の contour 予測用補助 branch を有効化する。
  - `True` のとき、train 時の出力 dict に `pred_mask_contours` が追加される。
  - 通常は `DEIMCriterion.use_contour_detection=True` と組み合わせて使う。
  - eval/deploy の出力には影響しない。

- `use_distance_aux_head`
  - decoder 内の distance-map 予測用補助 branch を有効化する。
  - `True` のとき、train 時の出力 dict に `pred_mask_distances` が追加される。
  - 通常は `DEIMCriterion.use_distance_transform=True` と組み合わせて使う。
  - eval/deploy の出力には影響しない。

- `aux_mask_feature_level`
  - contour / distance の補助 head が参照する encoder feature level を指定する。
  - `None` の場合は `mask_feature_level` を使う。
  - まずは `mask_feature_level` と同じ値に揃えて使うのが基本。
  - より低 stride の feature を使うと境界表現は鋭くなりやすいが、メモリ消費は増えやすい。
  - 初期推奨値:
    - stride-16 系モデル: `0`
    - stride-8/16/32 の 3-level モデル: `1`

### `DEIMCriterion`

- `use_boundary_aware_loss: bool = False`
- `boundary_aware_width: int = 3`
- `boundary_aware_weight: float = 2.0`
- `use_contour_detection: bool = False`
- `use_distance_transform: bool = False`
- `distance_transform_steps: int = 5`

#### `DEIMCriterion` パラメータ詳細

- `use_boundary_aware_loss`
  - main の `pred_masks` に対する boundary-weighted BCE を有効化する。
  - 補助 head を追加せずに試せるため、最も低リスクな最初の候補。
  - まずはこれ単体で効果を見るのが推奨。

- `boundary_aware_width`
  - GT mask から生成する境界帯の太さを制御する。
  - 値を大きくすると boundary とみなす領域が広がる。
  - 小さい値は境界に鋭く集中し、大きい値は少し粗いが安定した重み付けになる。
  - 初期推奨値: `3`
  - 出力解像度がかなり低い場合以外は、最初は増やしすぎない。

- `boundary_aware_weight`
  - 境界画素にかける BCE の重み倍率。
  - 大きいほど境界重視になる。
  - 高すぎると内部領域の安定性を損なう可能性がある。
  - 初期推奨値: `2.0`
  - 初期探索の現実的な範囲: `1.5` から `3.0`

- `use_contour_detection`
  - `pred_mask_contours` に対する contour supervision を有効化する。
  - 実際に学習させるには `DEIMTransformer.use_contour_aux_head=True` が必要。
  - 最初の単独実験よりも、`use_boundary_aware_loss=True` と併用する方が良い。

- `use_distance_transform`
  - `pred_mask_distances` に対する distance supervision を有効化する。
  - 実際に学習させるには `DEIMTransformer.use_distance_aux_head=True` が必要。
  - 境界付近の形状整合性や、接触インスタンスの分離改善を狙う設定。

- `distance_transform_steps`
  - 近似 distance target を作るときの iterative max-pooling 回数。
  - 値を大きくすると pseudo-distance の影響範囲が広がる。
  - 小さすぎると信号が弱く、大きすぎると target が平滑になりすぎる。
  - 初期推奨値: `5`
  - 初期アブレーションでは固定して、まず branch の有無を評価する。

### 推奨する初期 loss weight

- `loss_mask_bce: 5`
- `loss_mask_dice: 5`
- `loss_mask_boundary: 1.0`
- `loss_mask_contour: 0.5`
- `loss_mask_distance: 0.25`

#### Loss weight の詳細

- `loss_mask_bce`
  - matched valid mask に対するベースの mask supervision。
  - 追加要素の比較を明確にするため、初期実験では変えない。

- `loss_mask_dice`
  - 重なりを重視するベースの mask supervision。
  - これも初期実験では据え置く。

- `loss_mask_boundary`
  - main mask logits に対する boundary-aware BCE の重み。
  - ベースの BCE/Dice と同じ予測経路に作用するので、まずは控えめに始める。
  - 初期推奨値: `1.0`

- `loss_mask_contour`
  - contour 補助 supervision の重み。
  - 主目的ではなく構造正則化に近いので、main loss より小さめに置く。
  - 初期推奨値: `0.5`

- `loss_mask_distance`
  - distance 補助 supervision の重み。
  - exact distance ではなく近似 target を使うため、contour よりさらに弱めから始める。
  - 初期推奨値: `0.25`

## パラメータの組み合わせルール

- `use_contour_detection=True` を使うなら、通常は `use_contour_aux_head=True` も同時に有効化する。
- `use_distance_transform=True` を使うなら、通常は `use_distance_aux_head=True` も同時に有効化する。
- `use_boundary_aware_loss=True` は単独で有効化でき、最初の実験候補として最も推奨。
- contour と distance を同時に有効化する場合、最初の比較では `aux_mask_feature_level` を `mask_feature_level` に揃える。
- 初期アブレーションでは、次のグループを一度に一つだけ変える。
  - boundary-only の設定群
  - contour branch と `loss_mask_contour`
  - distance branch と `loss_mask_distance`

## 実装順

1. decoder に train 時専用の contour / distance head を追加する。
2. criterion に boundary-aware / contour / distance loss を追加する。
3. wholebody40 の instance config 群に、デフォルト off の設定キーを公開する。
4. 構文チェックと smoke test を回す。
5. `deimv2_dinov3_x_wholebody40_ins.yml` を基準に小規模比較を回す。

## 完了条件

- 補助 head が off のとき、decoder の出力は現状と同一である。
- eval/deploy の出力 schema が `pred_logits`, `pred_boxes`, `pred_masks` のままである。
- 通常 batch、空 mask 混在 batch、全 `mask_valid=False` batch で NaN が出ない。
- デフォルト off のままで既存 baseline config が動く。

## 実験マトリクス

| 実験 | `use_boundary_aware_loss` | `use_contour_detection` | `use_distance_transform` | `use_contour_aux_head` | `use_distance_aux_head` | 目的 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | False | False | False | False | False | 現行動作 |
| boundary only | True | False | False | False | False | 最初の候補 |
| boundary + contour | True | True | False | True | False | 境界分離の強化 |
| boundary + distance | True | False | True | False | True | 形状整合性の強化 |
| boundary + contour + distance | True | True | True | True | True | 全部入り候補 |

## 評価優先順位

1. `segm AP`
2. `segm AP75`
3. `bbox AP` 非劣化

## `-t last_full_epoch.pth` を使ったファインチューニングの整理

### 結論

- 今回追加したパラメータが未反映の古い checkpoint を `-t last_full_epoch.pth` で指定し、current config 側で今回追加したパラメータを有効化して fine-tuning することは可能。
- ただしこれは `resume` ではなく `tuning` であり、復元されるのは model の一致部分の重みだけ。
- optimizer、EMA、scaler、scheduler、resume metadata は引き継がれない。
- 今回の実装後に保存された新形式 checkpoint についても、`-t` 経路で non-tensor state を無視するように修正済みのため、原理上は再利用可能。

### `tuning` と `resume` の違い

- `resume`
  - 学習状態をそのまま再開する経路。
  - model だけでなく optimizer、EMA、scaler、scheduler、loader 状態、RNG 状態まで復元する。
  - 現在の実装では `DEIMTransformer` と `DEIMCriterion` の追加パラメータも checkpoint から復元対象。

- `tuning`
  - 学習状態の再開ではなく、model 重みの流用から新しい学習を開始する経路。
  - current config で model / criterion / optimizer を新規構築した上で、checkpoint の model 重みだけ部分ロードする。
  - そのため、今回追加した loss 有効化フラグや weight は checkpoint 側ではなく current config 側の値が使われる。

### 古い checkpoint から fine-tuning できる理由

- `-t` は current config を先に読んで model を current 構成で構築する。
- その後、checkpoint から `model` または `ema.module` の state_dict を取り出し、一致する key と shape の重みだけを流し込む。
- 古い checkpoint には今回追加した以下の重みが存在しない:
  - `contour_embed_head.*`
  - `contour_feature_head.*`
  - `distance_embed_head.*`
  - `distance_feature_head.*`
- これらは missing key 扱いになり、current config で生成された初期値のまま学習開始する。
- 既存の backbone / encoder / decoder / main `pred_masks` 経路は重みを継承できるため、追加 branch だけを新規学習する形で fine-tuning できる。

### current config 側で有効化してよい項目

- `DEIMTransformer.use_contour_aux_head`
- `DEIMTransformer.use_distance_aux_head`
- `DEIMTransformer.aux_mask_feature_level`
- `DEIMCriterion.use_boundary_aware_loss`
- `DEIMCriterion.boundary_aware_width`
- `DEIMCriterion.boundary_aware_weight`
- `DEIMCriterion.use_contour_detection`
- `DEIMCriterion.use_distance_transform`
- `DEIMCriterion.distance_transform_steps`
- `DEIMCriterion.weight_dict` の追加 loss weight

上記は `tuning` では checkpoint から復元する対象ではなく、current config で新しく有効化される。

### 実運用上の意味

- 古い baseline の `last_full_epoch.pth` を使って、新しい aux head と新しい loss を入れた構成へ移行することは可能。
- その場合の学習開始点は次のように分かれる:
  - 既存本体: 旧 checkpoint の重みを継承
  - 新規 aux head: ランダム初期化
  - 新規 loss 設定: current config の値を使用

### 現時点の制約

- 今回の resume 改善で `state_dict` に `_extra_state` が入るようになった。
- これに対して、`tuning` 側では non-tensor state を部分ロード対象から除外するよう修正済み。
- そのため `_extra_state` を含む新形式 checkpoint を `-t` に渡しても、現在は tensor parameter のみで安全に matching される。
- 新旧どちらの checkpoint でも、`-t` の意味は変わらず「一致する model 重みだけ部分ロードし、current config で新規学習を開始する」こと。
- 依然として注意すべき点は以下:
  - 新しい aux head 自体の重みは、checkpoint に存在しなければランダム初期化になる
  - criterion や optimizer の状態は `tuning` では引き継がれない
  - current config と checkpoint の構造差が大きい場合は、missing key が増えるため読み込みログの確認が必要

### 推奨運用

- 旧 baseline checkpoint から新構成へ移行したい場合:
  - `-t old_last_full_epoch.pth` を使う
  - current config で今回追加した機能を明示的に有効化する
  - 新規 aux head はランダム初期化から学習される前提で使う

- 新形式 checkpoint を使って別条件へ再 fine-tuning したい場合:
  - `-t newer_checkpoint.pth` も利用可能
  - `_extra_state` は tuning 時に無視されるため、部分ロードでは落ちない
  - ただし `resume` のような完全復元ではないので、current config 側の設定で新しい学習を開始することを前提にする

- 完全再開したい場合:
  - `-r` を使う
  - checkpoint 保存時と同じ config を使う

### 追加で確認すべきこと

- `-t` で読み込んだ際の missing key 一覧をログに出し、新規 aux head のみ missing であることを確認できるようにするか
- `old baseline -> new aux config` の移行を正式運用にするなら、推奨コマンド例を別途記載するか

## 進捗メモ

- decoder 共通基盤: 実装済み
- criterion の追加 loss: 実装済み
- wholebody40 instance config への設定公開: 実装済み
- 実データでの比較評価: 未実施
