# MiniGPT Addition

> このプロジェクトはGPTの学習を目的とした教育・実験用実装です。
> Transformerの仕組みを理解することを重視しています。

## Results

| Operation      | Accuracy  |
|----------------|----------:|
| Addition       | 100.00%   |
| Subtraction    | 100.00%   |
| Multiplication | 100.00%   |

Total: 118,803 / 118,803 correct (100.00%)

## プロジェクトの目的

本プロジェクトは、GPTの仕組みを理解することを目的として、PyTorchのみを用いて「2桁整数の足し算と引き算と掛け算」を学習する小規模なGPTを実装したものです。

## 特徴

- シンプルなGPTの実装（マルチヘッドAttention）
- 文字単位のトークナイザー
- Transformerデコーダー
- 加減乗算データセットでの学習
- 貪欲法（Greedy Decoding）によるテキスト生成
- 加減乗算モデルからContinue Pretrainingを行い、補正データを追加学習


## 学習手順

1. 加減乗算の学習データをまとめて混合データを作成
    make_dataset.pyで、"errors_flag = False"にする
    minigpt/dataset.pyで、datasetをtotal=199*199*3で生成。"errors_flag = False"にする
2. 混合データ(train_mix.txt)を30epochで学習(train.py)
　　# ===== ハイパーパラメータ =====
    learning_rate = 1e-4
    num_epochs = 30
3. モデルを評価して加減乗算のエラー抽出(test_add_sub_mul.py)
4. エラー補正データを作成
　　make_dataset.pyで、"errors_flag = True"にする
5. エラー補正データ(train_mix_errors.txt)を1epochで再学習
    minigpt/dataset.pyで、datasetをtotal=5000で生成。"errors_flag = True"にする
    train.pyで、Continue pretraining
　　# ===== ハイパーパラメータ =====
    learning_rate = 1e-5
    num_epochs = 1
6. モデルを評価して乗算エラー抽出(test_add_sub_mul.py)
7. 乗算エラー補正データを作成
    make_dataset.pyで、"errors_flag = True"にする
8. 乗算エラー補正データ(train_mix_errors.txt)を1epochで再学習
    minigpt/dataset.pyで、datasetをtotal=5000で生成。"errors_flag = True"にする
    train.pyで、Continue pretraining
　　# ===== ハイパーパラメータ =====
    learning_rate = 1e-5
    num_epochs = 1
9. モデルを評価して正解率100%(エラー0)を確認(test_add_sub_mul.py)
  % python test_eval_add_sub_mul.py
  Addition   : 100.00%
  Add Errors : 0
  Add Numeric Errors : 0
  Add Format Errors : 0
  Subtraction   : 100.00%
  Sub Errors : 0
  Sub Numeric Errors : 0
  Sub Format Errors : 0
  Multiplication   : 100.00%
  Mul Errors : 0
  Mul Numeric Errors : 0
  Mul Format Errors : 0


## Model configuration

```python
GPT_CONFIG = {
    "vocab_size": 15,
    "context_length": 12,
    "embed_dim": 256,
    "n_heads": 4,
}
```

## Full evaluation

A tiny GPT model that learns
addition, subtraction, and multiplication
for all integer pairs in [-99, 99].

Final evaluation:

| Operation      | Accuracy  |
|----------------|----------:|
| Addition       | 100.00%   |
| Subtraction    | 100.00%   |
| Multiplication | 100.00%   |

Total: 118,803 / 118,803 correct (100.00%)


## Observed failure patterns

- Sign inversion when the result magnitude is very small (especially ±1)
- Errors involving carry into the hundreds place (e.g. 103 → 093)


## 実行例(2桁整数の加算)
**入力**
```
03+58
```
**出力**
```
03+58=+0061
```

## 実行例(2桁整数の減算)
**入力**
```
30-50
```
**出力**
```
30-50=-0020
```

## 実行例(2桁整数の乗算)
**入力**
```
30*50
```
**出力**
```
30*50=+1500
```

（inference.py が '=' を自動で付けています）


## 開発環境

- Python 3.11
- PyTorch


## 事前準備(環境構築手順)

```bash
pip install -r requirements.txt
```

Macの場合
```bash
pip install torch torchvision torchaudio
```

Windows CUDAの場合
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 学習

まず、学習データを生成します。

```bash
python make_dataset.py
```

生成された `train_mix.txt` を `data` ディレクトリに配置します。

その後、学習を開始します。（詳細は学習手順を参照）

```bash
python train.py
```

## 推論

```bash
python inference.py
```

## 評価

```bash
python test_eval_add_sub_mul.py
```

## ステータス

⚠️ 本プロジェクトは現在活発に開発中です。

最初のリリース（v0.1.0）は実験的な実装です。
- v0.1.0では２桁加減乗算(-99～99)で全問正解を達成。
- 今後は3桁演算やモデル分析を進める予定。

## 学習時間の目安

参考環境（Windows 11 / GTX 1660 Ti）

- 加算モデル: 約46分
- 加減乗算混合モデル(30 epoch): 約12時間

## Observed failure patterns during development

- Sign inversion when the result magnitude is very small (especially ±1)
- Errors involving carry into the hundreds place (e.g. 103 → 093)
- Multiplication errors concentrated at ±20, ±40, ±80, ±100 offsets