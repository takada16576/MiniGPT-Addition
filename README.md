# MiniGPT Addition

> このプロジェクトはGPTの学習を目的とした教育・実験用実装です。
> 高い加算精度よりも、Transformerの仕組みを理解することを重視しています。

## プロジェクトの目的

本プロジェクトは、GPTの仕組みを理解することを目的として、PyTorchのみを用いて「2桁整数の足し算と引き算」を学習する小規模なGPTを実装したものです。

## 特徴

- シンプルなGPTの実装（マルチヘッドAttention）
- 文字単位のトークナイザー
- Transformerデコーダー
- 加算データセットでの学習
- 貪欲法（Greedy Decoding）によるテキスト生成
- 加算モデルからContinue Pretrainingを行い、減算を追加学習


## 学習手順

1. Generate the addition dataset.
2. Train the addition model.
3. Generate the subtraction dataset.
4. Continue pretraining from the addition model.

## Full evaluation (79,202 test cases)

- Addition: 39,601 problems → **99.81%**
- Subtraction: 39,601 problems → **99.93%**
- Overall: 79,202 problems → **99.87%**


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
03+58=061
```

## 実行例(2桁整数の減算)
**入力**
```
30-50
```
**出力**
```
30-50=-020
```

（inference.py が '=' を自動で付けています）


## 開発環境

- Python 3.11
- PyTorch


## 事前準備

```bash
pip install -r requirements.txt
```

## 学習

まず、学習データを生成します。

```bash
python make_dataset.py
```

生成された `train_add_sub.txt` を `data` ディレクトリに配置します。

その後、学習を開始します。

```bash
python train.py
```

## 推論

```bash
python inference.py
```

## ステータス

⚠️ 本プロジェクトは現在活発に開発中です。

最初のリリース（v0.1.0）は実験的な実装です。
既知の問題：
- 乱数シードによっては、モデルが収束しない場合があります。
- 演算精度については、現在も改善を進めています。


## 学習時間の目安

参考環境（Windows 11 / GTX 1660 Ti）では、加算モデルの学習に約46分かかりました。


## 今後の予定

- 掛け算への対応
- 演算精度の向上
- Transformerの計算特性の分析
