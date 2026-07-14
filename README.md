# MiniGPT Addition

## プロジェクトの目的

本プロジェクトは、GPTの仕組みを理解することを目的として、PyTorchのみを用いて整数の足し算を学習する小規模なGPTを実装したものです。

## 特徴

- シンプルなGPTの実装（シングルヘッドAttention）
- 文字単位のトークナイザー
- Transformerデコーダー
- 加算データセットでの学習
- 貪欲法（Greedy Decoding）によるテキスト生成

## 実行例

**入力**

```
03+58=
```

**出力**

```
03+58=061
```

## 開発環境

- Python 3.11
- PyTorch

## 学習

まず、学習データを生成します。

```bash
python make_dataset.py
```

生成された `train.txt` を `data` ディレクトリに配置します。

その後、学習を開始します。

```bash
python train.py
```

## 推論

```bash
python inference.py
```