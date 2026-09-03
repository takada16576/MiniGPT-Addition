# MiniGPT Addition

> このプロジェクトはGPTの学習を目的とした教育・実験用実装です。
> Transformerの仕組みを理解することを重視しています。

## Results

2桁整数演算 — Chain of Thought
Operation	Accuracy
Addition	100.00%
Subtraction	100.00%
Multiplication	100.00%

2桁整数 [-99, 99] の加算・減算・乗算について、Chain of Thought（CoT）形式による学習を行い、全問正解を達成しました。

## プロジェクトの目的

本プロジェクトは、GPTの仕組みを理解することを目的として、PyTorchのみを用いて小規模なGPTを実装したものです。
2桁整数の足し算・引き算・掛け算を題材として、通常の直接回答だけでなく、計算途中のステップを生成させるChain of Thought（CoT）形式について実験しています。
最終的には、モデル自身に計算手順を生成させ、その結果から正しい演算結果を導くことを目指しています。

## 特徴

- シンプルなGPTの実装（マルチヘッドAttention）
- 文字単位のトークナイザー
- Transformerデコーダー
- 加算・減算・乗算データセット
- 貪欲法（Greedy Decoding）によるテキスト生成
- Chain of Thought（CoT）形式による演算
- CoTによる計算ステップの学習
- 学習済みモデルを利用したContinue Pretraining
- 全39,601通りの2桁整数ペアによる演算評価

## Chain of Thought

通常の演算では、

+0034*+0067=+0002278

のように、入力から直接答えを生成します。

CoT版では、計算途中のステップを含む形式にしています。

例えば、

+0034*+0067=+0034*+0007=+000000238|+0034*+0060=+000002040|+0238++2040=+000002278|+000002278

のように、

1の位との乗算
10の位との乗算
部分積の加算
最終結果

という計算手順をモデルに学習させます。

この形式によって、単純に答えを暗記させるのではなく、演算の途中経過を生成する能力について実験しています。

## データセット

2桁整数の範囲は [-99, 99] です。

整数の組み合わせは、

199 × 199 = 39,601 通りです。

加算・減算・乗算の3種類を使用するため、評価対象は、

39,601 × 3 = 118,803 問題です。

## CoTデータ生成

1桁演算用：

python make_dataset_1dig_cot.py

2桁演算用：

python make_dataset_2dig_cot.py


## 学習

CoT用の学習プログラムは train_cot.py です。

python train_cot.py

CoT形式では計算途中のステップを生成するため、通常の演算形式よりも必要なコンテキスト長が大きくなります。

## Model configuration

現在のモデルは以下の設定を使用しています。

GPT_CONFIG = {
    "vocab_size": 15,
    "context_length": 102,
    "embed_dim": 256,
    "n_heads": 4,
}

## 学習済みモデル

2桁整数の加算・減算・乗算をCoT形式で学習したモデル：

minigpt/model_mix_2dig_cot_100.pt

このモデルでは、2桁演算の全評価問題について100%の正解率を達成しています。

## 評価

### サンプル評価
python test_eval_mix_2dig_cot_sample.py

### 全問評価
python test_eval_mix_2dig_cot_full.py

評価対象：

Addition       : 39,601
Subtraction    : 39,601
Multiplication : 39,601
--------------------------------
Total          : 118,803

最終結果：

Operation	Accuracy
Addition	100.00%
Subtraction	100.00%
Multiplication	100.00%
Total: 118,803 / 118,803 correct (100.00%)

## 推論
python inference.py

## 実行例
python inference.py 

### 2桁整数の加算
#### 入力
03+58
#### 出力
03+58=+0061

### 2桁整数の減算
#### 入力
30-50
#### 出力
30-50=-0020

### 2桁整数の乗算
#### 入力
30*50
#### 出力
30*50=+1500

inference.py が = を自動的に付加します。

## 開発環境
Python 3.11
PyTorch

## 事前準備（環境構築）
pip install -r requirements.txt
Mac
pip install torch torchvision torchaudio
Windows CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

プロジェクト構成

## 現在の主要ファイル：

```text
MiniGPT-Addition/
├── inference.py
├── make_dataset_1dig_cot.py
├── make_dataset_2dig_cot.py
├── train_cot.py
├── test_eval_mix_2dig_cot_full.py
├── test_eval_mix_2dig_cot_sample.py
├── test_mix_2dig_cot.txt
├── minigpt/
│   ├── dataset.py
│   ├── generate.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── utils.py
│   └── model_mix_2dig_cot_100.pt
└── README.md

## Observed failure patterns during development

開発途中では、以下のようなエラーが確認されました。

結果が +1 / -1 付近になる場合の符号反転
繰り上がりが発生する計算での誤り
乗算における特定の値付近での誤り
CoT形式に変更した際のコンテキスト長不足
計算途中のステップは正しいものの、最終結果だけが誤るケース

CoT学習では、学習エポック数の増加に伴って乗算の正解率が大きく向上しました。

## ステータス
現在
2桁CoT演算モデル完成

2桁加算：100%
2桁減算：100%
2桁乗算：100%
全118,803問題で正解
CoT形式による計算手順の生成を実現

## 開発の流れ
```text
1桁演算
   ↓
2桁演算
   ↓
2桁加減乗算
   ↓
2桁演算 100%
   ↓
Chain of Thought（CoT）導入
   ↓
2桁CoT演算 100%
   ↓
3桁CoT演算 ← 次の目標

今後は、2桁CoTで得られたモデルと学習方法をベースとして、3桁整数の演算へ拡張する予定です。

## 学習時間の目安

参考環境：
Windows 11
Intel Core i7-8700
NVIDIA GeForce GTX 1660 Ti 6GB
RAM 16GB

また、MacBook Air M4でも学習・評価を行っています。

学習時間は、データセットのサイズ、batch size、モデル設定、epoch数などによって変化します。

## License

教育・実験目的のプロジェクトです。
