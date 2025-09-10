# LDA分析パイプライン

インタビューデータからLDA（Latent Dirichlet Allocation）を使用してトピックモデリングを実行するパイプラインです。

## ファイル構成

- `preprocessing.py`: テキストデータの前処理スクリプト
- `main.py`: LDA分析実行スクリプト
- `run_pipeline.py`: パイプライン全体を実行するスクリプト
- `requirements.txt`: 必要なPythonパッケージ
- `data/`: 生データ（.textファイル）を格納するディレクトリ
- `processed_data/`: 前処理済みデータを保存するディレクトリ
- `results/`: 分析結果を保存するディレクトリ

## セットアップ

### 1. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 2. MeCabのインストール（macOS）

```bash
# Homebrewを使用
brew install mecab mecab-ipadic

# または、condaを使用
conda install -c conda-forge mecab
```

## 使用方法

### 基本的な使用方法

1. `data/`ディレクトリに`.text`ファイルを配置
2. パイプラインを実行：

```bash
python run_pipeline.py
```

### オプション付きの実行

```bash
# トピック数を指定
python run_pipeline.py --n_topics 3

# 前処理のみ実行
python run_pipeline.py --skip_lda

# LDA分析のみ実行（前処理済みの場合）
python run_pipeline.py --skip_preprocessing
```

### 個別実行

```bash
# 前処理のみ
python preprocessing.py

# LDA分析のみ
python main.py
```

## データ形式

`data/`ディレクトリに配置する`.text`ファイルは以下の形式を想定しています：

```
インタビュー対象者：団体名　A様
インタビュー日時：2025年3月19日（水）13：00～15：00
インタビュー実施者：S
インタビュー場所：Zoom

S：それではよろしくお願いします。

A様：よろしくお願いします。緊張してます？（笑）

S：えっ、それはもちろん（笑）

A様：実際の会話内容...
```

## 出力結果

`results/`ディレクトリに以下のファイルが生成されます：

- `topic_words.csv`: 各トピックの主要単語
- `document_topics.csv`: 各文書のトピック分布
- `coherence_score.txt`: 一貫性スコア
- `lda_model.pkl`: 学習済みLDAモデル
- `topic_words.png`: トピック単語の可視化
- `document_topic_heatmap.png`: 文書-トピック分布のヒートマップ
- `wordcloud_topic_*.png`: 各トピックのワードクラウド
- `statement_topic_distribution.png`: 発言ごとのトピック分布可視化
- `statement_topic_summary.csv`: 発言ごとの主要トピックサマリー
- `statement_topics.csv`: 発言ごとの詳細トピック分布
- `statement_metadata.csv`: 発言メタデータ
- `stop_words.txt`: 使用されたストップワード一覧
- `word_frequencies.csv`: 単語の出現頻度データ
- `preprocessing_config.txt`: 前処理設定情報

## 前処理の詳細

1. **対象者発言抽出**: インタビューテキストから対象者（面接を受ける人）の発言のみを抽出
2. **発言ごとの分割**: 各発言を個別の文書として分割（発言ごとに分析）
3. **初期形態素解析**: MeCabを使用して日本語テキストをトークン化
4. **自動ストップワード生成**: 出現頻度に基づいてストップワードを自動決定
   - 高頻度単語（80%以上の文書に出現）を除去
   - 低頻度単語（1%未満の文書にのみ出現）を除去
5. **最終形態素解析**: 自動生成されたストップワードを適用
6. **文書-単語行列作成**: LDA用の数値データに変換

## LDA分析の詳細

1. **トピック数最適化**: 一貫性スコアを使用して最適なトピック数を自動決定
2. **モデル学習**: scikit-learnのLatentDirichletAllocationを使用
3. **結果可視化**: トピック単語、文書分布、ワードクラウドを生成
4. **評価**: 一貫性スコアでトピックの品質を評価

## 注意事項

- 日本語フォント（Hiragino Sans GB）を使用しているため、macOS以外ではフォントパスを変更してください
- 大量のテキストファイルがある場合、処理時間が長くなる可能性があります
- MeCabの辞書によっては形態素解析の結果が異なる場合があります

## トラブルシューティング

### MeCab関連のエラー

```bash
# MeCabがインストールされていない場合
brew install mecab mecab-ipadic

# Pythonパッケージがインストールされていない場合
pip install mecab-python3
```

### フォント関連のエラー

`main.py`の224行目のフォントパスを環境に合わせて変更してください：

```python
font_path='/System/Library/Fonts/Hiragino Sans GB.ttc'  # macOS用
```

### メモリ不足エラー

大量のデータを処理する場合、`preprocessing.py`の`min_df`や`max_df`パラメータを調整してください。

### ストップワードの調整

自動ストップワード生成の閾値を調整したい場合、`preprocessing.py`の`main()`関数内で以下のパラメータを変更してください：

```python
preprocessor = TextPreprocessor(
    auto_stop_words=True,      # 自動ストップワード生成を有効
    high_freq_threshold=0.8,   # 高頻度単語の閾値（80%以上の文書に出現）
    low_freq_threshold=0.01,   # 低頻度単語の閾値（1%未満の文書にのみ出現）
    target_speaker="対象者"     # 対象者の発言のみを抽出
)
```

- `high_freq_threshold`: 高頻度単語の閾値（0.0-1.0）
- `low_freq_threshold`: 低頻度単語の閾値（0.0-1.0）
- `target_speaker`: 抽出したい発言者の名前（デフォルト: "対象者"）

### 発言者設定の変更

異なる発言者名を使用している場合、`target_speaker`パラメータを変更してください：

```python
# 例：A様の発言のみを抽出したい場合
preprocessor = TextPreprocessor(target_speaker="A様")

# 例：面接者の発言のみを抽出したい場合
preprocessor = TextPreprocessor(target_speaker="面接者")
```
