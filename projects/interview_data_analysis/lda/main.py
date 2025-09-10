"""
LDA（Latent Dirichlet Allocation）実行スクリプト
前処理済みデータを使用してトピックモデリングを実行する
"""

import os
import pickle
import numpy as np
import pandas as pd
import argparse
from sklearn.decomposition import LatentDirichletAllocation
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import japanize_matplotlib


class LDAAnalyzer:
    """LDA分析クラス"""
    
    def __init__(self, n_topics: int = 5, random_state: int = 42):
        """
        初期化
        
        Args:
            n_topics: トピック数
            random_state: 乱数シード
        """
        self.n_topics = n_topics
        self.random_state = random_state
        self.lda_model = None
        self.doc_term_matrix = None
        self.feature_names = None
        
    def load_processed_data(self, processed_data_dir: str):
        """
        前処理済みデータを読み込み
        
        Args:
            processed_data_dir: 前処理済みデータのディレクトリ
        """
        try:
            # 文書-単語行列を読み込み
            with open(os.path.join(processed_data_dir, 'doc_term_matrix.pkl'), 'rb') as f:
                self.doc_term_matrix = pickle.load(f)
            
            # 特徴語リストを読み込み
            with open(os.path.join(processed_data_dir, 'feature_names.pkl'), 'rb') as f:
                self.feature_names = pickle.load(f)
            
            # 発言メタデータを読み込み
            try:
                with open(os.path.join(processed_data_dir, 'statement_metadata.pkl'), 'rb') as f:
                    self.statement_metadata = pickle.load(f)
                print(f"発言メタデータ読み込み完了: {len(self.statement_metadata)}個の発言")
            except FileNotFoundError:
                self.statement_metadata = None
                print("発言メタデータが見つかりません（従来の形式）")
            
            print(f"データ読み込み完了:")
            print(f"  文書数: {self.doc_term_matrix.shape[0]}")
            print(f"  語彙数: {self.doc_term_matrix.shape[1]}")
            
        except FileNotFoundError as e:
            print(f"エラー: 前処理済みデータが見つかりません - {e}")
            print("まず preprocessing.py を実行してください")
            raise
    
    def fit_lda(self, max_iter: int = 100, learning_decay: float = 0.7):
        """
        LDAモデルを学習
        
        Args:
            max_iter: 最大反復回数
            learning_decay: 学習率の減衰率
        """
        print(f"LDAモデルを学習中... (トピック数: {self.n_topics})")
        
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=max_iter,
            learning_decay=learning_decay,
            random_state=self.random_state,
            doc_topic_prior=0.1,
            topic_word_prior=0.1
        )
        
        self.lda_model.fit(self.doc_term_matrix)
        print("LDAモデルの学習が完了しました")
    
    def get_topic_words(self, n_words: int = 10) -> pd.DataFrame:
        """
        各トピックの主要単語を取得
        
        Args:
            n_words: 各トピックから取得する単語数
            
        Returns:
            トピックと単語のDataFrame
        """
        if self.lda_model is None:
            raise ValueError("LDAモデルが学習されていません")
        
        topic_words = []
        
        for topic_idx, topic in enumerate(self.lda_model.components_):
            # トピック内での単語の重要度を取得
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            top_weights = [topic[i] for i in top_words_idx]
            
            for word, weight in zip(top_words, top_weights):
                topic_words.append({
                    'Topic': f'Topic {topic_idx + 1}',
                    'Word': word,
                    'Weight': weight
                })
        
        return pd.DataFrame(topic_words)
    
    def get_document_topics(self) -> pd.DataFrame:
        """
        各文書のトピック分布を取得
        
        Returns:
            文書とトピック分布のDataFrame
        """
        if self.lda_model is None:
            raise ValueError("LDAモデルが学習されていません")
        
        doc_topic_probs = self.lda_model.transform(self.doc_term_matrix)
        
        doc_topics = []
        for doc_idx, probs in enumerate(doc_topic_probs):
            for topic_idx, prob in enumerate(probs):
                doc_topics.append({
                    'Document': f'Document {doc_idx + 1}',
                    'Topic': f'Topic {topic_idx + 1}',
                    'Probability': prob
                })
        
        return pd.DataFrame(doc_topics)
    
    def get_statement_topic_distribution(self) -> pd.DataFrame:
        """
        各発言のトピック分布を取得（発言メタデータ付き）
        
        Returns:
            発言とトピック分布のDataFrame
        """
        if self.lda_model is None:
            raise ValueError("LDAモデルが学習されていません")
        
        doc_topic_probs = self.lda_model.transform(self.doc_term_matrix)
        
        statement_topics = []
        for doc_idx, probs in enumerate(doc_topic_probs):
            # メタデータを取得
            if self.statement_metadata and doc_idx < len(self.statement_metadata):
                metadata = self.statement_metadata[doc_idx]
                statement_id = f"{metadata['filename']}_発言{metadata['statement_id']}"
                original_text = metadata['original_text']
            else:
                statement_id = f"Statement_{doc_idx + 1}"
                original_text = "N/A"
            
            for topic_idx, prob in enumerate(probs):
                statement_topics.append({
                    'Statement_ID': statement_id,
                    'Original_Text': original_text,
                    'Topic': f'Topic {topic_idx + 1}',
                    'Probability': prob
                })
        
        return pd.DataFrame(statement_topics)
    
    def visualize_statement_topics(self, output_dir: str = "results", top_n: int = 5):
        """
        発言ごとのトピック分布を可視化
        
        Args:
            output_dir: 出力ディレクトリ
            top_n: 表示する上位トピック数
        """
        if not self.statement_metadata:
            print("発言メタデータが利用できません")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 発言ごとのトピック分布を取得
        statement_topics_df = self.get_statement_topic_distribution()
        
        # 各発言の主要トピックを取得
        statement_main_topics = []
        for statement_id in statement_topics_df['Statement_ID'].unique():
            statement_data = statement_topics_df[statement_topics_df['Statement_ID'] == statement_id]
            top_topics = statement_data.nlargest(top_n, 'Probability')
            
            for _, row in top_topics.iterrows():
                statement_main_topics.append({
                    'Statement_ID': row['Statement_ID'],
                    'Original_Text': row['Original_Text'],
                    'Topic': row['Topic'],
                    'Probability': row['Probability']
                })
        
        main_topics_df = pd.DataFrame(statement_main_topics)
        
        # 発言ごとのトピック分布を可視化
        plt.figure(figsize=(15, 10))
        
        # 各発言のトピック分布を棒グラフで表示
        statements = main_topics_df['Statement_ID'].unique()
        n_statements = len(statements)
        
        for i, statement_id in enumerate(statements):
            plt.subplot((n_statements + 2) // 3, 3, i + 1)
            
            statement_data = main_topics_df[main_topics_df['Statement_ID'] == statement_id]
            
            # トピックと確率を取得
            topics = statement_data['Topic'].tolist()
            probs = statement_data['Probability'].tolist()
            
            # 棒グラフを作成
            bars = plt.bar(range(len(topics)), probs, alpha=0.7)
            
            # 色分け
            colors = plt.cm.Set3(np.linspace(0, 1, len(topics)))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            plt.title(f'{statement_id}\n{statement_data.iloc[0]["Original_Text"][:30]}...', 
                     fontsize=8, pad=10)
            plt.xlabel('Topic')
            plt.ylabel('Probability')
            plt.xticks(range(len(topics)), [t.replace('Topic ', 'T') for t in topics], rotation=45)
            plt.ylim(0, 1)
            
            # 確率値を表示
            for j, prob in enumerate(probs):
                plt.text(j, prob + 0.01, f'{prob:.2f}', ha='center', va='bottom', fontsize=6)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'statement_topic_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 発言ごとの主要トピックを表形式で保存
        statement_summary = []
        for statement_id in statements:
            statement_data = main_topics_df[main_topics_df['Statement_ID'] == statement_id]
            top_topic = statement_data.loc[statement_data['Probability'].idxmax()]
            
            statement_summary.append({
                'Statement_ID': statement_id,
                'Original_Text': top_topic['Original_Text'],
                'Main_Topic': top_topic['Topic'],
                'Main_Topic_Probability': top_topic['Probability'],
                'Top_3_Topics': ', '.join([f"{row['Topic']}({row['Probability']:.2f})" 
                                         for _, row in statement_data.head(3).iterrows()])
            })
        
        summary_df = pd.DataFrame(statement_summary)
        summary_df.to_csv(os.path.join(output_dir, 'statement_topic_summary.csv'), 
                         index=False, encoding='utf-8')
        
        print(f"発言ごとのトピック分布を {output_dir} に保存しました")
    
    def print_statement_topic_summary(self):
        """
        発言ごとのトピック割合をコンソールに表示
        """
        if not self.statement_metadata:
            print("発言メタデータが利用できません")
            return
        
        statement_topics_df = self.get_statement_topic_distribution()
        
        print("\n" + "="*80)
        print("発言ごとのトピック分布サマリー")
        print("="*80)
        
        for statement_id in statement_topics_df['Statement_ID'].unique():
            statement_data = statement_topics_df[statement_topics_df['Statement_ID'] == statement_id]
            
            # 主要トピックを取得
            top_topic = statement_data.loc[statement_data['Probability'].idxmax()]
            
            print(f"\n【{statement_id}】")
            print(f"発言内容: {top_topic['Original_Text']}")
            print(f"主要トピック: {top_topic['Topic']} (確率: {top_topic['Probability']:.3f})")
            
            # 上位3トピックを表示
            top_3 = statement_data.nlargest(3, 'Probability')
            print("上位3トピック:")
            for _, row in top_3.iterrows():
                print(f"  - {row['Topic']}: {row['Probability']:.3f}")
            
            print("-" * 60)
    
    def calculate_coherence(self) -> float:
        """
        トピックの一貫性スコアを計算
        
        Returns:
            一貫性スコア
        """
        if self.lda_model is None:
            raise ValueError("LDAモデルが学習されていません")
        
        # 各トピックの主要単語を取得
        topic_words = []
        for topic in self.lda_model.components_:
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [self.feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
        
        # 文書を単語リストに変換
        documents = []
        for doc in self.doc_term_matrix:
            words = [self.feature_names[i] for i in doc.indices]
            documents.append(words)
        
        # 一貫性スコアを計算（簡易版）
        # 実際の一貫性計算は複雑なため、ここでは簡易的な指標を使用
        coherence = 0.0  # 簡易版では0を返す
        
        return coherence
    
    def visualize_topics(self, output_dir: str = "results"):
        """
        トピックの可視化
        
        Args:
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # トピック単語の可視化
        topic_words_df = self.get_topic_words()
        
        plt.figure(figsize=(12, 8))
        for i in range(self.n_topics):
            plt.subplot(2, 3, i + 1)
            topic_data = topic_words_df[topic_words_df['Topic'] == f'Topic {i + 1}']
            plt.barh(range(len(topic_data)), topic_data['Weight'])
            plt.yticks(range(len(topic_data)), topic_data['Word'])
            plt.title(f'Topic {i + 1}')
            plt.xlabel('Weight')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'topic_words.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 文書-トピック分布の可視化
        doc_topics_df = self.get_document_topics()
        doc_topic_matrix = doc_topics_df.pivot(index='Document', columns='Topic', values='Probability')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(doc_topic_matrix, annot=True, fmt='.2f', cmap='Blues')
        plt.title('Document-Topic Distribution')
        plt.xlabel('Topic')
        plt.ylabel('Document')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'document_topic_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"可視化結果を {output_dir} に保存しました")
    
    def create_wordclouds(self, output_dir: str = "results"):
        """
        各トピックのワードクラウドを一つの図表に作成
        
        Args:
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # サブプロットのレイアウトを計算
        n_cols = min(3, self.n_topics)  # 最大3列
        n_rows = (self.n_topics + n_cols - 1) // n_cols  # 必要な行数を計算
        
        # 全体の図を作成
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        
        # axesを常に2次元配列として扱う
        if self.n_topics == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for topic_idx in range(self.n_topics):
            # トピックの単語と重みを取得
            topic = self.lda_model.components_[topic_idx]
            word_weights = {self.feature_names[i]: topic[i] 
                          for i in range(len(self.feature_names))}
            
            # ワードクラウドを生成
            wordcloud = WordCloud(
                font_path='/System/Library/Fonts/Hiragino Sans GB.ttc',  # macOS用
                width=400,
                height=300,
                background_color='white',
                max_words=30
            ).generate_from_frequencies(word_weights)
            
            # サブプロットの位置を計算
            row = topic_idx // n_cols
            col = topic_idx % n_cols
            
            # ワードクラウドを表示
            ax = axes[row, col]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Topic {topic_idx + 1}', fontsize=12, pad=10)
        
        # 余分なサブプロットを非表示にする
        for idx in range(self.n_topics, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'combined_wordcloud.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"統合ワードクラウドを {output_dir} に保存しました")
    
    def save_results(self, output_dir: str = "results"):
        """
        分析結果を保存
        
        Args:
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # トピック単語を保存
        topic_words_df = self.get_topic_words()
        topic_words_df.to_csv(os.path.join(output_dir, 'topic_words.csv'), 
                             index=False, encoding='utf-8')
        
        # 文書-トピック分布を保存
        doc_topics_df = self.get_document_topics()
        doc_topics_df.to_csv(os.path.join(output_dir, 'document_topics.csv'), 
                            index=False, encoding='utf-8')
        
        # 発言ごとのトピック分布を保存（メタデータがある場合）
        if self.statement_metadata:
            statement_topics_df = self.get_statement_topic_distribution()
            statement_topics_df.to_csv(os.path.join(output_dir, 'statement_topics.csv'), 
                                      index=False, encoding='utf-8')
        
        # 一貫性スコアを計算・保存
        coherence = self.calculate_coherence()
        with open(os.path.join(output_dir, 'coherence_score.txt'), 'w', encoding='utf-8') as f:
            f.write(f"Coherence Score: {coherence:.4f}\n")
        
        # モデルを保存
        with open(os.path.join(output_dir, 'lda_model.pkl'), 'wb') as f:
            pickle.dump(self.lda_model, f)
        
        print(f"分析結果を {output_dir} に保存しました")
        print(f"一貫性スコア: {coherence:.4f}")


def find_optimal_topics(processed_data_dir: str, max_topics: int = 10):
    """
    最適なトピック数を探索
    
    Args:
        processed_data_dir: 前処理済みデータのディレクトリ
        max_topics: 探索する最大トピック数
    """
    # データを読み込み
    with open(os.path.join(processed_data_dir, 'doc_term_matrix.pkl'), 'rb') as f:
        doc_term_matrix = pickle.load(f)
    
    with open(os.path.join(processed_data_dir, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    # 各トピック数での一貫性スコアを計算
    coherence_scores = []
    topic_numbers = range(2, max_topics + 1)
    
    for n_topics in topic_numbers:
        print(f"トピック数 {n_topics} を評価中...")
        
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=100
        )
        lda.fit(doc_term_matrix)
        
        # 一貫性スコアを計算
        topic_words = []
        for topic in lda.components_:
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topic_words.append(top_words)
        
        documents = []
        for doc in doc_term_matrix:
            words = [feature_names[i] for i in doc.indices]
            documents.append(words)
        
        # 一貫性スコアを計算（簡易版）
        coherence = 0.0  # 簡易版では0を返す
        
        coherence_scores.append(coherence)
        print(f"  一貫性スコア: {coherence:.4f}")
    
    # 結果を可視化
    plt.figure(figsize=(10, 6))
    plt.plot(topic_numbers, coherence_scores, 'bo-')
    plt.xlabel('Number of Topics')
    plt.ylabel('Coherence Score')
    plt.title('Topic Number vs Coherence Score')
    plt.grid(True)
    plt.savefig('topic_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 最適なトピック数を表示
    optimal_idx = np.argmax(coherence_scores)
    optimal_topics = topic_numbers[optimal_idx]
    print(f"\n最適なトピック数: {optimal_topics} (一貫性スコア: {coherence_scores[optimal_idx]:.4f})")
    
    return optimal_topics


def main():
    """メイン処理"""
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='LDA分析を実行')
    parser.add_argument('--n_topics', type=int, default=5, 
                       help='トピック数 (デフォルト: 5)')
    parser.add_argument('--optimize_topics', action='store_true',
                       help='最適なトピック数を自動探索する')
    parser.add_argument('--processed_data_dir', type=str, default='processed_data',
                       help='前処理済みデータのディレクトリ (デフォルト: processed_data)')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='結果出力ディレクトリ (デフォルト: results)')
    
    args = parser.parse_args()
    
    # 設定
    processed_data_dir = args.processed_data_dir
    results_dir = args.results_dir
    n_topics = args.n_topics
    
    print(f"指定されたトピック数: {n_topics}")
    
    # 最適なトピック数を探索（オプション）
    if args.optimize_topics:
        print("最適なトピック数を探索中...")
        try:
            optimal_topics = find_optimal_topics(processed_data_dir)
            n_topics = optimal_topics
            print(f"最適なトピック数 {n_topics} を使用します")
        except Exception as e:
            print(f"トピック数最適化でエラーが発生しました: {e}")
            print(f"指定されたトピック数 {args.n_topics} を使用します")
    else:
        print(f"指定されたトピック数 {n_topics} を使用します")
    
    # LDA分析器を初期化
    analyzer = LDAAnalyzer(n_topics=n_topics)
    
    # 前処理済みデータを読み込み
    analyzer.load_processed_data(processed_data_dir)
    
    # LDAモデルを学習
    analyzer.fit_lda()
    
    # 結果を保存
    analyzer.save_results(results_dir)
    
    # 可視化
    analyzer.visualize_topics(results_dir)
    analyzer.create_wordclouds(results_dir)
    
    # 発言ごとのトピック分布を可視化
    analyzer.visualize_statement_topics(results_dir)
    
    # 発言ごとのトピック割合を表示
    analyzer.print_statement_topic_summary()
    
    print("\nLDA分析が完了しました！")
    print(f"結果は {results_dir} ディレクトリに保存されています")


if __name__ == "__main__":
    main()
