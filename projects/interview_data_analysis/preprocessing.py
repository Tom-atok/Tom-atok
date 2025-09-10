"""
テキストデータの前処理スクリプト
インタビューデータからLDA用の文書を生成する
"""

import os
import re
import glob
from typing import List, Dict, Tuple
import MeCab
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from collections import Counter


class TextPreprocessor:
    """テキスト前処理クラス"""
    
    def __init__(self, stop_words: List[str] = None, 
                 auto_stop_words: bool = True,
                 high_freq_threshold: float = 0.8,
                 low_freq_threshold: float = 0.01,
                 target_speaker: str = "対象者"):
        """
        初期化
        
        Args:
            stop_words: 手動で指定するストップワードのリスト
            auto_stop_words: 自動ストップワード生成を使用するか
            high_freq_threshold: 高頻度単語の閾値（文書に占める割合）
            low_freq_threshold: 低頻度単語の閾値（文書に占める割合）
            target_speaker: 抽出したい発言者の名前
        """
        self.manual_stop_words = stop_words or self._get_default_stop_words()
        self.auto_stop_words = auto_stop_words
        self.high_freq_threshold = high_freq_threshold
        self.low_freq_threshold = low_freq_threshold
        self.target_speaker = target_speaker
        self.stop_words = self.manual_stop_words.copy()
        self.mecab = MeCab.Tagger()
        self.word_frequencies = None
        
    def _get_default_stop_words(self) -> List[str]:
        """デフォルトのストップワードを取得"""
        return [
            'する', 'ある', 'いる', 'なる', 'する', 'こと', 'もの', 'ため', 'とき', 'ところ',
            'の', 'に', 'は', 'を', 'が', 'で', 'と', 'から', 'まで', 'より', 'へ', 'も',
            'です', 'ます', 'でした', 'ました', 'です', 'ます', 'だ', 'である', 'です',
            'それ', 'これ', 'あれ', 'どれ', 'その', 'この', 'あの', 'どの', 'それで', 'そこで',
            'インタビュー', '対象者', '日時', '実施者', '場所', '様', 'さん', 'です', 'ます',
            'えー', 'あー', 'うーん', 'そうですね', 'はい', 'いいえ', 'えっと', 'まあ'
        ]
    
    def load_text_files(self, data_dir: str) -> Dict[str, str]:
        """
        dataディレクトリ内のテキストファイルを読み込む
        
        Args:
            data_dir: データディレクトリのパス
            
        Returns:
            ファイル名をキー、内容を値とする辞書
        """
        text_files = {}
        pattern = os.path.join(data_dir, "*.text")
        
        for file_path in glob.glob(pattern):
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    text_files[filename] = content
                print(f"読み込み完了: {filename}")
            except Exception as e:
                print(f"エラー: {filename} の読み込みに失敗しました - {e}")
                
        return text_files
    
    def extract_interview_content(self, text: str, target_speaker: str = "対象者") -> List[str]:
        """
        インタビューテキストから対象者の発言を個別に抽出
        
        Args:
            text: 元のテキスト
            target_speaker: 抽出したい発言者の名前（デフォルト: "対象者"）
            
        Returns:
            対象者の発言リスト（発言ごとに分割）
        """
        lines = text.split('\n')
        statements = []
        current_statement = []
        current_speaker = None
        
        for line in lines:
            line = line.strip()
            # 空行やメタデータ行をスキップ
            if not line or line.startswith('インタビュー'):
                continue
            
            # 発言者を識別
            speaker_match = re.match(r'^([^：]+)：', line)
            if speaker_match:
                # 前の発言を保存
                if current_statement and current_speaker and target_speaker in current_speaker:
                    statement_text = ' '.join(current_statement)
                    if statement_text.strip():
                        statements.append(statement_text)
                
                # 新しい発言を開始
                current_speaker = speaker_match.group(1).strip()
                current_statement = []
                
                # 対象者の発言かどうかをチェック
                if target_speaker in current_speaker:
                    # 発言内容を抽出
                    content = line.split('：', 1)[1].strip()
                    if content:
                        current_statement.append(content)
                continue
            
            # 前の行が対象者の発言で、現在の行が発言の続きの場合
            if current_speaker and target_speaker in current_speaker and line:
                current_statement.append(line)
        
        # 最後の発言を保存
        if current_statement and current_speaker and target_speaker in current_speaker:
            statement_text = ' '.join(current_statement)
            if statement_text.strip():
                statements.append(statement_text)
        
        return statements
    
    def calculate_word_frequencies(self, all_tokens: List[str]) -> Dict[str, Tuple[int, float]]:
        """
        全トークンの出現頻度を計算
        
        Args:
            all_tokens: 全文書のトークンリスト
            
        Returns:
            単語をキー、(出現回数, 文書頻度)を値とする辞書
        """
        # 単語の出現回数をカウント
        word_counts = Counter(all_tokens)
        total_tokens = len(all_tokens)
        
        # 文書頻度を計算（各単語が何文書に出現するか）
        # ここでは簡易的に全トークンから計算
        word_frequencies = {}
        
        for word, count in word_counts.items():
            # 出現頻度（全トークンに占める割合）
            token_frequency = count / total_tokens
            word_frequencies[word] = (count, token_frequency)
        
        return word_frequencies
    
    def generate_auto_stop_words(self, word_frequencies: Dict[str, Tuple[int, float]], 
                                n_documents: int) -> List[str]:
        """
        出現頻度に基づいて自動的にストップワードを生成
        
        Args:
            word_frequencies: 単語頻度辞書
            n_documents: 文書数
            
        Returns:
            自動生成されたストップワードのリスト
        """
        auto_stop_words = []
        
        for word, (count, token_frequency) in word_frequencies.items():
            # 高頻度単語（出現頻度が閾値を超える）
            if token_frequency > self.high_freq_threshold:
                auto_stop_words.append(word)
                print(f"高頻度ストップワード追加: {word} (頻度: {token_frequency:.3f})")
            
            # 低頻度単語（出現頻度が閾値を下回る）
            elif token_frequency < self.low_freq_threshold:
                auto_stop_words.append(word)
                print(f"低頻度ストップワード追加: {word} (頻度: {token_frequency:.3f})")
        
        return auto_stop_words
    
    def update_stop_words(self, all_tokens: List[str], n_documents: int):
        """
        出現頻度に基づいてストップワードを更新
        
        Args:
            all_tokens: 全文書のトークンリスト
            n_documents: 文書数
        """
        if not self.auto_stop_words:
            return
        
        print("出現頻度に基づいてストップワードを生成中...")
        
        # 単語頻度を計算
        self.word_frequencies = self.calculate_word_frequencies(all_tokens)
        
        # 自動ストップワードを生成
        auto_stop_words = self.generate_auto_stop_words(self.word_frequencies, n_documents)
        
        # 手動ストップワードと自動ストップワードを結合
        self.stop_words = list(set(self.manual_stop_words + auto_stop_words))
        
        print(f"ストップワード更新完了:")
        print(f"  手動ストップワード: {len(self.manual_stop_words)}個")
        print(f"  自動ストップワード: {len(auto_stop_words)}個")
        print(f"  合計ストップワード: {len(self.stop_words)}個")
    
    def tokenize_japanese(self, text: str) -> List[str]:
        """
        日本語テキストを形態素解析してトークン化
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークンのリスト
        """
        tokens = []
        node = self.mecab.parseToNode(text)
        
        while node:
            if node.surface:
                # 品詞情報を取得
                pos = node.feature.split(',')[0]
                # 名詞、動詞、形容詞のみを抽出
                if pos in ['名詞', '動詞', '形容詞']:
                    # ストップワードでない場合のみ追加
                    if node.surface not in self.stop_words:
                        # 長さが1文字以上のもののみ
                        if len(node.surface) > 1:
                            tokens.append(node.surface)
            node = node.next
            
        return tokens
    
    def preprocess_texts(self, text_files: Dict[str, str]) -> Tuple[List[str], List[Dict]]:
        """
        テキストファイルを前処理してLDA用の文書リストを作成（発言ごとに分割）
        
        Args:
            text_files: ファイル名をキー、内容を値とする辞書
            
        Returns:
            (前処理済み文書のリスト, 発言メタデータのリスト)
        """
        # 第1段階: 初期の形態素解析（ストップワード除去前）
        initial_tokens_list = []
        valid_statements = []
        statement_metadata = []
        
        for filename, content in text_files.items():
            print(f"初期前処理中: {filename}")
            
            # 対象者の発言を個別に抽出
            statements = self.extract_interview_content(content, target_speaker=self.target_speaker)
            
            if not statements:
                print(f"警告: {filename} から対象者の有効な発言を抽出できませんでした")
                continue
            
            print(f"  {len(statements)}個の発言を抽出")
            
            for i, statement in enumerate(statements):
                # 初期の形態素解析（ストップワード除去前）
                initial_tokens = self.tokenize_japanese_initial(statement)
                
                if len(initial_tokens) < 3:  # トークンが少なすぎる場合はスキップ
                    print(f"  警告: 発言{i+1}のトークン数が少なすぎます ({len(initial_tokens)}個)")
                    continue
                
                initial_tokens_list.append(initial_tokens)
                valid_statements.append(statement)
                statement_metadata.append({
                    'filename': filename,
                    'statement_id': i + 1,
                    'original_text': statement,
                    'token_count': len(initial_tokens)
                })
                
                print(f"  発言{i+1}: {len(initial_tokens)}個のトークン")
        
        if not initial_tokens_list:
            print("有効な発言がありません")
            return [], []
        
        # 全トークンを結合して頻度分析
        all_tokens = []
        for tokens in initial_tokens_list:
            all_tokens.extend(tokens)
        
        # 自動ストップワードを生成
        self.update_stop_words(all_tokens, len(valid_statements))
        
        # 第2段階: ストップワード除去後の最終処理
        processed_docs = []
        final_metadata = []
        
        for i, (statement, metadata) in enumerate(zip(valid_statements, statement_metadata)):
            print(f"最終前処理中: {metadata['filename']} - 発言{metadata['statement_id']}")
            
            # ストップワード除去後の形態素解析
            final_tokens = self.tokenize_japanese(statement)
            
            if len(final_tokens) < 2:  # ストップワード除去後もトークンが少なすぎる場合はスキップ
                print(f"  警告: 最終トークン数が少なすぎます ({len(final_tokens)}個)")
                continue
            
            # トークンを文字列に結合
            processed_doc = ' '.join(final_tokens)
            processed_docs.append(processed_doc)
            
            # メタデータを更新
            metadata['final_token_count'] = len(final_tokens)
            metadata['processed_text'] = processed_doc
            final_metadata.append(metadata)
            
            print(f"  最終完了: {len(final_tokens)}個のトークン")
        
        print(f"\n総発言数: {len(processed_docs)}個")
        return processed_docs, final_metadata
    
    def tokenize_japanese_initial(self, text: str) -> List[str]:
        """
        初期の形態素解析（ストップワード除去前）
        
        Args:
            text: 入力テキスト
            
        Returns:
            トークンのリスト
        """
        tokens = []
        node = self.mecab.parseToNode(text)
        
        while node:
            if node.surface:
                # 品詞情報を取得
                pos = node.feature.split(',')[0]
                # 名詞、動詞、形容詞のみを抽出
                if pos in ['名詞', '動詞', '形容詞']:
                    # 長さが1文字以上のもののみ
                    if len(node.surface) > 1:
                        tokens.append(node.surface)
            node = node.next
            
        return tokens
    
    def create_document_term_matrix(self, documents: List[str], 
                                  min_df: int = 2, 
                                  max_df: float = 0.8) -> tuple:
        """
        文書-単語行列を作成
        
        Args:
            documents: 前処理済み文書のリスト
            min_df: 最小文書頻度
            max_df: 最大文書頻度
            
        Returns:
            (文書-単語行列, 特徴語リスト)のタプル
        """
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            token_pattern=r'\S+'  # スペース区切りのトークン
        )
        
        doc_term_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"文書数: {doc_term_matrix.shape[0]}")
        print(f"語彙数: {doc_term_matrix.shape[1]}")
        
        return doc_term_matrix, feature_names
    
    def save_processed_data(self, documents: List[str], 
                          doc_term_matrix, 
                          feature_names: List[str], 
                          statement_metadata: List[Dict],
                          output_dir: str):
        """
        前処理済みデータを保存
        
        Args:
            documents: 前処理済み文書のリスト
            doc_term_matrix: 文書-単語行列
            feature_names: 特徴語リスト
            statement_metadata: 発言メタデータのリスト
            output_dir: 出力ディレクトリ
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 文書を保存
        with open(os.path.join(output_dir, 'processed_documents.txt'), 'w', encoding='utf-8') as f:
            for i, (doc, metadata) in enumerate(zip(documents, statement_metadata)):
                f.write(f"Statement {i+1} ({metadata['filename']} - 発言{metadata['statement_id']}):\n")
                f.write(f"Original: {metadata['original_text']}\n")
                f.write(f"Processed: {doc}\n")
                f.write(f"Tokens: {metadata['final_token_count']}\n\n")
        
        # 文書-単語行列を保存
        with open(os.path.join(output_dir, 'doc_term_matrix.pkl'), 'wb') as f:
            pickle.dump(doc_term_matrix, f)
        
        # 特徴語リストを保存
        with open(os.path.join(output_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(feature_names, f)
        
        # 発言メタデータを保存
        with open(os.path.join(output_dir, 'statement_metadata.pkl'), 'wb') as f:
            pickle.dump(statement_metadata, f)
        
        # 発言メタデータをCSVで保存
        import pandas as pd
        metadata_df = pd.DataFrame(statement_metadata)
        metadata_df.to_csv(os.path.join(output_dir, 'statement_metadata.csv'), 
                          index=False, encoding='utf-8')
        
        # ストップワード情報を保存
        self.save_stop_words_info(output_dir)
        
        print(f"前処理済みデータを {output_dir} に保存しました")
    
    def save_stop_words_info(self, output_dir: str):
        """
        ストップワード情報を保存
        
        Args:
            output_dir: 出力ディレクトリ
        """
        # ストップワードリストを保存
        with open(os.path.join(output_dir, 'stop_words.txt'), 'w', encoding='utf-8') as f:
            f.write("=== ストップワード一覧 ===\n\n")
            f.write(f"手動ストップワード ({len(self.manual_stop_words)}個):\n")
            for word in sorted(self.manual_stop_words):
                f.write(f"  - {word}\n")
            
            f.write(f"\n自動ストップワード ({len(self.stop_words) - len(self.manual_stop_words)}個):\n")
            auto_stop_words = [w for w in self.stop_words if w not in self.manual_stop_words]
            for word in sorted(auto_stop_words):
                f.write(f"  - {word}\n")
        
        # 単語頻度情報を保存
        if self.word_frequencies:
            with open(os.path.join(output_dir, 'word_frequencies.csv'), 'w', encoding='utf-8') as f:
                f.write("word,count,frequency\n")
                for word, (count, frequency) in sorted(self.word_frequencies.items(), 
                                                     key=lambda x: x[1][1], reverse=True):
                    f.write(f"{word},{count},{frequency:.6f}\n")
        
        # 設定情報を保存
        with open(os.path.join(output_dir, 'preprocessing_config.txt'), 'w', encoding='utf-8') as f:
            f.write("=== 前処理設定 ===\n\n")
            f.write(f"対象発言者: {self.target_speaker}\n")
            f.write(f"自動ストップワード生成: {self.auto_stop_words}\n")
            f.write(f"高頻度閾値: {self.high_freq_threshold}\n")
            f.write(f"低頻度閾値: {self.low_freq_threshold}\n")
            f.write(f"総ストップワード数: {len(self.stop_words)}\n")


def main():
    """メイン処理"""
    # 設定
    data_dir = "data"
    output_dir = "processed_data"
    
    # 前処理器を初期化（自動ストップワード生成を有効化）
    preprocessor = TextPreprocessor(
        auto_stop_words=True,      # 自動ストップワード生成を有効
        high_freq_threshold=0.8,   # 高頻度単語の閾値（80%以上の文書に出現）
        low_freq_threshold=0.01,   # 低頻度単語の閾値（1%未満の文書にのみ出現）
        target_speaker="対象者"     # 対象者の発言のみを抽出
    )
    
    # テキストファイルを読み込み
    print("テキストファイルを読み込み中...")
    text_files = preprocessor.load_text_files(data_dir)
    
    if not text_files:
        print("読み込めるテキストファイルがありません")
        return
    
    # 前処理を実行
    print("前処理を実行中...")
    processed_docs, statement_metadata = preprocessor.preprocess_texts(text_files)
    
    if not processed_docs:
        print("前処理済み文書が生成されませんでした")
        return
    
    # 文書-単語行列を作成
    print("文書-単語行列を作成中...")
    doc_term_matrix, feature_names = preprocessor.create_document_term_matrix(processed_docs)
    
    # 結果を保存
    preprocessor.save_processed_data(processed_docs, doc_term_matrix, feature_names, statement_metadata, output_dir)
    
    print("前処理が完了しました！")


if __name__ == "__main__":
    main()
