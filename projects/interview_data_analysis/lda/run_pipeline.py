"""
LDA分析パイプライン実行スクリプト
前処理からLDA分析まで一括で実行する
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def check_dependencies():
    """必要な依存関係がインストールされているかチェック"""
    required_packages = [
        'sklearn', 'pandas', 'numpy', 'matplotlib', 
        'seaborn', 'wordcloud', 'mecab'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'mecab':
                import MeCab
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("以下のパッケージが不足しています:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n以下のコマンドでインストールしてください:")
        print("pip install -r requirements.txt")
        return False
    
    return True


def run_preprocessing():
    """前処理を実行"""
    print("=" * 50)
    print("前処理を開始します...")
    print("=" * 50)
    
    try:
        # preprocessing.pyを実行
        result = subprocess.run([sys.executable, 'preprocessing.py'], 
                              capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("前処理が正常に完了しました")
            print(result.stdout)
        else:
            print("前処理でエラーが発生しました:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"前処理の実行中にエラーが発生しました: {e}")
        return False
    
    return True


def run_lda_analysis(n_topics=None):
    """LDA分析を実行"""
    print("=" * 50)
    print("LDA分析を開始します...")
    print("=" * 50)
    
    try:
        # main.pyを実行
        cmd = [sys.executable, 'main.py']
        if n_topics:
            cmd.extend(['--n_topics', str(n_topics)])
            
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            print("LDA分析が正常に完了しました")
            print(result.stdout)
        else:
            print("LDA分析でエラーが発生しました:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"LDA分析の実行中にエラーが発生しました: {e}")
        return False
    
    return True


def check_data_directory():
    """dataディレクトリとテキストファイルの存在をチェック"""
    data_dir = Path("../data")  # 上位ディレクトリのdataフォルダを参照
    
    if not data_dir.exists():
        print("エラー: dataディレクトリが見つかりません")
        return False
    
    text_files = list(data_dir.glob("*.text"))
    if not text_files:
        print("エラー: dataディレクトリに.textファイルが見つかりません")
        return False
    
    print(f"発見されたテキストファイル: {len(text_files)}個")
    for file in text_files:
        print(f"  - {file.name}")
    
    return True


def create_directories():
    """必要なディレクトリを作成"""
    directories = ["processed_data", "results"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"ディレクトリ作成/確認: {directory}")


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(description='LDA分析パイプラインを実行')
    parser.add_argument('--n_topics', type=int, default=None,
                       help='トピック数（指定しない場合は自動で最適化）')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='前処理をスキップ（既に前処理済みの場合）')
    parser.add_argument('--skip_lda', action='store_true',
                       help='LDA分析をスキップ（前処理のみ実行）')
    
    args = parser.parse_args()
    
    print("LDA分析パイプラインを開始します")
    print("=" * 60)
    
    # 依存関係をチェック
    print("依存関係をチェック中...")
    if not check_dependencies():
        sys.exit(1)
    print("依存関係チェック完了")
    
    # データディレクトリをチェック
    print("\nデータディレクトリをチェック中...")
    if not check_data_directory():
        sys.exit(1)
    
    # 必要なディレクトリを作成
    print("\n必要なディレクトリを作成中...")
    create_directories()
    
    # 前処理を実行
    if not args.skip_preprocessing:
        if not run_preprocessing():
            print("前処理でエラーが発生したため、パイプラインを終了します")
            sys.exit(1)
    else:
        print("前処理をスキップしました")
    
    # LDA分析を実行
    if not args.skip_lda:
        if not run_lda_analysis(args.n_topics):
            print("LDA分析でエラーが発生したため、パイプラインを終了します")
            sys.exit(1)
    else:
        print("LDA分析をスキップしました")
    
    print("\n" + "=" * 60)
    print("パイプラインが正常に完了しました！")
    print("=" * 60)
    
    # 結果の概要を表示
    results_dir = Path("results")
    if results_dir.exists():
        print("\n生成されたファイル:")
        for file in results_dir.iterdir():
            if file.is_file():
                print(f"  - {file.name}")
    
    print("\n分析結果は 'results' ディレクトリに保存されています")


if __name__ == "__main__":
    main()
