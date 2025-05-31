# Irony NLP

日本語対話の皮肉検出(分類)に関する研究のレポジトリです。  
コーパス収集・機械学習用の一部のコードが入っています。

### 主要ファイル

- [`BERT_detection/finetune_bert.py`](BERT_detection/finetune_bert.py):  
BERTモデルをファインチューニングするためのスクリプト。
- [`BERT_detection/eval_classification_model.py`](BERT_detection/eval_classification_model.py):  
分類モデルの評価を行うスクリプト。
- [`corpus_collection/corpus_generate/corpus_generate.ipynb`](corpus_collection/corpus_generate/corpus_generate.ipynb):  
(試験的)LLMによるコーパスの拡張生成のJupyter Notebook。
- [`LLM_detection/llmdetection_fewshot.ipynb`](LLM_detection/llmdetection_fewshot.ipynb):  
Few-shotでのLLMによる検出を行うためのJupyter Notebook。
- [`LLM_detection/llmdetection_oneshot.ipynb`](LLM_detection/llmdetection_oneshot.ipynb):  
One-shotでのLLMによる検出を行うためのJupyter Notebook。

### 関連レポジトリ
- テキストデータラベリングのためのWebUI  
https://github.com/10nm/labeling-webui
- CSVの分割・結合の為のWebアプリケーション  
https://github.com/10nm/csvsplitter-webui

### 予定
- すべての.mdファイルをレポジトリに移設する
- BERTの新しいデータセットでの再学習・再評価
- LLMの再評価
