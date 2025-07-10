import os
import torch
import json
import argparse
import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset, ClassLabel
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F

# =====================================================================================
# 1. ヘルパー関数
# =====================================================================================

def load_final_model(model_path, config):
    """保存されたstate_dict (.pt) からモデルを読み込む"""
    print(f"Loading model state_dict from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=len(config['id2label']),
        id2label=config['id2label'],
        label2id=config['label2id']
    )
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    print("Model loaded successfully.")
    return tokenizer, model

def preprocess_test_data(test_csv_path, tokenizer, config):
    """テストデータを読み込み、前処理を行う"""
    print(f"Loading and preprocessing test data from: {test_csv_path}")
    dataset = load_dataset('csv', data_files={'test': test_csv_path}, split='test')
    
    dataset = dataset.rename_column("label", "labels")
    class_names = list(config['label2id'].keys())
    dataset = dataset.cast_column("labels", ClassLabel(names=class_names))

    def tokenize_function(examples):
        return tokenizer(
            examples['Utterance'], examples['Response'],
            padding='max_length', truncation=True, max_length=config['max_length']
        )
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['Utterance', 'Response'])
    tokenized_dataset.set_format('torch')
    return tokenized_dataset

def evaluate(model, dataloader, device, original_dataset):
    """
    モデルの推論を実行し、予測と正解ラベルを返す
    さらに全サンプルの詳細な推論記録も返す
    - Utterance, Response（入力テキスト）
    - 正解ラベル
    - モデルの予測ラベル
    - 予測確率（softmax値）
    - 元CSVの全カラム
    """
    model.eval()
    model.to(device)
    all_predictions, all_references = [], []
    detailed_records = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating on Test Set")):
            labels = batch.pop('labels').to(device)
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_references.extend(labels.cpu().numpy())
            batch_size = labels.size(0)
            start_idx = i * dataloader.batch_size
            for j in range(batch_size):
                idx = start_idx + j
                orig = original_dataset[idx]
                # 予測確率をリストで保存
                prob_list = probs[j].cpu().numpy().tolist()
                # 元CSVの全カラムを記録
                record = dict(orig)
                record.update({
                    "idx": idx,
                    "Utterance": orig.get("Utterance", None),
                    "Response": orig.get("Response", None),
                    "true_label": orig["labels"] if "labels" in orig else orig.get("label", None),
                    "pred_label": int(predictions[j].cpu().numpy()),
                    "pred_probs": prob_list
                })
                detailed_records.append(record)
    return all_predictions, all_references, detailed_records

def display_and_save_results(predictions, references, config, output_dir):
    """評価結果を計算し、表示・保存する"""
    precision, recall, f1, _ = precision_recall_fscore_support(references, predictions, average='binary', zero_division=0)
    accuracy = accuracy_score(references, predictions)

    results = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1-score": f1}

    print("\n========== Final Test Set Evaluation Results ==========")
    for key, value in results.items():
        print(f"{key.capitalize():<12}: {value:.4f}")
    print("=====================================================")

    # 混同行列の描画と保存
    id2label = config['id2label']
    labels_display = [id2label[str(i)] for i in sorted(id2label.keys(), key=int)]
    cm = confusion_matrix(references, predictions)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_display)
    
    # カラーバーを非表示にし、レイアウトを調整
    disp.plot(cmap="Blues", ax=ax, colorbar=False, values_format='d')
    plt.title("Confusion Matrix on Final Test Set")
    plt.tight_layout() # ラベルの見切れを防ぐ
    
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "final_test_confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.show()

    results_path = os.path.join(output_dir, "final_test_evaluation_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Evaluation results saved to: {results_path}")

def save_detailed_predictions(detailed_records, output_dir):
    """
    全サンプルの詳細な推論記録を保存する
    """
    path = os.path.join(output_dir, "detailed_predictions.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for rec in detailed_records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")
    print(f"Detailed predictions saved to: {path}")

def save_evaluation_metadata(
    output_dir, model_pt_path, test_csv_path, config, batch_size, device, timestamp
):
    """
    評価に関する全ての情報を記録する
    - モデルのパス
    - テストCSVのパス
    - 評価時刻
    - 使用したモデル名
    - バッチサイズ
    - デバイス
    - ラベルマッピング
    - その他config
    """
    metadata = {
        "model_pt_path": model_pt_path,
        "test_csv_path": test_csv_path,
        "evaluation_timestamp": timestamp,
        "model_name": config["model_name"],
        "batch_size": batch_size,
        "device": str(device),
        "max_length": config["max_length"],
        "id2label": config["id2label"],
        "label2id": config["label2id"]
    }
    metadata_path = os.path.join(output_dir, "evaluation_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    print(f"Evaluation metadata saved to: {metadata_path}")

# =====================================================================================
# 2. main関数
# =====================================================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate the final fine-tuned model on the test set.")
    parser.add_argument("--model_pt_path", type=str, required=True, help="Path to the saved model state_dict (.pt file).")
    parser.add_argument("--test_csv_path", type=str, required=True, help="Path to the test CSV file (e.g., test_set.csv).")
    parser.add_argument("--output_dir_base", type=str, default="final_evaluation", help="Base directory to save evaluation results.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation.")
    args = parser.parse_args()

    config = {
        "model_name": "tohoku-nlp/bert-base-japanese-whole-word-masking",
        "max_length": 256,
        "id2label": {"0": "NOTIRONY", "1": "IRONY"},
        "label2id": {"NOTIRONY": 0, "IRONY": 1}
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★★★★★ ここから修正 ★★★★★
    # 固有の出力ディレクトリ名を作成
    model_name_short = os.path.basename(args.model_pt_path).replace('.pt', '')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir_base, f"{model_name_short}_{timestamp}")
    print(f"\nResults will be saved to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)  # ここで先に作成しておく

    # 評価メタデータを保存
    save_evaluation_metadata(
        output_dir=output_dir,
        model_pt_path=args.model_pt_path,
        test_csv_path=args.test_csv_path,
        config=config,
        batch_size=args.batch_size,
        device=device,
        timestamp=timestamp
    )
    # ★★★★★ ここまで修正 ★★★★★

    # 1. モデルとトークナイザーを読み込み
    tokenizer, model = load_final_model(args.model_pt_path, config)
    
    # 2. テストデータを準備
    test_dataset = preprocess_test_data(args.test_csv_path, tokenizer, config)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # 元のテストデータも取得
    original_test_dataset = load_dataset('csv', data_files={'test': args.test_csv_path}, split='test')

    # 3. 推論を実行
    predictions, references, detailed_records = evaluate(model, test_dataloader, device, original_test_dataset)
    
    # 4. 結果を表示・保存
    display_and_save_results(predictions, references, config, output_dir)
    # 詳細推論記録を保存
    save_detailed_predictions(detailed_records, output_dir)

if __name__ == "__main__":
    main()