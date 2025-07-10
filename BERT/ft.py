import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import load_dataset, DatasetDict, ClassLabel
from tqdm.auto import tqdm
import evaluate
import datetime
import time
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import argparse
import tempfile

# 乱数シードの固定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 設定ファイルの読み込み
def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print("--- Configuration ---")
    print(json.dumps(config, indent=2))
    print("---------------------\n")
    return config

# データセットの読み込みと前処理
def load_and_preprocess_data(config, tokenizer):
    """CSVを読み込み、発話ペアとしてトークナイズまで行う"""
    # 1. CSVからデータを読み込む
    raw_dataset = load_dataset('csv', data_files={'train': config['train_csv']}, split='train')

    # configからラベル名を取得してClassLabelを定義
    class_names = list(config['label2id'].keys())
    # cast_column を使って 'label' カラムの型を変換する
    raw_dataset = raw_dataset.cast_column("label", ClassLabel(names=class_names))

    # 2. テキストをクリーニングする関数
    def clean_text(example):
        utr = example.get('Utterance', '') or ''
        res = example.get('Response', '') or ''
        example['Utterance'] = utr.replace('\\n', ' ')
        example['Response'] = res.replace('\\n', ' ')
        return example
    cleaned_dataset = raw_dataset.map(clean_text)

    # 3. トークナイズ関数
    def tokenize_function(examples):
        return tokenizer(
            examples['Utterance'],
            examples['Response'],
            padding='max_length',
            truncation=True,
            max_length=config['max_length']
        )

    # 4. データセット全体を一度にトークナイズ（バッチ処理で高速化）
    tokenized_dataset = cleaned_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['Utterance', 'Response']
    )
    
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format('torch')

    # 5. 訓練データと検証データに分割（層化サンプリング）
    train_val_split = tokenized_dataset.train_test_split(
        test_size=0.15, seed=config['seed'], stratify_by_column='labels'
    )
    dataset_dict = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test']
    })

    print("--- Dataset structure ---")
    print(dataset_dict)
    sample = dataset_dict['train'][0]
    print("\n--- Sample of tokenized data ---")
    print("input_ids:", sample['input_ids'].tolist())
    print("token_type_ids:", sample['token_type_ids'].tolist())
    print("decoded:", tokenizer.decode(sample['input_ids']))
    print("--------------------------------\n")
    
    return dataset_dict

# モデルの初期化
def initialize_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    num_labels = len(config['id2label'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=num_labels,
        id2label=config['id2label'],
        label2id=config['label2id']
    )
    return tokenizer, model

# 学習パラメータの設定
def configure_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config['weight_decay'],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config['learning_rate'])
    return optimizer

def configure_scheduler(optimizer, num_training_steps, config):
    num_warmup_steps = int(num_training_steps * config['num_warmup_steps_ratio'])
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler

# モデルの学習
def train(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device, config):
    num_epochs = config['num_epochs']
    progress_bar = tqdm(range(len(train_dataloader) * num_epochs))
    
    # メトリクスをループ外でロード
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    history = {
        'train_loss': [], 'train_accuracy': [],
        'val_loss': [], 'val_accuracy': [], 'val_f1': [],
        'val_precision': [], 'val_recall': []
    }
    
    best_metric_value = -float('inf') if config['metric_goal'] == 'maximize' else float('inf')
    epochs_without_improvement = 0
    
    # ベストモデルを一時ファイルに保存
    best_model_path = os.path.join(tempfile.gettempdir(), "best_model.pt")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # --- 学習フェーズ ---
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            total_train_loss += loss.item()
            loss.backward()
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)

        # --- 検証フェーズ ---
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        val_metrics = {
            'val_accuracy': accuracy_metric.compute(predictions=all_preds, references=all_labels)['accuracy'],
            'val_f1': f1_metric.compute(predictions=all_preds, references=all_labels, average='binary')['f1'],
            'val_precision': precision_metric.compute(predictions=all_preds, references=all_labels, average='binary')['precision'],
            'val_recall': recall_metric.compute(predictions=all_preds, references=all_labels, average='binary')['recall'],
            'val_loss': avg_val_loss
        }
        
        history['val_loss'].append(val_metrics['val_loss'])
        history['val_accuracy'].append(val_metrics['val_accuracy'])
        history['val_f1'].append(val_metrics['val_f1'])
        history['val_precision'].append(val_metrics['val_precision'])
        history['val_recall'].append(val_metrics['val_recall'])
        
        print(f"  Validation Loss: {val_metrics['val_loss']:.4f} | Accuracy: {val_metrics['val_accuracy']:.4f} | F1: {val_metrics['val_f1']:.4f}")

        # ベストモデルの判定と保存
        metric_to_check = val_metrics[config['metric_for_best_model']]
        if (config['metric_goal'] == 'maximize' and metric_to_check > best_metric_value) or \
           (config['metric_goal'] == 'minimize' and metric_to_check < best_metric_value):
            best_metric_value = metric_to_check
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved with {config['metric_for_best_model']}: {best_metric_value:.4f}")
        else:
            epochs_without_improvement += 1
            
        # Early Stopping
        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

    # ベストモデルのstate_dictを読み込んで返す
    best_model_state = torch.load(best_model_path)
    return history, best_model_state

# ログの保存と可視化
def save_log_and_visualize(log_data, save_datetime, config):
    log_dir = config['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # ログファイルの保存
    log_file = os.path.join(log_dir, f"training_log_{save_datetime}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    print(f"Training log saved to {log_file}")

    # 損失グラフの描画と保存
    plt.figure(figsize=(10, 5))
    plt.plot(log_data['history']['train_loss'], label='Training Loss')
    plt.plot(log_data['history']['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"loss_graph_{save_datetime}.png"))
    plt.show()

    # F1スコアグラフの描画と保存
    plt.figure(figsize=(10, 5))
    plt.plot(log_data['history']['val_f1'], label='Validation F1-Score')
    plt.title('Validation F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, f"f1_graph_{save_datetime}.png"))
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    parser.add_argument("--learning_rate", type=float, help="Override learning rate")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    args = parser.parse_args()

    config = load_config(args.config)
    # コマンドライン引数でconfigを上書き
    if args.learning_rate: config['learning_rate'] = args.learning_rate
    if args.batch_size: config['batch_size'] = args.batch_size
    
    set_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer, model = initialize_model(config)
    model.to(device)

    tokenized_datasets = load_and_preprocess_data(config, tokenizer)

    train_dataloader = DataLoader(tokenized_datasets['train'], shuffle=True, batch_size=config['batch_size'])
    val_dataloader = DataLoader(tokenized_datasets['validation'], shuffle=False, batch_size=config['batch_size'])

    optimizer = configure_optimizer(model, config)
    num_training_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = configure_scheduler(optimizer, num_training_steps, config)

    start_time = time.time()
    history, best_model_state = train(
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler, device, config
    )
    end_time = time.time()

    # ベストモデルの状態でモデルを更新
    model.load_state_dict(best_model_state)
    
    # ログデータの作成
    save_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_data = {
        "config": config,
        "training_start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "training_end_time": datetime.datetime.fromtimestamp(end_time).isoformat(),
        "training_duration_seconds": end_time - start_time,
        "dataset_info": {
            "train_samples": len(tokenized_datasets['train']),
            "validation_samples": len(tokenized_datasets['validation']),
            "label_distribution": tokenized_datasets['train'].to_pandas()['labels'].value_counts().to_dict()
        },
        "history": history
    }

    save_log_and_visualize(log_data, save_datetime, config)
    
    # モデルの保存
    model_dir = config['model_dir']
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_save_path = os.path.join(model_dir, f"best_model_{save_datetime}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path) # トークナイザーも同じ場所に保存するのが一般的
    
    print(f"Best model saved to {model_save_path}")
    print("Fine-tuning completed.")


if __name__ == "__main__":
    main()