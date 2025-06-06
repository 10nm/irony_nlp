import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm
import evaluate
import datetime
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.optim import AdamW


# 乱数シードの固定
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 設定ファイルの読み込み
def load_config(config_file):
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


# データの前処理
def format_string(string):
    return string.replace('\\n', ' ')


def combine_sentences(utr, res):
    return f"{format_string(utr)} [SEP] {format_string(res)}"


def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df['text'] = df.apply(
        lambda x: combine_sentences(x['Utterance'], x['Response']), axis=1)
    df = df.drop(columns=['Utterance', 'Response'])
    df = df.sample(frac=1, random_state=config['seed']).reset_index(drop=True)
    return df


# データセットの作成
class TextDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# データローダーの作成
def create_dataloaders(texts, labels, tokenizer, max_length, batch_size):
    dataset = TextDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


# モデルの初期化
def initialize_model(model_name, num_labels, id2label, label2id):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    return tokenizer, model


# 学習パラメータの設定
def configure_optimizer(model, learning_rate, weight_decay):
    no_decay = ["bias", "LayerNorm.weight"]
    named_params = list(model.named_parameters())  # キャッシュ
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    return optimizer


def configure_scheduler(optimizer, num_training_steps, num_warmup_steps_ratio):
    num_warmup_steps = int(num_training_steps * num_warmup_steps_ratio)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return lr_scheduler


# 学習情報の記録
def get_training_info(config):
    training_info = {}
    training_info['start_time'] = datetime.datetime.now().isoformat()
    training_info['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    training_info['model_name'] = config['model_name']
    training_info['max_length'] = config['max_length']
    training_info['batch_size'] = config['batch_size']
    training_info['num_epochs'] = config['num_epochs']
    training_info['learning_rate'] = config['learning_rate']
    training_info['weight_decay'] = config['weight_decay']
    training_info['num_warmup_steps'] = int(config['num_epochs'] * len(train_dataloader) * config['num_warmup_steps_ratio'])
    training_info['id2label'] = config['id2label']
    training_info['label2id'] = config['label2id']
    return training_info


# モデルの学習
def train(model, train_dataloader, val_dataloader, optimizer, lr_scheduler, num_epochs, device, early_stopping_patience):
    progress_bar = tqdm(range(len(train_dataloader) * num_epochs))
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model = None

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        train_metric = evaluate.load("accuracy")
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss_val = outputs.loss
            total_loss += loss_val.item()
            loss_val.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            predictions = torch.argmax(outputs.logits, dim=-1)
            train_metric.add_batch(
                predictions=predictions, references=batch['labels'])
        avg_train_loss = total_loss / len(train_dataloader)
        train_metric = train_metric.compute()
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_metric['accuracy'])

        # 検証ループ
        model.eval()
        total_eval_loss = 0
        val_progress_bar = tqdm(range(len(val_dataloader)))
        metric = evaluate.load("accuracy")
        for batch in val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
                loss_val = outputs.loss
                total_eval_loss += loss_val.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions,
                             references=batch['labels'])
            val_progress_bar.update(1)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        eval_metric = metric.compute()
        val_losses.append(avg_val_loss)
        val_accuracies.append(eval_metric['accuracy'])

        print(f"epoch {epoch+1}: train_loss: {avg_train_loss:.4f}, val_loss: {
            avg_val_loss:.4f}, accuracy: {eval_metric['accuracy']:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_without_improvement = 0
            best_model = model.state_dict()
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch+1}."
            )
            break
    return train_accuracies, val_accuracies, train_losses, val_losses, best_model


# モデルの評価
def evaluate_model(model, eval_dataloader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    eval_progress_bar = tqdm(range(len(eval_dataloader)))
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())
        eval_progress_bar.update(1)

    y_true = all_labels
    y_preds = all_predictions

    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.flatten()

    accuracy_eval = (tp + tn) / (tp + tn + fp +
                                 fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / \
        (precision + recall) if (precision + recall) > 0 else 0
    return y_true, y_preds, cm, accuracy_eval, precision, recall, f1


# 評価結果の表示
def display_evaluation_metrics(y_true, y_preds, cm, accuracy_eval, precision, recall, f1, save_datetime, log_dir):
    labels = ['NOTIRONY', 'IRONY']
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Confusion Matrix")
    fig_save_path = os.path.join(
        log_dir, f"confusion_matrix_{save_datetime}.png")
    plt.savefig(fig_save_path)
    plt.show()

    print(f"Accuracy : {accuracy_eval:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall : {recall:.4f}")
    print(f"F1 : {f1:.4f}")


# ログの保存
def save_training_log_json(training_info, train_results, eval_results, raw_answers, save_datetime, log_dir):
    log_file = os.path.join(log_dir, f"training_log_{save_datetime}.json")
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump({
            'training_info': training_info,
            'train_results': train_results,
            'raw_answers': raw_answers,
            'eval_results': eval_results
        }, f, ensure_ascii=False, indent=4)

    print(f"Training log saved to {log_file}")


# メイン処理
def main():
    # 設定ファイルの読み込み
    config_file = 'config.json'
    config = load_config(config_file)

    # 乱数シードの固定
    set_seed(config['seed'])

    # デバイスの設定
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # データセットの準備
    df = load_and_preprocess_data(config['train_csv'])
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=config['seed']
    )

    # モデルとトークナイザーの初期化
    tokenizer, model = initialize_model(
        config['model_name'], 2, config['id2label'], config['label2id'])

    # データローダーの作成
    train_dataloader = create_dataloaders(
        train_texts, train_labels, tokenizer, config['max_length'], config['batch_size'])
    val_dataloader = create_dataloaders(
        val_texts, val_labels, tokenizer, config['max_length'], config['batch_size'])

    # モデルをGPUに転送
    model.to(DEVICE)

    # 学習パラメータの設定
    optimizer = configure_optimizer(
        model, config['learning_rate'], config['weight_decay'])
    num_training_steps = config['num_epochs'] * len(train_dataloader)
    lr_scheduler = configure_scheduler(
        optimizer, num_training_steps, config['num_warmup_steps_ratio']
    )
    training_info = get_training_info(config)

    # モデルの学習
    train_accuracies, val_accuracies, train_losses, val_losses, best_model = train(
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler, config['num_epochs'], DEVICE, config['early_stopping_patience']
    )

    model.load_state_dict(best_model)

    end_time = datetime.datetime.now().isoformat()
    training_info['end_time'] = end_time
    train_time = datetime.datetime.strptime(end_time, '%Y-%m-%dT%H:%M:%S.%f') - \
        datetime.datetime.strptime(
            training_info['start_time'], '%Y-%m-%dT%H:%M:%S.%f')
    training_info['train_time'] = str(train_time)

    train_results = {
        'train_accuracy': train_accuracies,
        'train_loss': train_losses,
        'val_accuracy': val_accuracies,
        'val_loss': val_losses
    }

    # 評価用データセットの準備
    df_eva = load_and_preprocess_data(config['eval_csv'])
    eval_texts = df_eva['text'].tolist()
    eval_labels = df_eva['label'].tolist()

    # 評価用データローダーを作成
    eval_dataloader = create_dataloaders(
        eval_texts, eval_labels, tokenizer, config['max_length'], config['batch_size'])

    # モデルの評価
    y_true, y_preds, cm, accuracy_eval, precision, recall, f1 = evaluate_model(
        model, eval_dataloader, DEVICE
    )

    # 評価結果の表示
    save_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = config['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    display_evaluation_metrics(
        y_true, y_preds, cm, accuracy_eval, precision, recall, f1, save_datetime, log_dir
    )

    raw_answers = {
        'TruePositive': str(cm[1, 1]),
        'TrueNegative': str(cm[0, 0]),
        'FalsePositive': str(cm[0, 1]),
        'FalseNegative': str(cm[1, 0])
    }

    eval_results = {
        'accuracy': accuracy_eval,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # ログの保存
    save_training_log_json(training_info, train_results,
                           eval_results, raw_answers, save_datetime, log_dir)

    # モデルの保存
    model_dir = config['model_dir']
    model_save_path = os.path.join(model_dir, f"model_{save_datetime}")
    tokenizer_save_path = os.path.join(
        model_dir, f"tokenizer_{save_datetime}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print("Fine-tuning completed.")


if __name__ == "__main__":
    # 乱数シードの固定
    SEED = 42
    set_seed(SEED)

    main()