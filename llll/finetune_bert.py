import random
import numpy as np
import torch
import pandas as pd
from IPython.display import display
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
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# パラメータの設定
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_EPOCHS = 6  # エポック数を増やして早期打ち切りを有効に
LEARNING_RATE = 5e-7  # 学習率を調整
WEIGHT_DECAY = 0.001  # 重み減衰を調整
NUM_WARMUP_STEPS_RATIO = 0.1  # 全学習ステップの10%をウォームアップに利用
EARLY_STOPPING_PATIENCE = 3

ID2LABEL = {0: 'NOTIRONY', 1: 'IRONY'}
LABEL2ID = {'NOTIRONY': 0, 'IRONY': 1}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
display(DEVICE)


# データの前処理
def format_string(string):
    return string.replace('\\n', ' ')


def combine_sentences(utr, res):
    return f"{format_string(utr)} [SEP] {format_string(res)}"


def load_and_preprocess_data(csv_file_irony, csv_file_not_irony):
    df_label_irony = pd.read_csv(csv_file_irony)

    if 'conv' in df_label_irony.columns:
        df_label_irony['Response'] = df_label_irony['conv'].combine_first(
            df_label_irony['Response'])
        df_label_irony = df_label_irony.drop(columns=['conv'])

    df_label_not = pd.read_csv(csv_file_not_irony)

    df_label_irony['text'] = df_label_irony.apply(
        lambda x: combine_sentences(x['Utterance'], x['Response']), axis=1)
    df_label_not['text'] = df_label_not.apply(
        lambda x: combine_sentences(x['Utterance'], x['Response']), axis=1)

    df_label_irony = df_label_irony.drop(columns=['Utterance', 'Response'])
    df_label_not = df_label_not.drop(columns=['Utterance', 'Response'])

    df = pd.concat([df_label_irony, df_label_not], ignore_index=True)
    df['labels'] = [1] * len(df_label_irony) + [0] * len(df_label_not)
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
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
    print(weight_decay),
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
def get_training_info(model_name, max_length, batch_size, num_epochs, learning_rate, weight_decay, num_warmup_steps):
    training_info = {}
    training_info['start_time'] = datetime.datetime.now().isoformat()
    training_info['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    training_info['model_name'] = model_name
    training_info['max_length'] = max_length
    training_info['batch_size'] = batch_size
    training_info['num_epochs'] = num_epochs
    training_info['learning_rate'] = learning_rate
    training_info['weight_decay'] = weight_decay
    training_info['num_warmup_steps'] = num_warmup_steps
    training_info['id2label'] = ID2LABEL
    training_info['label2id'] = LABEL2ID
    return training_info


# バッチサイズの動的な調整
def adjust_batch_size(dataloader, model, device, batch_size):
    while True:
        try:
            batch = next(iter(dataloader))
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size = batch_size // 2
                print(f"Reduced batch size to {batch_size}")
                if batch_size == 0:
                    raise RuntimeError("Batch size reached 0. Cannot proceed.")
                dataloader = create_dataloaders(
                    dataloader.dataset.texts, dataloader.dataset.labels,
                    dataloader.dataset.tokenizer, dataloader.dataset.max_length, batch_size
                )
            else:
                raise


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
    # データセットの準備
    df = load_and_preprocess_data('./source_ironic.csv',
                                  './source_not_ironic.csv')
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['labels'].tolist(), test_size=0.2, random_state=SEED
    )

    # モデルとトークナイザーの初期化
    model_name = "tohoku-nlp/bert-base-japanese-whole-word-masking"
    tokenizer, model = initialize_model(model_name, 2, ID2LABEL, LABEL2ID)

    # データローダーの作成
    train_dataloader = create_dataloaders(
        train_texts, train_labels, tokenizer, MAX_LENGTH, BATCH_SIZE)
    val_dataloader = create_dataloaders(
        val_texts, val_labels, tokenizer, MAX_LENGTH, BATCH_SIZE)

    # モデルをGPUに転送
    model.to(DEVICE)

    # バッチサイズを調整
    adjusted_batch_size = adjust_batch_size(
        train_dataloader, model, DEVICE, BATCH_SIZE)

    # 再度データローダーを作成
    train_dataloader = create_dataloaders(
        train_texts, train_labels, tokenizer, MAX_LENGTH, adjusted_batch_size)
    val_dataloader = create_dataloaders(
        val_texts, val_labels, tokenizer, MAX_LENGTH, adjusted_batch_size)

    # 学習パラメータの設定
    optimizer = configure_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)
    lr_scheduler = configure_scheduler(
        optimizer, num_training_steps, NUM_WARMUP_STEPS_RATIO
    )
    training_info = get_training_info(
        model_name, MAX_LENGTH, adjusted_batch_size, NUM_EPOCHS, LEARNING_RATE,
        WEIGHT_DECAY, int(num_training_steps * NUM_WARMUP_STEPS_RATIO)
    )

    # モデルの学習
    train_accuracies, val_accuracies, train_losses, val_losses, best_model = train(
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler, NUM_EPOCHS, DEVICE, EARLY_STOPPING_PATIENCE
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
    df_eva = load_and_preprocess_data('./eva_ironic.csv',
                                      './eva_not_ironic.csv')
    eval_texts = df_eva['text'].tolist()
    eval_labels = df_eva['labels'].tolist()

    # 評価用データローダーを作成
    eval_dataloader = create_dataloaders(
        eval_texts, eval_labels, tokenizer, MAX_LENGTH, adjusted_batch_size)

    # モデルの評価
    y_true, y_preds, cm, accuracy_eval, precision, recall, f1 = evaluate_model(
        model, eval_dataloader, DEVICE
    )

    # 評価結果の表示
    save_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "training_logs"
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
    model_dir = "models"
    model_save_path = os.path.join(model_dir, f"model_{save_datetime}")
    tokenizer_save_path = os.path.join(
        model_dir, f"tokenizer_{save_datetime}")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(tokenizer_save_path)

    print("Fine-tuning completed.")


if __name__ == "__main__":
    main()
