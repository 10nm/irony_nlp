import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import datetime
import numpy as np

# 定数設定
MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ADJUST_BATCH_SIZE = BATCH_SIZE
NUM_LABELS = 2 # ラベル数（二値分類）

# データセットの定義
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# データのロードと前処理
def load_and_preprocess_data(ironic_path, not_ironic_path):
    df_ironic = pd.read_csv(ironic_path)
    df_not_ironic = pd.read_csv(not_ironic_path)
    
    df_ironic['labels'] = 1
    df_not_ironic['labels'] = 0

    #utteranceとresponseを連結、間に[SEP]トークンを挿入
    df_ironic['text'] = df_ironic['Utterance'] + " [SEP] " + df_ironic['Response']
    df_not_ironic['text'] = df_not_ironic['Utterance'] + " [SEP] " + df_not_ironic['Response']

    df = pd.concat([df_ironic, df_not_ironic], ignore_index=True)
    df = df.sample(frac=1, random_state=42)
    df = df.dropna()

    return df

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

# データローダーの作成
def create_dataloaders(texts, labels, tokenizer, max_length, batch_size):
    dataset = TextDataset(texts, labels, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

# 混同行列をプロットして保存する関数
def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 ha="center", va="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path) # ここで保存
    plt.close() # メモリリークを防ぐためにclose()

# 評価結果の表示
def display_evaluation_metrics(y_true, y_preds, cm, accuracy_eval, precision, recall, f1, save_datetime, log_dir):
    metrics_log_file = os.path.join(log_dir, f"{save_datetime}_evaluation_metrics.txt")
    cm_image_path = os.path.join(log_dir, f"{save_datetime}_confusion_matrix.png")
    metrics_log_file = os.path.join(log_dir, f"{save_datetime}_evaluation_metrics.txt")
    with open(metrics_log_file, "w") as f:
        f.write("Evaluation Metrics:\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Accuracy: {accuracy_eval:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Evaluation metrics saved to: {metrics_log_file}")
    print(f"Confusion Matrix:\n{cm}")
    print(f"Accuracy: {accuracy_eval:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # 混同行列のプロットと保存
    plot_confusion_matrix(cm, classes=['Not Ironic', 'Ironic'], save_path=cm_image_path, title="Confusion Matrix")
    print(f"Confusion matrix image saved to: {cm_image_path}")


def main():
    # 保存されたモデルのパス
    MODEL_SAVE_PATH = "/home/wslnm/irony_nlp/llll/models/model_20250125_164236"  
    TOKENIZER_SAVE_PATH = "/home/wslnm/irony_nlp/llll/models/tokenizer_20250125_164236" 

    # 保存されたtokenizerをロード
    loaded_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_SAVE_PATH)

    # 保存されたモデルをロード
    loaded_model = AutoModelForSequenceClassification.from_pretrained(MODEL_SAVE_PATH).to(DEVICE)

    # 評価用データセットの準備
    df_eva = load_and_preprocess_data('./eva_ironic.csv',
                                      './eva_not_ironic.csv')
    eval_texts = df_eva['text'].tolist()
    eval_labels = df_eva['labels'].tolist()

    # 評価用データローダーを作成
    eval_dataloader = create_dataloaders(
        eval_texts, eval_labels, loaded_tokenizer, MAX_LENGTH, ADJUST_BATCH_SIZE)

    # モデルの評価
    y_true, y_preds, cm, accuracy_eval, precision, recall, f1 = evaluate_model(
        loaded_model, eval_dataloader, DEVICE
    )

    # 評価結果の表示
    save_datetime = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    display_evaluation_metrics(
        y_true, y_preds, cm, accuracy_eval, precision, recall, f1, save_datetime, log_dir
    )

if __name__ == "__main__":
    main()