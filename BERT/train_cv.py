import os
import gc
import json
import random
import numpy as np
import pandas as pd
import torch
import evaluate
import datetime
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import Dataset, ClassLabel
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================================
# 1. 基本設定とヘルパー関数
# =====================================================================================
def set_seed(seed):
    """乱数シードを固定する"""  
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def initialize_model(config):
    """モデルとトークナイザーを初期化する"""
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=len(config['id2label']),
        id2label=config['id2label'],
        label2id=config['label2id'],
    )
    return tokenizer, model

def preprocess_data(df, tokenizer, config):
    """DataFrameをトークナイズ済みのHugging Face Datasetに変換する"""
    dataset = Dataset.from_pandas(df)
    class_names = list(config['label2id'].keys())
    dataset = dataset.cast_column("label", ClassLabel(names=class_names))
    dataset = dataset.rename_column("label", "labels")

    def tokenize_function(examples):
        return tokenizer(
            examples['Utterance'],
            examples['Response'],
            padding='max_length', truncation=True, max_length=config['max_length']
        )
    tokenized_dataset = dataset.map(
        tokenize_function, batched=True,
        remove_columns=['Utterance', 'Response', '__index_level_0__']
    )
    tokenized_dataset.set_format('torch')
    return tokenized_dataset

# =====================================================================================
# Focal Loss 実装
# =====================================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for binary/multiclass classification.
    Args:
        alpha (float or list): class weight. If float, binary classification. If list, multiclass.
        gamma (float): focusing parameter.
        reduction (str): 'mean', 'sum', or 'none'
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: (batch, num_classes)
        targets: (batch,) int64
        """
        if logits.size(-1) == 1 or logits.size(-1) == 2:
            # Binary classification (assume logits shape [B,2])
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            if isinstance(self.alpha, (float, int)):
                at = torch.ones_like(targets, dtype=logits.dtype, device=logits.device) * self.alpha
                at = torch.where(targets == 1, at, 1 - at)
            else:
                at = torch.tensor([self.alpha[i] for i in targets.cpu().numpy()], device=logits.device, dtype=logits.dtype)
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            # Multiclass
            ce_loss = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                at = torch.tensor([self.alpha[i] for i in targets.cpu().numpy()], device=logits.device, dtype=logits.dtype)
            else:
                at = 1.0
            focal_loss = at * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# =====================================================================================
# 2. グラフ描画と結果保存の関数
# =====================================================================================
def plot_fold_history(history, fold, output_dir):
    """個別のFoldの学習履歴をプロットし、保存する"""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    epochs = range(1, len(history['train_loss']) + 1)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history['train_loss'], 'o-', color='tab:blue', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o--', color='tab:cyan', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='tab:red')
    ax2.plot(epochs, history['val_f1'], 's-', color='tab:red', label='Validation F1')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    
    fig.suptitle(f'Fold {fold+1} - Training History')
    fig.legend(loc='upper right', bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    save_path = os.path.join(output_dir, f'fold_{fold+1}_history.png')
    plt.savefig(save_path)
    plt.close(fig)

def plot_cv_summary(all_histories, best_fold_index, output_dir):
    """CV全体のサマリーグラフ（Val Loss, Val F1）をプロットし、保存する"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Validation Loss のプロット
    for i, history in enumerate(all_histories):
        epochs = range(1, len(history['val_loss']) + 1)
        if i == best_fold_index:
            ax1.plot(epochs, history['val_loss'], 'o-', linewidth=2.5, label=f'Best Fold ({i+1})')
        else:
            ax1.plot(epochs, history['val_loss'], '.-', linewidth=1.0, alpha=0.6, label=f'Fold {i+1}')
    ax1.set_ylabel('Validation Loss')
    ax1.set_title('Validation Loss across Folds')
    ax1.grid(True)
    ax1.legend()

    # Validation F1 のプロット
    for i, history in enumerate(all_histories):
        epochs = range(1, len(history['val_f1']) + 1)
        if i == best_fold_index:
            ax2.plot(epochs, history['val_f1'], 's-', linewidth=2.5, label=f'Best Fold ({i+1})')
        else:
            ax2.plot(epochs, history['val_f1'], '.-', linewidth=1.0, alpha=0.6, label=f'Fold {i+1}')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Validation F1 Score')
    ax2.set_title('Validation F1 Score across Folds')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'cv_summary_plot.png')
    plt.savefig(save_path)
    print(f"\nCV summary plot saved to: {save_path}")
    plt.close(fig)

def summarize_and_save_results(config, all_histories, oof_val_scores, best_model_state_overall, best_fold_index, output_dir):
    """最終結果を集計し、モデルとログを保存する"""
    mean_f1 = np.mean(oof_val_scores)
    std_f1 = np.std(oof_val_scores)
    
    print("\n\n========== Cross-Validation Final Results ==========")
    print(f"F1 Scores for each fold: {[round(f, 4) for f in oof_val_scores]}")
    print(f"Mean CV F1 Score: {mean_f1:.4f}")
    print(f"Std Dev CV F1 Score: {std_f1:.4f}")
    print(f"Best F1 score was {oof_val_scores[best_fold_index]:.4f} in fold {best_fold_index+1}")
    
    # 総合グラフの描画
    plot_cv_summary(all_histories, best_fold_index, output_dir)
    
    # 最高のモデルを保存
    model_save_path = os.path.join(output_dir, f"best_model_cv_f1_{mean_f1:.4f}.pt")
    torch.save(best_model_state_overall, model_save_path)
    print(f"Best overall model state saved to: {model_save_path}")

    # 最高のモデルのログをJSONで保存
    log_data = {
        "config": config,
        "cv_summary": {
            "mean_f1": mean_f1, "std_f1": std_f1,
            "all_fold_f1": oof_val_scores
        },
        "best_fold_info": {
            "fold_index": best_fold_index,
            "f1_score": oof_val_scores[best_fold_index]
        },
        "best_fold_history": all_histories[best_fold_index]
    }
    log_save_path = os.path.join(output_dir, "summary_log.json")
    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    print(f"Summary log saved to: {log_save_path}")


# =====================================================================================
# 3. 1 Foldの学習・検証関数
# =====================================================================================
def train_one_fold(fold, train_df, val_df, config, device, output_dir):
    print(f"\n========== FOLD {fold+1}/{config['n_splits']} ==========")
    
    tokenizer, model = initialize_model(config)
    model.to(device)

    train_dataset = preprocess_data(train_df, tokenizer, config)
    val_dataset = preprocess_data(val_df, tokenizer, config)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    num_training_steps = config['num_epochs'] * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * config['num_warmup_steps_ratio'])
    lr_scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    
    metrics = {name: evaluate.load(name) for name in ["accuracy", "precision", "recall", "f1"]}
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
    best_val_f1 = 0.0

    # best_val_lossの初期化
    best_val_loss = float('inf')

    best_model_state = None
    epochs_without_improvement = 0

    # Focal Lossの初期化
    use_focal = config.get('use_focal_loss', False)
    if use_focal:
        focal_alpha = config.get('focal_alpha', 0.25)
        focal_gamma = config.get('focal_gamma', 2.0)
        focal_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction='mean')

    for epoch in range(config['num_epochs']):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            if use_focal:
                # Focal Lossでloss計算
                loss = focal_loss_fn(outputs.logits, batch['labels'])
            else:
                loss = outputs.loss
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)
        
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                if use_focal:
                    loss = focal_loss_fn(outputs.logits, batch['labels'])
                else:
                    loss = outputs.loss
                total_val_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        history['val_loss'].append(avg_val_loss)
        
        val_acc = metrics['accuracy'].compute(predictions=all_preds, references=all_labels)['accuracy']
        val_pre = metrics['precision'].compute(predictions=all_preds, references=all_labels, average='binary')['precision']
        val_rec = metrics['recall'].compute(predictions=all_preds, references=all_labels, average='binary')['recall']
        val_f1 = metrics['f1'].compute(predictions=all_preds, references=all_labels, average='binary')['f1']

        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_pre)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1}/{config['num_epochs']} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {val_f1:.4f}")

        #if val_f1 > best_val_f1:
        #    best_val_f1 = val_f1
        #    best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        #    epochs_without_improvement = 0
        #else:
        #    epochs_without_improvement += 1
       
        # lossの未改善によるEarly Stopping
        if avg_val_loss < best_val_loss: 
            best_val_loss = avg_val_loss
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config['early_stopping_patience']:
            print(f"  -> Early stopping triggered at epoch {epoch+1}.")
            break
            
    print(f"Fold {fold+1} Best Val F1: {best_val_f1:.4f}")
    plot_fold_history(history, fold, os.path.join(output_dir, "fold_histories"))

    del model, tokenizer, train_dataloader, val_dataloader
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_val_f1, best_model_state, history

# =====================================================================================
# 4. main関数
# =====================================================================================
def main():
    config = {
        "model_name": "tohoku-nlp/bert-base-japanese-whole-word-masking",
        "n_splits": 5, "seed": 42, "max_length": 256, "batch_size": 16,
        "num_epochs": 15, "learning_rate": 8.0e-6, "weight_decay": 0.2,
        "num_warmup_steps_ratio": 0.3, "early_stopping_patience": 3,
        "id2label": {"0": "NOTIRONY", "1": "IRONY"},
        "label2id": {"NOTIRONY": 0, "IRONY": 1},
        "dataset_path": "../datasets/train_unbalanced.csv",
        "use_focal_loss": True,  # focal lossを使う場合はTrue
        "focal_alpha": 0.25,     # IRONYクラスの重み
        "focal_gamma": 2.0       # フォーカスパラメータ
    }
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- Configuration ---")
    print(json.dumps(config, indent=2))
    
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"cv_results_{timestamp}"
    os.makedirs(os.path.join(output_dir, "fold_histories"), exist_ok=True)
    print(f"\nOutput will be saved to: {output_dir}")

    # データセットパスをconfigから取得
    df = pd.read_csv(config['dataset_path'])
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
    
    all_histories = []
    oof_val_scores = []
    best_overall_f1 = 0.0
    best_model_state_overall = None
    best_fold_index = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        
        val_f1, model_state, history = train_one_fold(fold, train_df, val_df, config, device, output_dir)
        
        oof_val_scores.append(val_f1)
        all_histories.append(history)

        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_model_state_overall = model_state
            best_fold_index = fold
            
    # 最終結果の集計と保存
    summarize_and_save_results(config, all_histories, oof_val_scores, best_model_state_overall, best_fold_index, output_dir)


if __name__ == '__main__':
    main()

