import os
import gc
import json
import random
import datetime
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import evaluate
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from datasets import Dataset, ClassLabel
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    """
    Sets the random seed for reproducibility across all relevant libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FocalLoss(nn.Module):
    """
    An implementation of Focal Loss for multi-class classification, robust against
    class imbalance. It is a dynamically scaled cross-entropy loss.

    Attributes:
        alpha (torch.Tensor): A tensor of weights for each class.
        gamma (float): The focusing parameter to down-weight easy examples.
        reduction (str): Specifies the reduction to apply to the output:
                         'none' | 'mean' | 'sum'.
    """
    def __init__(self, alpha: List[float], gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha)
        self.register_buffer('alpha', torch.tensor(alpha))
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculates the focal loss.

        Args:
            logits: The model's raw output (batch_size, num_classes).
            targets: The ground truth labels (batch_size,).

        Returns:
            The calculated focal loss.
        """
        alpha = self.alpha.to(logits.device)

        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        # Gather the alpha values corresponding to the target classes
        alpha_t = alpha.gather(0, targets)

        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        if self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def create_output_directory(config: Dict[str, Any]) -> str:
    """
    Creates a unique output directory for the experiment.
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    model_name_short = config['model_name'].split('/')[-1]
    output_dir = f"results_{model_name_short}_{timestamp}"
    
    # Create subdirectories for detailed logs and plots
    os.makedirs(os.path.join(output_dir, "fold_plots"), exist_ok=True)
    
    print(f"\n[INFO] Output will be saved to: {output_dir}")
    return output_dir


def initialize_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """
    Initializes and returns the tokenizer and model from the Hugging Face Hub.
    """
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    model = AutoModelForSequenceClassification.from_pretrained(
        config['model_name'],
        num_labels=len(config['id2label']),
        id2label=config['id2label'],
        label2id=config['label2id'],
    )
    return tokenizer, model


def create_tokenized_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, config: Dict[str, Any]) -> Dataset:
    """
    Converts a pandas DataFrame into a tokenized Hugging Face Dataset.
    """
    dataset = Dataset.from_pandas(df)
    
    # Ensure the label column is correctly cast
    class_names = list(config['label2id'].keys())
    dataset = dataset.cast_column("label", ClassLabel(names=class_names))
    dataset = dataset.rename_column("label", "labels")

    def tokenize_function(examples):
        # Assuming 'Utterance' and 'Response' are the text columns to be combined
        return tokenizer(
            examples['Utterance'],
            examples['Response'],
            padding='max_length',
            truncation=True,
            max_length=config['max_length']
        )
    
    # Remove original text columns and index if it exists
    cols_to_remove = [col for col in ['Utterance', 'Response', '__index_level_0__'] if col in dataset.column_names]
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=cols_to_remove)
    tokenized_dataset.set_format('torch')
    return tokenized_dataset


def run_fold_training(
    fold: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Dict[str, Any],
    device: torch.device,
    output_dir: str
) -> Tuple[float, Dict[str, Any], Dict[str, List], float]:
    """
    Executes the training and evaluation loop for a single fold.
    """
    print(f"\n========== FOLD {fold + 1}/{config['n_splits']} ==========")

    # Initialize model and tokenizer for the current fold
    tokenizer, model = initialize_model_and_tokenizer(config)
    model.to(device)

    # Prepare datasets and dataloaders
    train_dataset = create_tokenized_dataset(train_df, tokenizer, config)
    val_dataset = create_tokenized_dataset(val_df, tokenizer, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    num_training_steps = config['num_epochs'] * len(train_dataloader)
    num_warmup_steps = int(num_training_steps * config['num_warmup_steps_ratio'])
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

    # Loss Function
    if config['use_focal_loss']:
        print("[INFO] Using Focal Loss.")
        loss_fn = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
    elif config['use_class_weights']:
        print("[INFO] Using CrossEntropyLoss with class weights.")
        loss_fn = nn.CrossEntropyLoss(weight=config['class_weights'].to(device))
    else:
        print("[INFO] Using standard CrossEntropyLoss.")
        loss_fn = nn.CrossEntropyLoss()

    # Metrics and History Tracking
    metrics = {name: evaluate.load(name) for name in ["accuracy", "precision", "recall", "f1"]}
    history = {'train_loss': [], 'val_loss': [], 'val_accuracy': [], 'val_precision': [], 'val_recall': [], 'val_f1': []}
    
    best_val_f1 = -1.0
    best_model_state = None
    epochs_no_improve = 0
    
    progress_bar = tqdm(range(config['num_epochs']), desc=f"Fold {fold+1}")
    for epoch in progress_bar:
        # Training loop
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = loss_fn(outputs.logits, batch['labels'])
            
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        avg_train_loss = total_train_loss / len(train_dataloader)
        history['train_loss'].append(avg_train_loss)

        # Evaluation loop
        model.eval()
        total_val_loss = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = loss_fn(outputs.logits, batch['labels'])
                total_val_loss += loss.item()
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_metrics = {
            'accuracy': metrics['accuracy'].compute(predictions=all_preds, references=all_labels)['accuracy'],
            'precision': metrics['precision'].compute(predictions=all_preds, references=all_labels, average='binary')['precision'],
            'recall': metrics['recall'].compute(predictions=all_preds, references=all_labels, average='binary')['recall'],
            'f1': metrics['f1'].compute(predictions=all_preds, references=all_labels, average='binary')['f1']
        }
        
        history['val_loss'].append(avg_val_loss)
        for key, value in val_metrics.items():
            history[f'val_{key}'].append(value)
        
        progress_bar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}", val_f1=f"{val_metrics['f1']:.4f}")

        # Early stopping and model saving based on validation F1 score
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"  -> New best F1: {best_val_f1:.4f} at epoch {epoch + 1}. Model saved.")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= config['early_stopping_patience']:
            print(f"  -> Early stopping triggered after {config['early_stopping_patience']} epochs with no improvement.")
            break
            
    print(f"Fold {fold + 1} finished. Best Validation F1: {best_val_f1:.4f}")
    plot_fold_history(history, fold, os.path.join(output_dir, "fold_plots"))

    del model, tokenizer, train_dataloader, val_dataloader, optimizer, lr_scheduler
    gc.collect()
    torch.cuda.empty_cache()
    
    return best_val_f1, best_model_state, history


def plot_fold_history(history: Dict[str, List], fold_idx: int, save_dir: str):
    """
    Plots and saves the training history (loss and F1 score) for a single fold.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.plot(epochs, history['train_loss'], 'o-', color='tab:blue', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'o--', color='tab:cyan', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Plot F1 Score on a second y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('F1 Score', color='tab:red')
    ax2.plot(epochs, history['val_f1'], 's-', color='tab:red', label='Validation F1')
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.set_ylim(0, 1.05)

    # Final Touches
    fig.suptitle(f'Fold {fold_idx + 1} - Training History', fontsize=16)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(save_dir, f'fold_{fold_idx + 1}_history.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def summarize_and_save_results(
    config: Dict,
    all_histories: List,
    oof_f1_scores: List,
    best_model_state: Dict,
    best_fold_idx: int,
    output_dir: str
):
    """
    Summarizes CV results, plots overall performance, and saves the final model and logs.
    """
    mean_f1 = np.mean(oof_f1_scores)
    std_f1 = np.std(oof_f1_scores)

    print("\n\n" + "="*50)
    print("      CROSS-VALIDATION FINAL SUMMARY")
    print("="*50)
    print(f"F1 Scores per Fold : {[f'{f:.4f}' for f in oof_f1_scores]}")
    print(f"Mean CV F1 Score   : {mean_f1:.4f}")
    print(f"Std Dev of F1 Score: {std_f1:.4f}")
    print(f"Best performing fold : Fold {best_fold_idx + 1} with F1 = {oof_f1_scores[best_fold_idx]:.4f}")
    print("="*50 + "\n")

    # Save the best model from all folds
    if best_model_state:
        model_save_path = os.path.join(output_dir, f"best_model_f1_{mean_f1:.4f}.pth")
        torch.save(best_model_state, model_save_path)
        print(f"[INFO] Best overall model state saved to: {model_save_path}")
    else:
        print("[WARNING] No best model was saved. All F1 scores might have been zero.")
    
    # Save a comprehensive log file
    log_data = {
        "config": config,
        "cv_summary": {
            "mean_f1": mean_f1,
            "std_f1": std_f1,
            "all_fold_f1": oof_f1_scores
        },
        "best_fold_details": {
            "fold_index": best_fold_idx,
            "f1_score": oof_f1_scores[best_fold_idx],
            "history": all_histories[best_fold_idx]
        }
    }
    # Convert tensors in config to lists for JSON serialization
    if 'class_weights' in log_data['config']:
        log_data['config']['class_weights'] = log_data['config']['class_weights'].tolist()
        
    log_save_path = os.path.join(output_dir, "summary_log.json")
    with open(log_save_path, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)
    print(f"[INFO] Summary log saved to: {log_save_path}")


def main():
    """
    Main function to orchestrate the entire training and evaluation pipeline.
    """
    # --- Configuration ---
    # Centralized configuration for the experiment.
    # Adjust these parameters to tune the model's performance.
    config = {
        "model_name": "tohoku-nlp/bert-base-japanese-whole-word-masking",
        "dataset_path": "../datasets/train_unbalanced.csv",
        "n_splits": 5,
        "seed": 42,
        "max_length": 256,
        "batch_size": 16,
        "num_epochs": 15,
        "learning_rate": 1e-5,          # Recommended learning rate for BERT fine-tuning
        "weight_decay": 0.1,
        "num_warmup_steps_ratio": 0.1,  # Reduced warmup ratio
        "early_stopping_patience": 3,
        "id2label": {"0": "NOTIRONY", "1": "IRONY"},
        "label2id": {"NOTIRONY": 0, "IRONY": 1},
        
        # --- Imbalance Handling ---
        # Choose one strategy: class_weights (recommended) or focal_loss.
        "use_class_weights": False,
        "use_focal_loss": True,
        "focal_alpha": [0.25, 0.75],    # Weights for [NOTIRONY, IRONY]
        "focal_gamma": 2.0,             # Standard value for gamma
    }

    # --- Initialization ---
    set_seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = create_output_directory(config)
    
    print("\n--- Experiment Configuration ---")
    print(json.dumps({k: str(v) if isinstance(v, (torch.Tensor, list)) else v for k,v in config.items()}, indent=2))
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    df = pd.read_csv(config['dataset_path'])

    # Calculate class weights if enabled
    if config['use_class_weights']:
        class_counts = df['label'].value_counts().sort_index()
        weights = 1.0 / class_counts
        normalized_weights = weights / weights.sum()
        config['class_weights'] = torch.tensor(normalized_weights.values, dtype=torch.float)
        print(f"[INFO] Calculated class weights: {config['class_weights'].tolist()}")

    # --- Cross-Validation Loop ---
    skf = StratifiedKFold(n_splits=config['n_splits'], shuffle=True, random_state=config['seed'])
    
    all_histories = []
    oof_f1_scores = []
    best_overall_f1 = -1.0
    best_model_state_overall = None
    best_fold_index = -1

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        fold_f1, model_state, history = run_fold_training(fold, train_df, val_df, config, device, output_dir)
        
        all_histories.append(history)
        oof_f1_scores.append(fold_f1)

        if fold_f1 > best_overall_f1:
            best_overall_f1 = fold_f1
            best_model_state_overall = model_state
            best_fold_index = fold
            
    # --- Finalization ---
    summarize_and_save_results(
        config, all_histories, oof_f1_scores, best_model_state_overall, best_fold_index, output_dir
    )

if __name__ == '__main__':
    main()
