import pandas as pd
from pathlib import Path
from typing import Tuple

def load_and_shuffle_csv(
    csv_path: Path, 
    seed: int = 42
) -> pd.DataFrame:
    """CSVを読み込み、指定したシードでシャッフル"""
    df = pd.read_csv(csv_path)
    df_shuffled = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_shuffled

def split_by_label(
    df: pd.DataFrame, 
    label_col: str = "label"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ラベルごとにデータフレームを分割"""
    df_1 = df[df[label_col] == 1].reset_index(drop=True)
    df_0 = df[df[label_col] == 0].reset_index(drop=True)
    return df_1, df_0

def cut_head(df: pd.DataFrame, n: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """先頭n件を切り取り、残りと分割"""
    head = df.iloc[:n].reset_index(drop=True)
    rest = df.iloc[n:].reset_index(drop=True)
    return head, rest

def extract_balanced_unbalanced(
    df: pd.DataFrame,
    label_col: str,
    num_1: int,
    num_0_balanced: int,
    num_0_unbalanced: int,
    prefix: str
):
    """均衡・不均衡データセットを抽出して保存"""
    df_1, df_0 = split_by_label(df, label_col)
    # 均衡
    bal_1 = df_1.iloc[:num_1].reset_index(drop=True)
    bal_0 = df_0.iloc[:num_0_balanced].reset_index(drop=True)
    balanced = pd.concat([bal_1, bal_0], ignore_index=True)
    balanced.to_csv(f"{prefix}_balanced.csv", index=False)
    # 不均衡
    unbal_1 = df_1.iloc[:num_1].reset_index(drop=True)
    unbal_0 = df_0.iloc[:num_0_unbalanced].reset_index(drop=True)
    unbalanced = pd.concat([unbal_1, unbal_0], ignore_index=True)
    unbalanced.to_csv(f"{prefix}_unbalanced.csv", index=False)

def main(
    input_csv: str = "../datasets/merged.csv",
    eval_csv: str = "eval.csv",
    train_csv: str = "train.csv",
    label_col: str = "label",
    eval_num_1: int = 100,
    eval_num_0: int = 4000,
    train_num_1: int = 200,
    train_num_0: int = 7800,
    seed: int = 42
):
    # シャッフル
    df = load_and_shuffle_csv(input_csv, seed)
    # ラベルごとに分割
    df_1, df_0 = split_by_label(df, label_col)
    # eval用に切り取り
    eval_1, rest_1 = cut_head(df_1, eval_num_1)
    eval_0, rest_0 = cut_head(df_0, eval_num_0)
    eval_df = pd.concat([eval_1, eval_0], ignore_index=True)
    eval_df.to_csv(eval_csv, index=False)
    # train用に切り取り
    train_1, _ = cut_head(rest_1, train_num_1)
    train_0, _ = cut_head(rest_0, train_num_0)
    train_df = pd.concat([train_1, train_0], ignore_index=True)
    train_df.to_csv(train_csv, index=False)

    # evalから均衡・不均衡データセット抽出
    extract_balanced_unbalanced(
        eval_df, label_col,
        num_1=eval_num_1,
        num_0_balanced=eval_num_1,     # 100:100
        num_0_unbalanced=eval_num_0-100, # 100:3900
        prefix="eval"
    )
    # trainから均衡・不均衡データセット抽出
    extract_balanced_unbalanced(
        train_df, label_col,
        num_1=train_num_1,
        num_0_balanced=train_num_1,      # 200:200
        num_0_unbalanced=train_num_0,    # 200:7800
        prefix="train"
    )
    
if __name__ == "__main__":
    main()