import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, pipeline
import pandas as pd
import os

# データセットの読み込み
with open('cps/Irony431.txt', 'r') as file:
    irony = [line.strip() for line in file]

with open('cps/notirony431.txt', 'r') as file:
    not_irony = [line.strip() for line in file]

# データフレームの作成
df = pd.DataFrame(
    [{'text': text, 'label': 0} for text in irony] +
    [{'text': text, 'label': 1} for text in not_irony]
)

num_epochs = 20

# モデルの保存先
model_dir = f'./models/finetuned_model{num_epochs}'

def train():
    device = 'cuda'
    model.to(device)
    global num_epochs

    # データのエンコーディング
    train_docs = df['text'].tolist()
    train_labels = df['label'].tolist()
    labels = torch.tensor(train_labels).to(device)

    encodings = tokenizer(train_docs, return_tensors='pt', padding=True, truncation=True, max_length=128)
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    # データローダーの作成
    dataset = TensorDataset(input_ids, attention_mask, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # オプティマイザーの設定
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # トレーニングループ
    model.train()
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{num_epochs} completed with loss: {loss.item()}")

    # モデルの保存
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

def test():
    # 評価
    model.eval()
    classfinetuned = pipeline("text-classification", model=model, tokenizer=tokenizer, device='cuda')

    while True:
        print("終了 exit: ")
        a = input()
        if a == "exit":
            break
        else:
            print(classfinetuned(a))


# モデルとトークナイザーの初期化または読み込み
if os.path.exists(model_dir):
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Saved model and tokenizer loaded.")
    test()
else:
    model_name = 'cl-tohoku/bert-base-japanese'
    id2label = {0: 'irony', 1: 'not_irony'}
    label2id = {'irony': 0, 'not_irony': 1}

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("New model and tokenizer initialized.")
    train()
    test()

