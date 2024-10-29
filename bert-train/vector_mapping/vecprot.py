import pandas as pd
from transformers import BertTokenizer
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
model = AutoModelForPreTraining.from_pretrained("tohoku-nlp/bert-base-japanese-v3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def vectorize(texts):
    inputs = tokenizer(texts, return_tensors="pt",padding=True, truncation=True, max_length=512)
    #  モデルを確実にGPUに載せる
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    output = outputs[0]
    pooled_output = torch.mean(output, dim=1)
    return pooled_output   

def load_corpus(path):
    with open(path, 'r') as file:
        corpus = []
        for line in file:
            corpus.append(line.strip())
        return corpus

def save_vectors(vectors, path):
    np.save(path, vectors.cpu().numpy())

def main():
    IronyCorpus = 'cps/Irony431.txt'
    NotIronyCorpus = 'cps/notirony431.txt'
    Irony = load_corpus(IronyCorpus)
    NotIrony = load_corpus(NotIronyCorpus)

    print(Irony)

    IronyVectors = vectorize(Irony)
    NotIronyVectors = vectorize(NotIrony)

    print(IronyVectors)
    print(NotIronyVectors)

    # ベクトル集合をファイルに保存
    save_vectors(IronyVectors, 'IronyVectors.npy')
    save_vectors(NotIronyVectors, 'NotIronyVectors.npy')

main()