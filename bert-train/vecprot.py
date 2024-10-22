import pandas as pd
from transformers import BertTokenizer
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
model = AutoModelForPreTraining.from_pretrained("tohoku-nlp/bert-base-japanese-v3")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def vectorize(texts):
    vectors = []
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
    output = outputs[0]
    return output   

def load_corpus(path):
    with open(path, 'r') as file:
        corpus = []
        for line in file:
            corpus.append(line.strip())
        return corpus

def main():
    IronyCorpus = 'corpus/Irony431.txt'
    NotIronyCorpus = 'notirony431.txt'
    Irony = load_corpus(IronyCorpus)
    NotIrony = load_corpus(NotIronyCorpus)

    print(Irony)

    IronyVectors = vectorize(Irony)
    NotIronyVectors = vectorize(NotIrony)

    print(IronyVectors)
    print(NotIronyVectors)

main()