import pandas as pd
from transformers import BertTokenizer
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModelForPreTraining
import torch

tokenizer = AutoTokenizer.from_pretrained("tohoku-nlp/bert-base-japanese-v3")
model = AutoModelForPreTraining.from_pretrained("tohoku-nlp/bert-base-japanese-v3")

texts = ["hello", "world"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def vectorize(texts):
    inputs = tokenizer(texts, return_tensors="pt",padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    output = outputs[0]
    return output

vectors = vectorize(texts)
print(vectors)