from tokenizer import CharTokenizer
import torch
import torch.nn as nn

with open("data/processed_dialogs.txt", encoding="utf-8") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
vocab_size = tokenizer.vocab_size
d_model = 128

print("Vocab size:", vocab_size)

sample = "[user] hello!\n[bot] hi!\n[end]\n"

embed = nn.Embedding(vocab_size, d_model)

ids = tokenizer.encode(sample)
vecs = embed(torch.tensor(ids))
decoded = tokenizer.decode(ids)

print(vecs.shape)
print(sample)
print(decoded)
