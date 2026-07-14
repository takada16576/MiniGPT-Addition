from minigpt.tokenizer import SimpleTokenizer, vocab

tokenizer = SimpleTokenizer(vocab)

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

with open("data/train.txt", "r", encoding="utf-8") as f:
    raw_text = []
    for _ in range(9000):
        temp = f.readline()
        raw_text.append(temp[:-1])

import torch
from torch.utils.data import Dataset, DataLoader

class GPTDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.data = []
        for text in texts:
            ids = tokenizer.encode(text)
            #print(f"ids:\n{ids}")
            x =torch.tensor(ids[:-1], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]