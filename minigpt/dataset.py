from minigpt.tokenizer import SimpleTokenizer, vocab

tokenizer = SimpleTokenizer(vocab)

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

#with open("data/train_add.txt", "r", encoding="utf-8") as f:
#with open("data/train_sub.txt", "r", encoding="utf-8") as f:
#with open("data/train_add_sub.txt", "r", encoding="utf-8") as f:
#with open("data/train_add_sub_mul.txt", "r", encoding="utf-8") as f:

#with open("data/train_mul.txt", "r", encoding="utf-8") as f:
#    raw_text = []
#    #total = 9000*2
#    #total = 199 * 199
#    total = 5000
#    for _ in range(total):
#        temp = f.readline()
#        raw_text.append(temp[:-1])

import random

random.seed(0)
#random.seed(1)
#random.seed(2)
#random.seed(3)

with open("data/train_mix.txt", "r", encoding="utf-8") as f:
    train_text = [line.rstrip("\n") for line in f]

total = 199*199*3
#total = 5000

raw_text = random.sample(train_text, total)

#with open("data/train_mix_errors_26.txt", "r", encoding="utf-8") as f:
#    raw_text += [line.rstrip("\n") for line in f]


random.shuffle(raw_text)


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