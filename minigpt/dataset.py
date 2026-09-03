from minigpt.tokenizer import SimpleTokenizer, vocab

tokenizer = SimpleTokenizer(vocab)

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

import random

random.seed(0)
#random.seed(1)
#random.seed(2)
#random.seed(3)

with open("data/train_mix_1dig_cot.txt", "r", encoding="utf-8") as f:
    train_text = [line.rstrip("\n") for line in f]
total = 19*19*3  # 1,083
raw_text = random.sample(train_text, total)

with open("data/train_mix_2dig_cot.txt", "r", encoding="utf-8") as f:
    train_text = [line.rstrip("\n") for line in f]
total = 199*199*3 - 10000  # 118,803 - 10,000
raw_text += random.sample(train_text, total)

errors_flag = False
if errors_flag:
    #with open("data/train_mix_aux_cot.txt", "r", encoding="utf-8") as f:
    #    raw_text += [line.rstrip("\n") for line in f]

    with open("data/train_mix_errors.txt", "r", encoding="utf-8") as f:
        raw_text += [line.rstrip("\n") for line in f]

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


from torch.nn.utils.rnn import pad_sequence

PAD_ID = 0

def collate_fn(batch):
    xs, ys = zip(*batch)

    xs = pad_sequence(
        xs,
        batch_first=True,
        padding_value=PAD_ID
    )

    ys = pad_sequence(
        ys,
        batch_first=True,
        padding_value=-100
    )

    return xs, ys