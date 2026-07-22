import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
from minigpt.utils import get_device, format_number
from minigpt.generate import generate
    
#--------------------------------------------------

#--------------------------------------------------
# 設定
import os, sys
#os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')
#model_save_path = 'minigpt/model_add.pt'
model_save_path = 'minigpt/model_add_sub.pt'
device = get_device()
max_new_tokens = 4

# モデルとトークナイザ
model = GPT(GPT_CONFIG).to(device)
model.load_state_dict(
    torch.load(model_save_path, map_location=device)
)
tokenizer = SimpleTokenizer(vocab)

# テキスト生成
print("「2桁整数の足し算と引き算」を学習したミニ生成AIです。\n式を入力してください。(例：'12+34') 'EXIT'で終了")
while True:
    inputs = input("Input: ")
    if (inputs == "EXIT") or (inputs == "exit"):
        break
    
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=inputs,
        max_new_tokens=max_new_tokens
    )
    if generated_text == None:
        continue
    _, right = generated_text.split('=')
    output = f"{inputs}={right}" 
    
    print(f"Output: {output}")
    #print(f"Output: {generated_text}")
