#評価スクリプト

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
from minigpt.utils import get_device, format_number
from minigpt.generate import generate

#--------------------------------------------------
# 設定
import os, sys
#os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

#model_save_path = 'minigpt/model_mix_256_4_ep30_retrain2.pt'
model_save_path = 'minigpt/model_mix_256_4_100.pt'
#model_save_path = 'minigpt/model_mix_256_4_ep30_v1_retrain3.pt'

device = get_device()
max_new_tokens = 5  # 4 -> 5

# モデルとトークナイザ
model = GPT(GPT_CONFIG).to(device)

model.load_state_dict(
    torch.load(model_save_path, map_location=device)
)
model.eval()

tokenizer = SimpleTokenizer(vocab)

def evaluate(op):
    ok = 0
    errors = []
    numeric_errors = []
    format_errors = []
    with torch.no_grad():
        for a in range(-99, 100):
            for b in range(-99, 100):
                if op == "+":
                    prompt = f"{a}+{b}"
                    answer = a + b
                elif op == "-":
                    prompt = f"{a}-{b}"
                    answer = a - b
                elif op == "*":
                    prompt = f"{a}*{b}"
                    answer = a * b
                generated = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                )
                #print(f"generated:{generated}")

                if generated is None:
                    continue

                #_, right = generated.split("=")
                parts = generated.split("=")
                if len(parts) != 2:
                    ff = f"{prompt} | expected={answer} | generated={generated}"
                    format_errors.append(ff)
                    errors.append(ff)
                    continue

                _, right = parts

                try:
                    value = int(right)
                except ValueError:
                    ff = f"{prompt} | expected={answer} | generated={generated}"
                    format_errors.append(ff)
                    errors.append(ff)
                    continue

                if value == answer:
                    ok += 1
                else:
                    ff = f"{prompt} | expected={answer} | generated={generated}"
                    numeric_errors.append(ff)
                    errors.append(ff)

    return ok, numeric_errors, format_errors, errors

total = 199 * 199
##################################################################
ok, numeric_errors, format_errors, errors = evaluate("+")
print(f"Addition   : {ok/total:.2%}")
print(f"Add Errors : {len(errors)}")
print(f"Add Numeric Errors : {len(numeric_errors)}")
print(f"Add Format Errors : {len(format_errors)}")
with open("add_errors.txt", "w", encoding="utf-8") as f:
    for e in errors:
        f.write(e + "\n")
with open("add_numeric_errors.txt", "w", encoding="utf-8") as f:
    for e in numeric_errors:
        f.write(e + "\n")
with open("add_format_errors.txt", "w", encoding="utf-8") as f:
    for e in format_errors:
        f.write(e + "\n")

##################################################################
ok, numeric_errors, format_errors, errors = evaluate("-")
print(f"Subtraction   : {ok/total:.2%}")
print(f"Sub Errors : {len(errors)}")
print(f"Sub Numeric Errors : {len(numeric_errors)}")
print(f"Sub Format Errors : {len(format_errors)}")
with open("sub_errors.txt", "w", encoding="utf-8") as f:
    for e in errors:
        f.write(e + "\n")
with open("sub_numeric_errors.txt", "w", encoding="utf-8") as f:
    for e in numeric_errors:
        f.write(e + "\n")
with open("sub_format_errors.txt", "w", encoding="utf-8") as f:
    for e in format_errors:
        f.write(e + "\n")

##################################################################
ok, numeric_errors, format_errors, errors = evaluate("*")
print(f"Multiplication   : {ok/total:.2%}")
print(f"Mul Errors : {len(errors)}")
print(f"Mul Numeric Errors : {len(numeric_errors)}")
print(f"Mul Format Errors : {len(format_errors)}")
with open("mul_errors.txt", "w", encoding="utf-8") as f:
    for e in errors:
        f.write(e + "\n")
with open("mul_numeric_errors.txt", "w", encoding="utf-8") as f:
    for e in numeric_errors:
        f.write(e + "\n")
with open("mul_format_errors.txt", "w", encoding="utf-8") as f:
    for e in format_errors:
        f.write(e + "\n")