#高速版　評価スクリプト

import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
from minigpt.utils import get_device, format_number
from minigpt.generate import generate_batch

#--------------------------------------------------
# 設定
import os, sys
#os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

model_save_path = 'minigpt/model_mix_2dig_cot8_ep25.pt'

device = get_device()

#max_new_tokens = 95     # 107-12=95
max_new_tokens = 79     # 91-12=79
#max_new_tokens = 90     # 102-12=90

# モデルとトークナイザ
model = GPT(GPT_CONFIG).to(device)

model.load_state_dict(
    torch.load(model_save_path, map_location=device)
)
model.eval()

tokenizer = SimpleTokenizer(vocab)

import time

def evaluate(op, batch_size=32):

    print(device)

    t0 = time.time()

    ok = 0
    errors = []
    numeric_errors = []
    format_errors = []

    prompts = []
    answers = []

    #total = 199 * 199

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

                prompts.append(prompt)
                answers.append(answer)

                # batchがいっぱいになったら生成
                if len(prompts) == batch_size:

                    generated_list = generate_batch(
                        model,
                        tokenizer,
                        prompts,
                        max_new_tokens
                    )

                    for prompt, answer, generated in zip(
                        prompts,
                        answers,
                        generated_list
                    ):

                        parts = generated.split("|")

                        if len(parts) < 2:

                            ff = (
                                f"{prompt} | "
                                f"expected={answer} | "
                                f"generated={generated}"
                            )

                            format_errors.append(ff)
                            errors.append(ff)

                            continue

                        right = parts[-1]

                        try:
                            value = int(right)

                        except ValueError:

                            ff = (
                                f"{prompt} | "
                                f"expected={answer} | "
                                f"generated={generated}"
                            )

                            format_errors.append(ff)
                            errors.append(ff)

                            continue

                        if value == answer:

                            ok += 1

                        else:

                            ff = (
                                f"{prompt} | "
                                f"expected={answer} | "
                                f"generated={generated}"
                            )

                            numeric_errors.append(ff)
                            errors.append(ff)

                    if (ok + len(errors)) % 1000 < batch_size:
                        print(
                            ok + len(errors),
                            "elapsed=",
                            round(time.time() - t0, 1),
                            "sec"
                        )

                    prompts = []
                    answers = []

        # 最後の端数
        if prompts:

            generated_list = generate_batch(
                model,
                tokenizer,
                prompts,
                max_new_tokens
            )

            for prompt, answer, generated in zip(
                prompts,
                answers,
                generated_list
            ):

                parts = generated.split("|")

                if len(parts) < 2:
                    ff = (
                        f"{prompt} | "
                        f"expected={answer} | "
                        f"generated={generated}"
                    )
                    format_errors.append(ff)
                    errors.append(ff)
                    continue

                right = parts[-1]

                try:
                    value = int(right)
                except ValueError:
                    ff = (
                        f"{prompt} | "
                        f"expected={answer} | "
                        f"generated={generated}"
                    )
                    format_errors.append(ff)
                    errors.append(ff)
                    continue

                if value == answer:
                    ok += 1
                else:
                    ff = (
                        f"{prompt} | "
                        f"expected={answer} | "
                        f"generated={generated}"
                    )
                    numeric_errors.append(ff)
                    errors.append(ff)

    return ok, numeric_errors, format_errors, errors


total = 199 * 199
##################################################################
ok, numeric_errors, format_errors, errors = evaluate(
    "*",
    batch_size=64
)
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
##################################################################
ok, numeric_errors, format_errors, errors = evaluate(
    "+",
    batch_size=64
)
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
ok, numeric_errors, format_errors, errors = evaluate(
    "-",
    batch_size=64
)
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

