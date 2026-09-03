# make_dataset.py

from minigpt.utils import format_number

import random

random.seed(0)
#random.seed(1)
#random.seed(2)
#random.seed(3)

#################################################################################
#加算＋減算+乗算データセット

all_data = []

for a in range(-9, 10):
    for b in range(-9, 10):
        all_data.append(
            f"{format_number(a,4)}+{format_number(b,4)}|"
            f"{format_number(a,4)}+{format_number(b,4)}={format_number(a+b,9)}|"
            f"{format_number(a+b,9)}"
        )
        all_data.append(
            f"{format_number(a,4)}-{format_number(b,4)}|"
            f"{format_number(a,4)}-{format_number(b,4)}={format_number(a-b,9)}|"
            f"{format_number(a-b,9)}"
        )
        all_data.append(
            f"{format_number(a,4)}*{format_number(b,4)}|"
            f"{format_number(a,4)}*{format_number(b,4)}={format_number(a*b,9)}|"
            f"{format_number(a*b,9)}"
        )
random.shuffle(all_data)

train = all_data

with open("train_mix_1dig_cot.txt", "w") as f:
    for line in train:
        f.write(line + "\n")


#################################################################################

#################################################################################
errors_flag = False
# errors.txtからテストデータを作成するスクリプト
from minigpt.utils import extract_errors

if errors_flag:
    extract_errors('add_errors.txt', '+', 'train_add_errors.txt')
    extract_errors('sub_errors.txt', '-', 'train_sub_errors.txt')
    extract_errors('mul_errors.txt', '*', 'train_mul_errors.txt')

    # 三つのファイルを開いて、新しいファイルに書き込む
    with open('train_add_errors.txt', 'r', encoding='utf-8') as f1, \
         open('train_sub_errors.txt', 'r', encoding='utf-8') as f2, \
         open('train_mul_errors.txt', 'r', encoding='utf-8') as f3, \
         open('train_mix_errors.txt', 'w', encoding='utf-8') as outfile:
    
        # 1つ目のファイルの内容を書き込む
        outfile.write(f1.read())
        outfile.write('\n')
        # 2つ目のファイルの内容を書き込む
        outfile.write(f2.read())
        outfile.write('\n')
        # 3つ目のファイルの内容を書き込む
        outfile.write(f3.read())

#################################################################################
