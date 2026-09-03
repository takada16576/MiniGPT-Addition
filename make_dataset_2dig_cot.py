# make_dataset.py

from minigpt.utils import format_number

import random

random.seed(0)

#################################################################################
#加算＋減算+乗算データセット

all_data = []

for a in range(-99, 100):
    for b in range(-99, 100):
        aa = abs(a)
        bb = abs(b)
        ones = bb % 10
        tens = bb // 10
        part1 = aa * ones
        part2 = aa * tens * 10
        result = part1 + part2
        all_data.append(
            f"{format_number(a,4)}*{format_number(b,4)}|"
            f"{format_number(aa,4)}*{format_number(ones,4)}={format_number(part1,9)}|"
            f"{format_number(aa,4)}*{format_number(tens*10,4)}={format_number(part2,9)}|"
            f"{format_number(part1,4)}+{format_number(part2,4)}={format_number(result,9)}|"
            f"{format_number(a*b,9)}"
        )
        
        if a < 0 and b > 0:
            all_data.append(
                f"{format_number(a,4)}+{format_number(b,4)}|"
                f"{format_number(b,4)}+{format_number(a,4)}={format_number(a+b,9)}|"
                f"{format_number(a,4)}+{format_number(b,4)}={format_number(a+b,9)}|"
                f"{format_number(a+b,9)}"
            )
        else:
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

#total = 199*199*3 = 39,601*3=118,803   2桁学習データ
#total = 1999*1999*3 = 3,996,001*3=11,988,003   3桁学習データ
random.shuffle(all_data)
train = all_data[10000:]    # 10,000~118,802 約１０万件
test = all_data[:10000]     # 0~9,999 １万件
print("total:", len(all_data))
print("train:", len(train))
print("test :", len(test))

with open("train_mix_2dig_cot.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("test_mix_2dig_cot.txt", "w") as f:
    for line in test:
        f.write(line + "\n")

#################################################################################
errors_flag = False
#################################################################################
# errors.txtからテストデータを作成するスクリプト
from minigpt.utils import extract_cot_errors

if errors_flag:
    extract_cot_errors('add_errors.txt', '+', 'train_add_errors.txt')
    extract_cot_errors('sub_errors.txt', '-', 'train_sub_errors.txt')
    extract_cot_errors('mul_errors.txt', '*', 'train_mul_errors.txt')

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
# 補助データセット auxiliary dataset

if errors_flag:
    all_data = []

    for a in range(-99, 100):
        for b in [-10, 10]:
            aa = abs(a)
            bb = abs(b)
            ones = bb % 10
            tens = bb // 10
            part1 = aa * ones
            part2 = aa * tens * 10
            result = part1 + part2
            all_data.append(
                f"{format_number(a,4)}*{format_number(b,4)}|"
                f"{format_number(aa,4)}*{format_number(ones,4)}={format_number(part1,9)}|"
                f"{format_number(aa,4)}*{format_number(tens*10,4)}={format_number(part2,9)}|"
                f"{format_number(part1,4)}+{format_number(part2,4)}={format_number(result,9)}|"
                f"{format_number(a*b,9)}"
            )

    train = all_data.copy()
    random.shuffle(train)

    print("aux total:", len(train))   # 199 * 2 = 398
    with open("train_mix_aux_cot.txt", "w") as f:
        for line in train:
            f.write(line + "\n")

#################################################################################