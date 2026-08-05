# make_dataset.py

from minigpt.utils import format_number

import random

random.seed(0)
#random.seed(1)
#random.seed(2)
#random.seed(3)

def make_question_add():
    a = random.randint(-99, 99)
    b = random.randint(-99, 99)
    #return f"{str(a).zfill(2)}+{str(b).zfill(2)}={str(a+b).zfill(3)}"   # ゼロパディング
    return f"{format_number(a, 2)}+{format_number(b, 2)}={format_number(a+b, 3)}"


def make_question_sub():
    a = random.randint(-99, 99)
    b = random.randint(-99, 99)
    return f"{format_number(a, 2)}-{format_number(b, 2)}={format_number(a-b, 3)}"

    
def make_question_mul():
    a = random.randint(-99, 99)
    b = random.randint(-99, 99)
    return f"{format_number(a, 2)}*{format_number(b, 2)}={format_number(a*b, 4)}"

#################################################################################
#加算＋減算データセット

train = []

#total = 9000*2
#for i in range(total):
#    if i < 9000:
#        train.append(make_question_add())
#    else:
#        train.append(make_question_sub())
for a in range(-99, 100):
    for b in range(-99, 100):
        train.append(
            #f"{format_number(a,2)}+{format_number(b,2)}={format_number(a+b,3)}"
            f"{format_number(a,2)}+{format_number(b,2)}={format_number(a+b,4)}"

        )
        train.append(
            #f"{format_number(a,2)}-{format_number(b,2)}={format_number(a-b,3)}"
            f"{format_number(a,2)}-{format_number(b,2)}={format_number(a-b,4)}"
        )

with open("train_add_sub.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

#################################################################################
#乗算データセット

train = []

#total = 199*199
#for i in range(total):
#    train.append(make_question_mul())

for a in range(-99, 100):
    for b in range(-99, 100):
        train.append(
            f"{format_number(a,2)}*{format_number(b,2)}={format_number(a*b,4)}"
        )

with open("train_mul.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

#################################################################################
#加算＋減算+乗算データセット

# 二つのファイルを開いて、新しいファイルに書き込む
with open('train_add_sub.txt', 'r', encoding='utf-8') as f1, \
     open('train_mul.txt', 'r', encoding='utf-8') as f2, \
     open('train_mix.txt', 'w', encoding='utf-8') as outfile:
    
    # 1つ目のファイルの内容を書き込む
    outfile.write(f1.read())
    
    # 必要に応じて改行を挟む
    #outfile.write('\n')
    
    # 2つ目のファイルの内容を書き込む
    outfile.write(f2.read())

#################################################################################
# errors.txtからテストデータを作成するスクリプト
from minigpt.utils import extract_errors

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
# 答えが -5～+5 の加算・減算問題
def format_number(n, digits):
    sign = "+" if n >= 0 else "-"
    return f"{sign}{abs(n):0{digits}d}"

train_data = []

for a in range(-99, 100):
    for b in range(-99, 100):

        # 加算
        result = a + b
        if -3 <= result <= 3:
            expr = (
                f"{format_number(a,2)}"
                f"+"
                f"{format_number(b,2)}"
                f"="
                f"{format_number(result,4)}"
            )
            train_data.append(expr)

        # 減算
        result = a - b
        if -3 <= result <= 3:
            expr = (
                f"{format_number(a,2)}"
                f"-"
                f"{format_number(b,2)}"
                f"="
                f"{format_number(result,4)}"
            )
            train_data.append(expr)

with open("train_sign_fix.txt", "w", encoding="utf-8") as f:
    for line in train_data:
        f.write(line + "\n")

print(f"count={len(train_data)}")