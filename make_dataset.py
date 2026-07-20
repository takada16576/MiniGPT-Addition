# make_dataset.py

from minigpt.utils import format_number

import random

random.seed(0)
#random.seed(1)
#random.seed(2)
#random.seed(3)

def make_question_add():
    #a = random.randint(0, 99)
    #b = random.randint(0, 99)
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

train = []

for _ in range(9000*2):
    train.append(make_question_add())

with open("train_add.txt", "w") as f:
#with open("train_sub.txt", "w") as f:
    for line in train:
        f.write(line + "\n")
