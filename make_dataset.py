# make_dataset.py

import random

random.seed(0)

def make_question():
    a = random.randint(0, 99)
    b = random.randint(0, 99)
    #return f"{a}+{b}={a+b}"
    return f"{str(a).zfill(2)}+{str(b).zfill(2)}={str(a+b).zfill(3)}"   # ゼロパディング

train = []
test = []

for _ in range(9000):
    train.append(make_question())

for _ in range(1000):
    test.append(make_question())

with open("train.txt", "w") as f:
    for line in train:
        f.write(line + "\n")

with open("test.txt", "w") as f:
    for line in test:
        f.write(line + "\n")