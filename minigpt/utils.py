import torch

def get_device():
    """利用可能なデバイスを自動検出して返す"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')
    
###############################################
#format_number(3, 2)     # +03
#format_number(-7, 2)    # -07
#format_number(108, 3)   # +108
#format_number(-15, 3)   # -015

def format_number(n, width):
    sign = '+' if n >= 0 else '-'
    return f"{sign}{abs(n):0{width}d}"

###############################################
# errors.txtからテストデータを抜き出すスクリプト
#95*84 | expected=7980 | generated=+95*+84=+8080
#98+-8 | expected=90 | generated=+98+-08=+091
#-28+78 | expected=50 | generated=-28++78=+051
import re
def extract_errors00(error_file, operator, output_file):
    train_data = []

    with open(error_file, encoding="utf-8") as f:
        for line in f:
            m = re.search(r'expected=([+-]?\d+)', line)
            if not m:
                continue

            expected = int(m.group(1))

            expr = line.split()[0]          # "-11*-11"
            left, right = expr.split(operator)

            train_data.append(
            f"{format_number(int(left),2)}{operator}"
            f"{format_number(int(right),2)}="
            f"{expected:+05d}"
        )
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))

import re

def extract_errors(error_file, operator, output_file):
    train_data = []

    pattern = rf'^([+-]?\d+)\{operator}([+-]?\d+)'

    with open(error_file, encoding="utf-8") as f:
        for line in f:
            m_expected = re.search(r'expected=([+-]?\d+)', line)
            if not m_expected:
                continue

            expected = int(m_expected.group(1))

            expr = line.split()[0]

            m_expr = re.match(pattern, expr)
            if not m_expr:
                continue

            left = int(m_expr.group(1))
            right = int(m_expr.group(2))

            train_data.append(
                f"{format_number(left,2)}{operator}"
                f"{format_number(right,2)}="
                f"{expected:+05d}"
            )


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))