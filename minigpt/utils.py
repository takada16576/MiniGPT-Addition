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
#-0095++0044 | expected=-000000051 | generated=-0095++0044=-000000059

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
                f"{format_number(left,4)}{operator}"
                f"{format_number(right,4)}="
                f"{expected:+010d}"
            )


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))

###############################################
# errors.txtからcotテストデータを抜き出すスクリプト
#68*15 | expected=1020 | generated=+0068*+0015=+0068*+0005=+000000340|+0068*+0010=+0000006
#+0047*+0088=+0047*+0008=+000000376|+0047*+0080=+000003760|+0376++3760=+000004136|+000004136

import re

def extract_cot_errors(error_file, operator, output_file):
    train_data = []

    pattern = rf'^([+-]?\d+)\{operator}([+-]?\d+)'

    with open(error_file, encoding="utf-8") as f:
        for line in f:
            m_expected = re.search(r'expected=([+-]?\d+)', line)
            if not m_expected:
                continue

            #expected = int(m_expected.group(1))

            expr = line.split()[0]

            m_expr = re.match(pattern, expr)
            if not m_expr:
                continue

            a = int(m_expr.group(1))
            b = int(m_expr.group(2))
            aa = abs(a)
            bb = abs(b)
            ones = bb % 10
            tens = bb // 10
            if operator == '*':
                part1 = aa * ones
                part2 = aa * tens * 10
                result = aa * bb
                train_data.append(
                    f"{format_number(a,4)}*{format_number(b,4)}|"
                    f"{format_number(aa,4)}*{format_number(ones,4)}={format_number(part1,9)}|"
                    f"{format_number(aa,4)}*{format_number(tens*10,4)}={format_number(part2,9)}|"
                    f"{format_number(part1,4)}+{format_number(part2,4)}={format_number(result,9)}|"
                    f"{format_number(a*b,9)}"
                )
            elif operator == '+':
                if a<0 and b>0:
                    train_data.append(
                        f"{format_number(a,4)}+{format_number(b,4)}|"
                        f"{format_number(b,4)}+{format_number(a,4)}={format_number(a+b,9)}|"
                        f"{format_number(a,4)}+{format_number(b,4)}={format_number(a+b,9)}|"
                        f"{format_number(a+b,9)}"
                    )
                else:
                    train_data.append(
                        f"{format_number(a,4)}+{format_number(b,4)}|"
                        f"{format_number(a,4)}+{format_number(b,4)}={format_number(a+b,9)}|"
                        f"{format_number(a+b,9)}"
                    )

            elif operator == '-':
                train_data.append(
                    f"{format_number(a,4)}-{format_number(b,4)}|"
                    f"{format_number(a,4)}-{format_number(b,4)}={format_number(a-b,9)}|"
                    f"{format_number(a-b,9)}"
                )


    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_data))

#############################################################
