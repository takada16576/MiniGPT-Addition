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
