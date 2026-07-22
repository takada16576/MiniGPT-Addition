import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.utils import format_number
from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
    
#--------------------------------------------------
def split_expression(expr, operators="+-*"):
    for i in range(1, len(expr)):
        if expr[i] in operators:
            return expr[:i], expr[i], expr[i+1:]
    return None

def normalize_prompt(prompt):
    try:
        left, op, right = split_expression(prompt)
        #print(left, op, right)

        a, b = int(left), int(right)

        if not (-99 <= a <= 99):
            return None
        if not (-99 <= b <= 99):
            return None

        #return f"{format_number(a,2)}-{format_number(b,2)}="
        return f"{format_number(a,2)}{op}{format_number(b,2)}="

    except Exception:
        return None
#--------------------------------------------------

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens):
    model.eval()    # 評価モード

    prompt = normalize_prompt(prompt)
    if prompt == None:
        return None

    # プロンプトをトークン化
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)[:-1]
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    eos_id = tokenizer.str_to_int["<EOS>"]
    # トークン生成ループ
    for _ in range(max_new_tokens):
        ids_cond = ids[:, -GPT_CONFIG["context_length"]:]
        logits = model(ids_cond)
        next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat((ids, next_id), dim=1)

        if next_id.item() == eos_id:
            break

    # デコード
    generated_text = tokenizer.decode(ids[0].tolist())
    #print("repr(generated_text):", repr(generated_text))
    return generated_text

