import torch
import torch.nn as nn
import torch.nn.functional as F

from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
    
#--------------------------------------------------
@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens):
    model.eval()    # 評価モード

    # プロンプトをトークン化
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt)[:-1]
    ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated_ids = ids.clone()
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
    return generated_text

