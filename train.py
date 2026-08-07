# 事前学習
import torch
import torch.nn.functional as F

import os, sys
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append('.')

from itertools import cycle

from minigpt.tokenizer import SimpleTokenizer, vocab
from minigpt.model import GPT, GPT_CONFIG
from minigpt.utils import get_device
from minigpt.dataset import GPTDataset, DataLoader, raw_text

######################################################################

tokenizer = SimpleTokenizer(vocab)

device = get_device()


# ===== ハイパーパラメータ =====
learning_rate = 1e-4    # 1e-4 -> 1e-5
num_epochs = 1     # 10 -> 10 -> 10 -> 1

# ===== データ準備 =====
dataset = GPTDataset(raw_text, tokenizer)
pin_memory = (device.type == "cuda")
dataloader = DataLoader(
    dataset,
    batch_size=8,   # 4 -> 8
    shuffle=True,
    drop_last=True,
    num_workers=0,  # 0 -> 2
    pin_memory=pin_memory,
)

# ===== モデル =====
model = GPT(GPT_CONFIG)

pretrained_path = None
#pretrained_path = 'minigpt/model_mix_256_4_ep30_v1_retrain2.pt'
#pretrained_path = "minigpt/model_mix_256_4_ep10.pt"
#pretrained_path = "minigpt/model_mix_256_4_ep20.pt"
#pretrained_path = "minigpt/model_mix_256_4_ep30.pt"
#pretrained_path = "minigpt/model_mix_256_4_ep30_retrain.pt"
#pretrained_path = "minigpt/model_mix_256_4_ep30_retrain2.pt"


if pretrained_path is not None:
    model.load_state_dict(
        torch.load(pretrained_path, map_location=device)
    )

model.to(device)

# ===== 学習 =====
model.train()

# オプティマイザ
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
losses = []

global_step = 0

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_x, batch_y in dataloader:
        #batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x, batch_y = batch_x.to(device, non_blocking=True), batch_y.to(device, non_blocking=True)
        # 予測と損失計算
        logits = model(batch_x)
        # 形状変換
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))

        # 勾配計算とパラメータ更新
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )
        optimizer.step()

        losses.append(loss.item())
        # 進捗表示
        global_step += 1
        
        if global_step % 50 == 0:
            print(
                f"Ep {epoch+1} :(Step {global_step}): "
                f"Train loss {loss.item():.4f}"
            )
        
    epoch_loss += loss.item()
    print(
        f"Epoch {epoch+1}: "
        f"Loss={epoch_loss/len(dataloader):.4f}"
    )

# ===== グラフ描画 =====
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_pretrain.png')
plt.show()

# ===== 保存 =====
#model_save_path = 'minigpt/model_mix_256_4_ep10.pt'
#model_save_path = 'minigpt/model_mix_256_4_ep20.pt'
#model_save_path = 'minigpt/model_mix_256_4_ep30.pt'
#model_save_path = 'minigpt/model_mix_256_4_ep30_retrain.pt'
#model_save_path = 'minigpt/model_mix_256_4_ep30_retrain2.pt' # 
# 　　　　　　　　　=> 正答率100%になったので、model_mix_256_4_100.ptに名称変更
model_save_path = 'minigpt/model_mix_256_4_ep1_v2.pt'
torch.save(model.state_dict(), model_save_path)
