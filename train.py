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
# 設定
device = get_device()
model_save_path = 'miniGPT/model_pretrain.pt'

# ハイパーパラメータ
learning_rate = 3e-4
#max_iters = 20000
#max_iters = 22500   # ステップで1epoch: 2250 (9000問 / 4 = 2250)　で。10epoch=22500
num_epochs = 10

# データ準備
dataset = GPTDataset(raw_text, tokenizer)
dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    #num_workers=0
)

# モデル、オプティマイザ
model = GPT(GPT_CONFIG).to(device)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
losses = []

global_step = 0
epoch_loss = 0

for epoch in range(num_epochs):

    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
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
        #pbar.set_postfix(loss=f"{loss.item():.4f}")
        global_step += 1
        
        if global_step % 50 == 0:
            print(
                f"Ep {epoch+1} :(Step {global_step}): "
                f"Train loss {loss.item():.4f}"
            )
        
    #pbar.set_postfix(loss=f"{loss.item():.4f}")
    epoch_loss += loss.item()
    print(
        f"Epoch {epoch+1}: "
        f"Loss={epoch_loss/len(dataloader):.4f}"
    )

##########################
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.savefig('loss_pretrain.png')
plt.show()

#############################
torch.save(model.state_dict(), model_save_path)
