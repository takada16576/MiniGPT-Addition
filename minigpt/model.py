import torch
import torch.nn as nn
import torch.nn.functional as F

#--------------------------------------------------
class Attention(nn.Module):
    def __init__(self, embed_dim, key_dim, context_length, dropout_rate=0.1):
        super().__init__()
        # Q, K, Vの変換行列   nn.Linear は最後の次元だけに作用します
        self.W_q = nn.Linear(embed_dim, key_dim, bias=False)    # (E, D) D 次元ベクトル
        self.W_k = nn.Linear(embed_dim, key_dim, bias=False)    # (E, D) D 次元ベクトル
        self.W_v = nn.Linear(embed_dim, key_dim, bias=False)    # (E, D) D 次元ベクトル
        self.W_o = nn.Linear(key_dim, embed_dim, bias=False)    # (D, E) E 次元ベクトル
        self.key_dim = key_dim

        # Dropoutを追加
        self.dropout = nn.Dropout(dropout_rate)

        # mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length))
        )

    def forward(self, x):   # x: (B, C, E)  # (batch_size, context_length, embed_dim)
        Q = self.W_q(x)     # Q: (B, C, D)  # (batch_size, context_length, key_dim)
        #print("Q.shape:", Q.shape)  # Q.shape: torch.Size([4, 9, 32])
        K = self.W_k(x)     # K: (B, C, D)  # (batch_size, context_length, key_dim)
        V = self.W_v(x)     # V: (B, C, D)  # (batch_size, context_length, key_dim)

        # Attention重みの計算
        K_t = K.transpose(-2, -1)      # (B, D, C)
        scores = torch.matmul(Q, K_t)   # (B, C, D)@(B, D, C) -> (B, C, C)
        scores = scores / (self.key_dim ** 0.5)
        #print("scores.shape:", scores.shape)    # scores.shape: torch.Size([4, 9, 9])

        # マスクの適用
        #B, C, E = x.shape
        _, C, _ = x.shape
        #mask = torch.tril(torch.ones(C, C, device=scores.device))
        #scores = scores.masked_fill(mask == 0, float('-inf'))
        scores = scores.masked_fill(
            self.mask[:C, :C] == 0,
            float("-inf")
        )
        weights = F.softmax(scores, dim=-1)
        #print("weights.shape", weights.shape)   # weights.shape torch.Size([4, 9, 9])
        weights = self.dropout(weights)
        context = torch.matmul(weights, V)   # (B, C, C)@(B, C, D) -> (B, C, D)

        output = self.W_o(context)           # (B, C, D)@(D, E) -> (B, C, E)
        #print("output.shape:", output.shape)    # output.shape: torch.Size([4, 9, 32])
        return output
    
#--------------------------------------------------
# LayerNorm

class LayerNorm(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * norm_x + self.beta

#--------------------------------------------------
# MLP(Multi-Layer Perceptron)

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):
        return self.layers(x)
#--------------------------------------------------

GPT_CONFIG = {
    "vocab_size": 13,
    "context_length": 9,
    "embed_dim": 32,
    "n_heads": 1,
    "n_layers": 12,
    "dropout_rate": 0.1,
    #"qkv_bias": False
}
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = Attention(
            embed_dim=cfg["embed_dim"],
            key_dim=cfg["embed_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["dropout_rate"])
        self.mlp = MLP(cfg["embed_dim"])
        self.ln1 = LayerNorm(cfg["embed_dim"])
        self.ln2 = LayerNorm(cfg["embed_dim"])
        self.drop = nn.Dropout(cfg["dropout_rate"])

    def forward(self, x):
        #x = x + self.att(self.ln1(x))
        #x = x + self.mlp(self.ln2(x))
        x = x + self.drop(self.att(self.ln1(x)))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

#--------------------------------------------------

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embed_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embed_dim"])
        self.drop_emb = nn.Dropout(cfg["dropout_rate"])
        #self.trf_blocks = nn.Sequential(
        #    *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        #)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["embed_dim"])
        self.out_head = nn.Linear(
            cfg["embed_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        B, C = in_idx.shape    # B: バッチサイズ、C: コンテキスト長
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(C, dtype=torch.long, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        #x = self.trf_blocks(x)
        for block in self.trf_blocks:
            x = block(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
#--------------------------------------------------
