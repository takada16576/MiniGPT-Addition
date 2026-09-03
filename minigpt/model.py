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
class MultiHeadAttentionV1(nn.Module):
    def __init__(self, embed_dim, key_dim, context_length, dropout_rate, n_heads):
        super().__init__()
        assert (embed_dim % n_heads == 0), "embed_dim must be divisible by n_heads"
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        E, H, D = embed_dim, n_heads, self.head_dim
        # Q, K, Vの変換行列   nn.Linear は最後の次元だけに作用します
        self.W_q = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_k = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_v = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_o = nn.Linear(H*D, E, bias=False)    # (H*D, E) E 次元ベクトル

        # Dropoutを追加
        self.dropout = nn.Dropout(dropout_rate)

        # mask
        #self.register_buffer(
        #    "mask",
        #    torch.tril(torch.ones(context_length, context_length), diagonal=1)
        #)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length), diagonal=0)
        )

    def forward(self, x):   # x: (B, C, E)  # (batch_size, context_length, embed_dim)
        B, C, E = x.shape
        #print(f"C = {C}")
        H, D = self.n_heads, self.head_dim

        # Q, K, V の計算.    (B, C, E)@(E, H*D) => (B, C, H*D)
        Q = self.W_q(x)     # Q: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        K = self.W_k(x)     # K: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        V = self.W_v(x)     # V: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)

        # 各ヘッドに分割して並べ替え
        Q = Q.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        K = K.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        V = V.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)

        # Attention重みの計算
        K_t = K.transpose(-2, -1)       # (B, H, D, C)
        attn_scores = torch.matmul(Q, K_t)   # (B, H, C, D)@(B, H, D, C) -> (B, H, C, C)
        attn_scores = attn_scores / (D ** 0.5)

        # マスクの適用
        attn_scores = attn_scores.masked_fill(
            self.mask[:C, :C] == 0,
            float("-inf")
        )

        # Attention重み
        attn_weights = F.softmax(attn_scores, dim=-1)     # (B, H, C, C)

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)   # (B, H, C, C)@(B, H, C, D) -> (B, H, C, D)

        # ヘッドの結合と出力変換
        context = context.transpose(1, 2).contiguous() # (B, C, H, D)
        context = context.reshape(B, C, H*D)          # (B. C, H*D)
        output = self.W_o(context)           # (B, C, H*D)@(H*D, E) -> (B, C, E)
        #print("output.shape:", output.shape)    # output.shape: torch.Size([4, 9, 32])
        return output

#--------------------------------------------------
class MultiHeadAttentionV2(nn.Module):
    def __init__(self, embed_dim, key_dim, context_length, dropout_rate, n_heads):
        super().__init__()
        assert (embed_dim % n_heads == 0), "embed_dim must be divisible by n_heads"
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        E, H, D = embed_dim, n_heads, self.head_dim
        # Q, K, Vの変換行列   nn.Linear は最後の次元だけに作用します
        self.W_q = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_k = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_v = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_o = nn.Linear(H*D, E, bias=False)    # (H*D, E) E 次元ベクトル

        # Dropoutを追加
        self.dropout = nn.Dropout(dropout_rate)

        # mask
        #self.register_buffer(
        #    "mask",
        #    torch.tril(torch.ones(context_length, context_length), diagonal=1)
        #)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length), diagonal=0)
        )

    def forward(self, x):   # x: (B, C, E)  # (batch_size, context_length, embed_dim)
        B, C, E = x.shape
        #print(f"C = {C}")
        H, D = self.n_heads, self.head_dim

        # Q, K, V の計算.    (B, C, E)@(E, H*D) => (B, C, H*D)
        Q = self.W_q(x)     # Q: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        K = self.W_k(x)     # K: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        V = self.W_v(x)     # V: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)

        # 各ヘッドに分割して並べ替え
        Q = Q.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        K = K.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        V = V.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)

        # Attention重みの計算
        K_t = K.transpose(-2, -1)       # (B, H, D, C)
        attn_scores = torch.matmul(Q, K_t)   # (B, H, C, D)@(B, H, D, C) -> (B, H, C, C)
        attn_scores = attn_scores / (D ** 0.5)

        # マスクの適用
        attn_scores = attn_scores.masked_fill(
            self.mask[:C, :C] == 0,
            float("-inf")
        )

        # Softmax前のスコアを保存
        self.last_attn_scores = attn_scores.detach().cpu()

        # Attention重み
        attn_weights = F.softmax(attn_scores, dim=-1)     # (B, H, C, C)

        # Softmax後の重みを保存（可視化用）
        self.last_attn_weights = attn_weights.detach().cpu()

        if not hasattr(self, "attn_history"):
            self.attn_history = []
        self.attn_history.append(attn_weights.detach().cpu())

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)   # (B, H, C, C)@(B, H, C, D) -> (B, H, C, D)

        # ヘッドの結合と出力変換
        context = context.transpose(1, 2).contiguous() # (B, C, H, D)
        context = context.reshape(B, C, H*D)          # (B. C, H*D)
        output = self.W_o(context)           # (B, C, H*D)@(H*D, E) -> (B, C, E)
        #print("output.shape:", output.shape)    # output.shape: torch.Size([4, 9, 32])
        return output

#--------------------------------------------------
class MultiHeadAttentionV3(nn.Module):
    def __init__(self, embed_dim, key_dim, context_length, dropout_rate, n_heads):
        super().__init__()
        assert (embed_dim % n_heads == 0), "embed_dim must be divisible by n_heads"
        self.key_dim = key_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        E, H, D = embed_dim, n_heads, self.head_dim
        # Q, K, Vの変換行列   nn.Linear は最後の次元だけに作用します
        self.W_q = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_k = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_v = nn.Linear(E, H*D, bias=False)    # (E, H*D) H*D 次元ベクトル
        self.W_o = nn.Linear(H*D, E, bias=False)    # (H*D, E) E 次元ベクトル
        
        # Dropoutを追加
        self.dropout = nn.Dropout(dropout_rate)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length), diagonal=0)
        )

        self.use_fixed_attention=False


    def forward(self, x):   # x: (B, C, E)  # (batch_size, context_length, embed_dim)
        B, C, E = x.shape
        #print(f"C = {C}")
        H, D = self.n_heads, self.head_dim

        # Q, K, V の計算.    (B, C, E)@(E, H*D) => (B, C, H*D)
        Q = self.W_q(x)     # Q: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        K = self.W_k(x)     # K: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)
        V = self.W_v(x)     # V: (B, C, H*D)  # (batch_size, context_length, n_head*key_dim)

        # 各ヘッドに分割して並べ替え
        Q = Q.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        K = K.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)
        V = V.reshape(B, C, H, D).transpose(1, 2)   # (B, H, C, D)

        # --------------------------------------------------------------
        if self.use_fixed_attention == True:
            # 固定Attention
            # -------------------------------
            # 固定Attention（一様分布）
            # -------------------------------
            mask = self.mask[:C, :C].float()      # (C, C)

            # 各行が1になるように正規化
            attn_weights = mask / mask.sum(dim=-1, keepdim=True)

            # (1,1,C,C) -> (B,H,C,C)
            attn_weights = attn_weights.unsqueeze(0).unsqueeze(0)
            attn_weights = attn_weights.expand(B, H, C, C)
        else:
            # 通常のAttention
            # Attention重みの計算
            K_t = K.transpose(-2, -1)       # (B, H, D, C)
            attn_scores = torch.matmul(Q, K_t)   # (B, H, C, D)@(B, H, D, C) -> (B, H, C, C)
            attn_scores = attn_scores / (D ** 0.5)
            # マスクの適用
            attn_scores = attn_scores.masked_fill(
                self.mask[:C, :C] == 0,
                float("-inf")
            )
            # Attention重み
            attn_weights = F.softmax(attn_scores, dim=-1)     # (B, H, C, C)
        # --------------------------------------------------------------

        

        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)   # (B, H, C, C)@(B, H, C, D) -> (B, H, C, D)

        # ヘッドの結合と出力変換
        context = context.transpose(1, 2).contiguous() # (B, C, H, D)
        context = context.reshape(B, C, H*D)          # (B. C, H*D)
        output = self.W_o(context)           # (B, C, H*D)@(H*D, E) -> (B, C, E)
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


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttentionV3(
            embed_dim=cfg["embed_dim"],
            key_dim=cfg["embed_dim"],
            context_length=cfg["context_length"],
            dropout_rate=cfg["dropout_rate"],
            n_heads=cfg["n_heads"])
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
GPT_CONFIG = {
    "vocab_size": 16,       # vocab_size = len(all_words)
    "context_length": 91,   # cotの時は91、"12+34=046"の時は9、"+12++34=+046"の時は12、"+12*+34=+0408"の時は13, "+0012*+0034=+000000408"の時は22
    "embed_dim": 256,       # 32 -> 64 -> 96 -> 128 -> 192 -> 256
    "n_heads": 4,           # 2 -> 1 -> 3
    "n_layers": 12,
    "dropout_rate": 0.1,
    #"qkv_bias": False
}
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
