"""
问答模型 - 精调版 (~175M参数)
架构: BiDAF attention + 4× Transformer Modeling Layers

★ 防过拟合设计 ★
- Dropout 0.3 (嵌入/输出/投影前)
- Attention Dropout 0.2
- Stochastic Depth 0.1 (随机丢弃Transformer层)
- LayerNorm预归一化 (更稳定)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    vocab_size, embed_dim, hidden_dim, num_model_layers, num_heads,
    dropout, attn_dropout, stochastic_depth,
    device, max_context_len, max_query_len
)


# ═══════════════════════════════════════════════
#  Stochastic Depth (DropPath)
# ═══════════════════════════════════════════════

class DropPath(nn.Module):
    """随机丢弃整个路径 (训练时随机跳过某个Transformer层)"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device) < keep_prob
        return x / keep_prob * mask


# ═══════════════════════════════════════════════
#  Transformer 组件
# ═══════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """多头注意力+Attention Dropout"""
    def __init__(self, d_model, num_heads=12):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=attn_dropout, training=self.training)

        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """SwiGLU FFN (比ReLU门控更稳定)"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Transformer块 (LayerNorm预归一化 + DropPath)"""
    def __init__(self, d_model, d_ff, num_heads=12, layer_idx=0):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_path_drop = DropPath(stochastic_depth * layer_idx / num_model_layers)

        self.ffn_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.ffn = FeedForward(d_model, d_ff)
        self.ffn_drop = nn.Dropout(dropout)
        self.ffn_path_drop = DropPath(stochastic_depth * layer_idx / num_model_layers)

    def forward(self, x):
        # Pre-Norm + residual (更稳定的梯度)
        x = x + self.attn_path_drop(self.attn_drop(self.attn(self.attn_norm(x))))
        x = x + self.ffn_path_drop(self.ffn_drop(self.ffn(self.ffn_norm(x))))
        return x


# ═══════════════════════════════════════════════
#  BiDAF 组件
# ═══════════════════════════════════════════════

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0,  # 只用1层，dropout不生效
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class BiDAFAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_c = nn.Linear(d, 1, bias=False)
        self.W_q = nn.Linear(d, 1, bias=False)

    def forward(self, context, query):
        B, T_c, D = context.shape
        T_q = query.size(1)

        c = context.unsqueeze(2)
        q = query.unsqueeze(1)

        S = self.W_c(c) + self.W_q(q) + (c * q).sum(dim=-1, keepdim=True)
        S = S.squeeze(-1)

        c2q = torch.bmm(F.softmax(S, dim=2), query)
        q2c_w = F.softmax(S.max(dim=2, keepdim=True)[0], dim=1)
        q2c = q2c_w * context

        return torch.cat([context, c2q, context * c2q, context * q2c], dim=2)


# ═══════════════════════════════════════════════
#  主模型
# ═══════════════════════════════════════════════

class ChatModel(nn.Module):
    """
    BiDAF + Transformer (~175M)
    """
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_context_len + max_query_len, embed_dim)
        self.embed_drop = nn.Dropout(dropout)

        self.context_lstm = BiLSTMEncoder(embed_dim, hidden_dim)
        self.query_lstm = BiLSTMEncoder(embed_dim, hidden_dim)

        self.bidaf = BiDAFAttention(hidden_dim * 2)

        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 8, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
        )

        d_model = hidden_dim * 2
        d_ff = d_model * 3
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, d_ff, num_heads, i)
            for i in range(num_model_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.out_drop = nn.Dropout(dropout)
        self.output_start = nn.Linear(d_model, 1, bias=False)
        self.output_end = nn.Linear(d_model, 1, bias=False)

        self._init_weights()

    def _init_weights(self):
        """小初始值 + 残差缩放 (防止深层梯度爆炸)"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)
        # 输出层缩小初始值
        nn.init.normal_(self.output_start.weight, mean=0.0, std=0.001)
        nn.init.normal_(self.output_end.weight, mean=0.0, std=0.001)

    def forward(self, context_ids, query_ids):
        B, T_c = context_ids.shape
        T_q = query_ids.size(1)

        pos_c = torch.arange(T_c, device=context_ids.device).unsqueeze(0)
        pos_q = torch.arange(T_q, device=query_ids.device).unsqueeze(0)

        c_emb = self.embed_drop(self.embedding(context_ids) + self.pos_embed(pos_c))
        q_emb = self.embed_drop(self.embedding(query_ids) + self.pos_embed(pos_q))

        c_enc = self.context_lstm(c_emb)
        q_enc = self.query_lstm(q_emb)

        g = self.bidaf(c_enc, q_enc)
        g = self.projection(g)

        for layer in self.transformer_layers:
            g = layer(g)

        g = self.out_norm(g)
        g = self.out_drop(g)

        start = self.output_start(g).squeeze(-1)
        end = self.output_end(g).squeeze(-1)

        return start, end

    def predict_answer(self, context_ids, query_ids):
        self.eval()
        with torch.no_grad():
            start_logits, end_logits = self(context_ids, query_ids)

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        T = context_ids.size(1)
        best_start, best_end, best_score = 0, 0, -1
        for s in range(T):
            for e in range(s, min(s + 20, T)):
                score = start_probs[0, s].item() * end_probs[0, e].item()
                if score > best_score:
                    best_score = score
                    best_start, best_end = s, e
        return best_start, best_end, best_score


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = ChatModel().to('cpu')
    total = count_parameters(model)
    print(f"ChatModel 参数量: {total:,} ({total/1e6:.1f}M)")
    # 快速前向
    ctx = torch.randint(1, 5000, (1, 16))
    qry = torch.randint(1, 5000, (1, 8))
    s, e = model(ctx, qry)
    print(f"前向OK -> start={list(s.shape)} end={list(e.shape)}")
