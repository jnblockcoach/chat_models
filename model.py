"""
问答模型 - BiDAF风格模型

参数量估算:
- 词嵌入: 5000 * 128 = 640,000
- Context BiLSTM: 4*(128*256+256²+256)*2 = 788,480
- Query BiLSTM: 4*(128*256+256²+256)*2 = 788,480
- BiDAF Attention (W*+v): ~262K
- 投影层 (4*512→256): 524,544
- Modeling BiLSTM (2层): 2 × 4*(256*256+256²+256)*2 = 2,101,248
- 输出层: 256*2+2 = 514
- 总计: ~4.1M ≈ 4M
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import vocab_size, embed_dim, hidden_dim, dropout, device


class BiLSTMEncoder(nn.Module):
    """BiLSTM编码器"""
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BiLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

    def forward(self, x):
        outputs, _ = self.lstm(x)
        return outputs


class BiDAFAttention(nn.Module):
    """BiDAF注意力流层"""
    def __init__(self, hidden_dim):
        super(BiDAFAttention, self).__init__()
        self.hidden_dim = hidden_dim
        # 相似度计算: S = w^T * tanh(W1*c + W2*q + W3*c⊙q)
        self.W1 = nn.Linear(hidden_dim * 2, 1, bias=False)  # c投影
        self.W2 = nn.Linear(hidden_dim * 2, 1, bias=False)  # q投影

    def forward(self, context, query, query_mask=None):
        """
        context: [batch, ctx_len, hidden*2]
        query: [batch, qry_len, hidden*2]
        """
        batch_size, ctx_len, dim = context.shape
        qry_len = query.size(1)

        # 计算相似度矩阵 S [batch, ctx_len, qry_len]
        # 简化的双线性相似度
        c_flat = context.unsqueeze(2)  # [batch, ctx_len, 1, dim]
        q_flat = query.unsqueeze(1)    # [batch, 1, qry_len, dim]

        # 使用简化的相似度: tanh(W1*c + W2*q + c⊙q)
        # 这里简化计算
        W1_c = self.W1(c_flat)  # [batch, ctx_len, 1, 1]
        W2_q = self.W2(q_flat)  # [batch, 1, qry_len, 1]
        dot = (c_flat * q_flat).sum(dim=-1, keepdim=True)  # [batch, ctx_len, qry_len, 1]

        S = (W1_c + W2_q + dot).squeeze(-1)  # [batch, ctx_len, qry_len]

        # Mask query padding
        if query_mask is not None:
            S = S.masked_fill(~query_mask.unsqueeze(1), -1e9)

        # Context-to-Query (C2Q) Attention
        c2q_weights = F.softmax(S, dim=2)  # [batch, ctx_len, qry_len]
        c2q = torch.bmm(c2q_weights, query)  # [batch, ctx_len, dim]

        # Query-to-Context (Q2C) Attention
        q2c_weights = F.softmax(S.max(dim=2, keepdim=True)[0], dim=1)  # [batch, ctx_len, 1]
        q2c = q2c_weights * context  # [batch, ctx_len, dim]

        # 拼接: [c, c2q, c⊙c2q, c⊙q2c]
        combined = torch.cat([
            context,
            c2q,
            context * c2q,
            context * q2c
        ], dim=2)  # [batch, ctx_len, dim * 4]

        return combined, S


class QAModel(nn.Module):
    """问答模型 (BiDAF风格)"""
    def __init__(self):
        super(QAModel, self).__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Context和Query编码
        self.context_lstm = BiLSTMEncoder(embed_dim, hidden_dim)
        self.query_lstm = BiLSTMEncoder(embed_dim, hidden_dim)

        # BiDAF注意力
        self.bidaf_attn = BiDAFAttention(hidden_dim)

        # 投影层 (4*hidden*2 → hidden*2)
        self.projection = nn.Linear(hidden_dim * 8, hidden_dim * 2)

        # Modeling层 (2层BiLSTM)
        self.modeling_lstm = nn.LSTM(
            input_size=hidden_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        # 输出层 (预测开始/结束位置)
        self.output_start = nn.Linear(hidden_dim * 2, 1)
        self.output_end = nn.Linear(hidden_dim * 2, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, context_ids, query_ids, query_mask=None):
        """
        context_ids: [batch, ctx_len]
        query_ids: [batch, qry_len]
        """
        # 词嵌入
        c_emb = self.dropout(self.embedding(context_ids))  # [batch, ctx_len, embed_dim]
        q_emb = self.dropout(self.embedding(query_ids))    # [batch, qry_len, embed_dim]

        # 编码
        c_encoded = self.context_lstm(c_emb)  # [batch, ctx_len, hidden*2]
        q_encoded = self.query_lstm(q_emb)    # [batch, qry_len, hidden*2]

        # BiDAF注意力
        g, S = self.bidaf_attn(c_encoded, q_encoded, query_mask)  # [batch, ctx_len, hidden*8]

        # 投影
        g = self.dropout(g)
        g_proj = F.relu(self.projection(g))  # [batch, ctx_len, hidden*2]

        # Modeling层
        m, _ = self.modeling_lstm(g_proj)  # [batch, ctx_len, hidden*2]
        m = self.dropout(m)

        # 预测开始位置
        start_logits = self.output_start(m).squeeze(-1)  # [batch, ctx_len]
        # 预测结束位置 (使用 Modeling 输出 + 门控)
        end_logits = self.output_end(m).squeeze(-1)      # [batch, ctx_len]

        return start_logits, end_logits

    def predict_answer_span(self, context_ids, query_ids, context_text):
        """预测答案区间"""
        self.eval()
        query_mask = (query_ids != 0).unsqueeze(0)

        with torch.no_grad():
            start_logits, end_logits = self(context_ids, query_ids, query_mask)

        start_probs = F.softmax(start_logits, dim=1)
        end_probs = F.softmax(end_logits, dim=1)

        # 找到最佳区间 (start <= end)
        batch_size = start_probs.size(0)
        best_start = 0
        best_end = 0
        best_score = -1

        for s in range(context_ids.size(1)):
            for e in range(s, min(s + 15, context_ids.size(1))):  # 限制最大15个token
                score = start_probs[0, s].item() * end_probs[0, e].item()
                if score > best_score:
                    best_score = score
                    best_start = s
                    best_end = e

        return best_start, best_end, best_score

    def count_parameters(self):
        """统计可训练参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
