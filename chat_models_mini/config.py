"""
问答模型 - 真·迷你配置 (~6.6M参数, CPU友好)
"""
import torch

# 词汇表 & 嵌入
vocab_size = 8000      # 大降! 30K→8K
embed_dim = 128        # 768→128
hidden_dim = 128       # 768→128
num_model_layers = 2   # 4→2
num_heads = 4          # 12→4

# 训练
lr = 3e-4
epochs = 10            # 多跑几轮
batch_size = 16        # batch大点CPU更快

# 防过拟合 (迷你模型没那么容易过拟合)
dropout = 0.2
attn_dropout = 0.1
weight_decay = 0.01
label_smoothing = 0.05
warmup_steps = 30
grad_clip = 1.0
stochastic_depth = 0.05
early_stop_patience = 10

# 数据
max_context_len = 96   # 短点, 加快训练
max_query_len = 20
num_train = 2000

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
