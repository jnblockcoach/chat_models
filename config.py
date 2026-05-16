"""
问答模型 - 防过拟合精调配置 (175M参数量级)
"""
import torch

# 词汇表
vocab_size = 32000
embed_dim = 768
hidden_dim = 768
num_model_layers = 4
num_heads = 12

# 训练
lr = 3e-4
epochs = 50            # 多训一点，但早停会提前截断
batch_size = 8

# ★ 防过拟合三件套 ★
dropout = 0.3           # ↑ 0.1→0.3 大幅增强
attn_dropout = 0.2      # 注意力额外dropout
weight_decay = 0.1      # ↑ 0.01→0.1 更强的权重衰减
label_smoothing = 0.1   # 标签平滑，防止过度自信
warmup_steps = 500      # LR预热，训练初期不过拟合
grad_clip = 1.0         # 梯度裁剪阈值
stochastic_depth = 0.1  # 随机丢弃整层 (DropPath)
early_stop_patience = 8 # 验证集连8轮不降就停

# 数据
max_context_len = 128
max_query_len = 24
num_train = 100000      # 10万条训练数据

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
