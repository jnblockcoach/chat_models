"""
问答模型 - 配置文件
"""
import torch

# 词汇表大小
vocab_size = 50000
# 词嵌入维度
embed_dim = 128
# 隐藏层维度
hidden_dim = 256
# 学习率
lr = 0.001
# 训练轮数
epochs = 10
# 批量大小
batch_size = 32
# dropout率
dropout = 0.3
# 最大序列长度
max_context_len = 80
max_query_len = 20
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 随机种子
seed = 42
