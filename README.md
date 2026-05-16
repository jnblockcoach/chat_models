# 问答模型 (Question Answering)

## 概述
基于 BiDAF (Bidirectional Attention Flow) 风格的问答模型，从上下文中定位答案。

## 模型结构
- 词嵌入层: vocab_size=5000, embed_dim=128
- Context/Query BiLSTM 编码: hidden_dim=256（双向）
- BiDAF 注意力流层: C2Q + Q2C 双向注意力
- 投影层: 8×hidden → 2×hidden
- Modeling BiLSTM: 2层双向 LSTM
- 输出层: 预测答案开始/结束位置

## 参数量
~4.1M (≈ 4M)

## 配置 (config.py)
| 参数 | 值 |
|------|-----|
| vocab_size | 5000 |
| embed_dim | 128 |
| hidden_dim | 256 |
| lr | 0.001 |
| epochs | 10 |
| batch_size | 32 |
| max_context_len | 80 |
| max_query_len | 20 |

## 使用方法

### 训练
```bash
cd nlp/qa_model
python train.py
```

### 测试
```bash
python test.py
```

## 数据
使用合成中文问答数据，包含20组上下文-问题-答案模板，共3000条训练数据。

## 文件结构
```
qa_model/
├── config.py    # 配置文件
├── model.py     # 模型定义 (BiDAF风格)
├── train.py     # 训练脚本
├── test.py      # 测试脚本 (含问答示例)
└── README.md    # 本文件
```
