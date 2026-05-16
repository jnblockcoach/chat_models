# ChatModel — 问答模型 (175M参数)

## 架构
- **BiDAF 双向注意力流** + **4层 Transformer Encoder**
- 词嵌入: 32K × 768
- 隐藏层: 768 (双向LSTM)
- Transformer: d_model=1536, d_ff=4608, n_layers=4
- 总参数量: **~175.8M**

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 超参数配置 |
| model.py | 模型定义 (ChatModel) |
| train.py | 训练脚本 (合成数据 2万条) |
| test.py | 测试脚本 (问答演示) |
| README.md | 本文件 |

## 训练
```bash
python train.py
```
- 自动生成 20,000 条合成问答数据
- 使用 AdamW + CosineAnnealingLR
- 梯度累积 4步 (等效 batch=32)
- 输出: `chat_model.pth`, `vocab.json`

## 测试
```bash
python test.py    # 加载模型进行问答演示
```

## 参数细节
```
  嵌入层:       24,576,000
  Context-LSTM:  9,449,472
  Query-LSTM:    9,449,472
  BiDAF:              3,072
  投影层:         9,441,792
  Transformer×4:122,732,544
  输出层:              3,074
  ─────────────────────────
  总计:        175,775,234  (175.8M)
```
