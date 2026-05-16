# ChatModel Mini — 问答模型快速验证版

> 基于 [ChatModel (175M)](..\chat_models) 的精简快速验证版本

## 与完整版的区别

| 项目 | 完整版 | Mini版 |
|------|--------|--------|
| 训练数据 | 100,000 条 | 2,000 条 |
| 训练轮数 | 50 epochs | 5 epochs |
| LR Warmup | 500 steps | 50 steps |
| 预计耗时 (CPU) | 数小时~数天 | 几分钟~十几分钟 |

**用途：** 在本地快速验证代码是否能跑通、模型能否收敛，再上云 GPU 训完整版。

## 快速开始

```bash
# 1. 安装依赖
pip install torch numpy

# 2. 训练（几步分钟）
python train.py

# 3. 测试问答
python test.py
```

## 输出文件

- `chat_model.pth` — 训练好的模型权重
- `chat_model_best.pth` — 验证集最优的模型
- `vocab.json` — 字符词典
- `training_status.json` — 训练状态（可中断恢复）

## 完整版

完整训练配置在上级目录 `chat_models/` 中，建议在 GPU 上运行。
