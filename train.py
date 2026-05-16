"""
问答模型 - 训练脚本
生成合成问答数据 (上下文+问题+答案区间), 训练
"""
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import vocab_size, batch_size, epochs, lr, device, seed, max_context_len, max_query_len
from model import QAModel

# 设置随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ========== 合成问答数据 ==========

# 上下文模板 (包含答案)
qa_pairs = [
    ("张三出生于1990年，是一位著名的计算机科学家。", "张三出生于哪一年？", "1990年"),
    ("北京是中国首都，位于华北地区。", "中国的首都是哪里？", "北京"),
    ("长江是中国最长的河流，全长约6300公里。", "中国最长的河流是什么？", "长江"),
    ("华为技术有限公司是一家中国科技公司，总部在深圳。", "华为的总部在哪里？", "深圳"),
    ("清华大学成立于1911年，是中国最著名的大学之一。", "清华大学成立于哪一年？", "1911年"),
    ("阿里巴巴集团由马云于1999年创立。", "阿里巴巴由谁创立？", "马云"),
    ("上海是中国最大的城市，人口超过2400万。", "中国最大的城市是哪里？", "上海"),
    ("人工智能是计算机科学的一个分支，始于1956年。", "人工智能始于哪一年？", "1956年"),
    ("太阳系有八大行星，其中最大的是木星。", "太阳系最大的行星是什么？", "木星"),
    ("珠穆朗玛峰高约8848米，是世界最高峰。", "世界最高峰是什么？", "珠穆朗玛峰"),
    ("人体有206块骨骼，其中最长的骨是股骨。", "人体最长的骨骼是什么？", "股骨"),
    ("水在100°C时沸腾，在0°C时结冰。", "水在多少度时沸腾？", "100°C"),
    ("孔子是春秋时期的思想家和教育家。", "孔子是什么时期的人？", "春秋时期"),
    ("地球绕太阳公转一周需要365天。", "地球公转一周需要多少天？", "365天"),
    ("《红楼梦》是曹雪芹写的中国古典小说。", "《红楼梦》的作者是谁？", "曹雪芹"),
    ("月球是地球唯一的天然卫星。", "地球唯一的天然卫星是什么？", "月球"),
    ("牛顿发现了万有引力定律。", "谁发现了万有引力定律？", "牛顿"),
    ("长城全长约21000公里，始建于秦朝。", "长城始建于哪个朝代？", "秦朝"),
    ("DNA的全称是脱氧核糖核酸。", "DNA的全称是什么？", "脱氧核糖核酸"),
    ("氧气在1774年被普里斯特利发现。", "氧气在哪一年被发现？", "1774年"),
]

def generate_synthetic_data(num_samples=3000):
    """生成合成问答数据"""
    data = []

    while len(data) < num_samples:
        context, question, answer = random.choice(qa_pairs)

        # 找到答案在context中的位置 (字符级别)
        answer_pos = context.find(answer)
        if answer_pos == -1:
            continue

        data.append((context, question, answer, answer_pos, answer_pos + len(answer)))

    return data[:num_samples]


# ========== 构建词典 ==========

def build_vocab(all_texts):
    """构建词汇表"""
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    idx2word = {0: "<PAD>", 1: "<UNK>"}
    word_count = {}

    for text in all_texts:
        for char in text:
            word_count[char] = word_count.get(char, 0) + 1

    sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
    for word, _ in sorted_words[:vocab_size - 2]:
        idx = len(word2idx)
        word2idx[word] = idx
        idx2word[idx] = word

    return word2idx, idx2word


def text_to_sequence(text, word2idx, max_len):
    """将文本转为序列"""
    seq = [word2idx.get(char, word2idx["<UNK>"]) for char in text]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return seq


def find_answer_in_seq(answer_start_char, answer_text, context_text, max_len):
    """找到答案在序列中的开始和结束位置"""
    # 截断context
    if len(context_text) > max_len:
        context_text = context_text[:max_len]

    start_pos = context_text.find(answer_text[:min(5, len(answer_text))])
    if start_pos == -1:
        start_pos = 0
    end_pos = min(start_pos + len(answer_text), len(context_text)) - 1
    if start_pos >= len(context_text):
        start_pos = len(context_text) - 1

    return max(0, start_pos), max(start_pos, min(end_pos, len(context_text) - 1))


# ========== 数据集 ==========

class QADataset(Dataset):
    """问答数据集"""
    def __init__(self, data, word2idx, max_c_len, max_q_len):
        self.data = data
        self.word2idx = word2idx
        self.max_c_len = max_c_len
        self.max_q_len = max_q_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        context, question, answer, ans_start_char, ans_end_char = self.data[idx]

        c_seq = text_to_sequence(context, self.word2idx, self.max_c_len)
        q_seq = text_to_sequence(question, self.word2idx, self.max_q_len)

        # 找到答案在序列中的位置
        start_idx, end_idx = find_answer_in_seq(
            ans_start_char, answer, context, self.max_c_len
        )

        return (
            torch.tensor(c_seq, dtype=torch.long),
            torch.tensor(q_seq, dtype=torch.long),
            torch.tensor(start_idx, dtype=torch.long),
            torch.tensor(end_idx, dtype=torch.long)
        )


# ========== 训练 ==========

def train():
    """主训练函数"""
    print("=" * 50)
    print("问答模型训练 (BiDAF风格)")
    print("=" * 50)

    # 生成数据
    print("\n生成合成问答数据...")
    all_data = generate_synthetic_data(3000)
    print(f"生成 {len(all_data)} 条问答数据")

    # 构建词典
    all_texts = []
    for c, q, a, _, _ in all_data:
        all_texts.append(c)
        all_texts.append(q)
        all_texts.append(a)

    word2idx, idx2word = build_vocab(all_texts)
    print(f"词汇表大小: {len(word2idx)}")

    # 划分训练集/测试集
    split = int(0.8 * len(all_data))
    train_data = all_data[:split]
    test_data = all_data[split:]

    train_dataset = QADataset(train_data, word2idx, max_context_len, max_query_len)
    test_dataset = QADataset(test_data, word2idx, max_context_len, max_query_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # 创建模型
    model = QAModel().to(device)
    total_params = model.count_parameters()
    print(f"\n模型可训练参数量: {total_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练循环
    print("\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_c, batch_q, batch_start, batch_end in train_loader:
            batch_c = batch_c.to(device)
            batch_q = batch_q.to(device)
            batch_start = batch_start.to(device)
            batch_end = batch_end.to(device)

            # 创建query mask
            q_mask = (batch_q != 0).to(device)

            optimizer.zero_grad()
            start_logits, end_logits = model(batch_c, batch_q, q_mask)

            loss_start = criterion(start_logits, batch_start)
            loss_end = criterion(end_logits, batch_end)
            loss = loss_start + loss_end

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader):.4f}")

    # 测试
    model.eval()
    correct_start = 0
    correct_end = 0
    total = 0

    with torch.no_grad():
        for batch_c, batch_q, batch_start, batch_end in test_loader:
            batch_c = batch_c.to(device)
            batch_q = batch_q.to(device)
            q_mask = (batch_q != 0).to(device)

            start_logits, end_logits = model(batch_c, batch_q, q_mask)

            _, pred_start = torch.max(start_logits, 1)
            _, pred_end = torch.max(end_logits, 1)

            correct_start += (pred_start.cpu() == batch_start).sum().item()
            correct_end += (pred_end.cpu() == batch_end).sum().item()
            total += batch_start.size(0)

    print(f"\n开始位置准确率: {100 * correct_start / total:.2f}%")
    print(f"结束位置准确率: {100 * correct_end / total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), "qa_model.pth")
    print("模型已保存到 qa_model.pth")

    import json
    with open("word2idx.json", "w", encoding="utf-8") as f:
        json.dump(word2idx, f, ensure_ascii=False)
    print("词典已保存到 word2idx.json")

    return model, word2idx, idx2word


if __name__ == "__main__":
    train()
