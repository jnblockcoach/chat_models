"""
问答模型 - 测试脚本
加载训练好的模型进行问答示例展示
"""
import json
import torch
import torch.nn.functional as F
from config import device, max_context_len, max_query_len
from model import QAModel


def load_model(model_path="qa_model.pth"):
    """加载训练好的模型"""
    model = QAModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型加载成功! 参数量: {model.count_parameters():,}")
    return model


def load_word2idx(path="word2idx.json"):
    """加载词典"""
    with open(path, "r", encoding="utf-8") as f:
        word2idx = json.load(f)
    print(f"词典加载成功, 词汇量: {len(word2idx)}")
    return word2idx


def text_to_sequence(text, word2idx, max_len):
    """将文本转为序列"""
    unk_idx = word2idx.get("<UNK>", 1)
    seq = [word2idx.get(char, unk_idx) for char in text]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [0] * (max_len - len(seq))
    return seq


def answer_question(context, question, model, word2idx):
    """回答问问题"""
    c_seq = text_to_sequence(context, word2idx, max_context_len)
    q_seq = text_to_sequence(question, word2idx, max_query_len)

    x_c = torch.tensor([c_seq], dtype=torch.long).to(device)
    x_q = torch.tensor([q_seq], dtype=torch.long).to(device)
    q_mask = (x_q != 0).to(device)

    with torch.no_grad():
        start_logits, end_logits = model(x_c, x_q, q_mask)

    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    # 找最佳区间
    best_start = 0
    best_end = 0
    best_score = -1

    for s in range(min(len(context), max_context_len)):
        for e in range(s, min(s + 10, min(len(context), max_context_len))):
            score = start_probs[0, s].item() * end_probs[0, e].item()
            if score > best_score:
                best_score = score
                best_start = s
                best_end = e

    answer = context[best_start:best_end + 1]
    return answer, best_score


def test():
    """测试函数"""
    print("=" * 50)
    print("问答模型 - 测试")
    print("=" * 50)

    import os
    if not os.path.exists("qa_model.pth") or not os.path.exists("word2idx.json"):
        print("\n未找到模型文件 (qa_model.pth) 或词典文件 (word2idx.json)")
        print("请先运行 train.py 训练模型")
        return

    # 加载模型和词典
    model = load_model("qa_model.pth")
    word2idx = load_word2idx("word2idx.json")

    # 评估测试数据
    print("\n生成测试数据并评估...")
    from train import generate_synthetic_data, find_answer_in_seq
    test_data = generate_synthetic_data(500)
    split = int(0.8 * 3000)
    test_data = test_data[split:]

    correct = 0
    total = len(test_data)
    for context, question, answer, _, _ in test_data:
        pred_answer, _ = answer_question(context, question, model, word2idx)
        if pred_answer == answer:
            correct += 1

    print(f"问答准确率 (正确答案匹配): {100 * correct / total:.2f}%")

    # 示例问答
    print("\n" + "=" * 50)
    print("问答示例展示:")
    print("=" * 50)

    test_cases = [
        ("张三出生于1990年，是一位著名的计算机科学家。", "张三出生于哪一年？"),
        ("北京是中国首都，位于华北地区。", "中国的首都是哪里？"),
        ("长江是中国最长的河流，全长约6300公里。", "中国最长的河流是什么？"),
        ("清华大学成立于1911年，是中国最著名的大学之一。", "清华大学成立于哪一年？"),
        ("阿里巴巴集团由马云于1999年创立。", "阿里巴巴由谁创立？"),
    ]

    for context, question in test_cases:
        answer, confidence = answer_question(context, question, model, word2idx)
        print(f"\n上下文: {context}")
        print(f"问题: {question}")
        print(f"答案: {answer} (置信度: {confidence:.4f})")


if __name__ == "__main__":
    test()
