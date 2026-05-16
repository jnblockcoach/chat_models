"""
问答模型 (175M) — 测试脚本
加载训练好的模型，演示问答预测
"""
import json
import torch
from config import max_context_len, max_query_len, device
from model import ChatModel

def text_to_seq(text, w2i, max_len):
    seq = [w2i.get(c, w2i["<UNK>"]) for c in text]
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq += [0] * (max_len - len(seq))
    return seq


def test():
    print("=" * 50)
    print("  ChatModel (175M) 问答测试")
    print("=" * 50)

    # 加载模型
    print("\n加载模型...")
    model = ChatModel().to(device)
    try:
        model.load_state_dict(torch.load("chat_model.pth", map_location=device))
        print("  ✅ 已加载 chat_model.pth")
    except:
        print("  ⚠️ 未找到训练好的模型，使用随机权重")
    model.eval()

    # 加载词典
    try:
        with open("vocab.json", "r", encoding="utf-8") as f:
            w2i = json.load(f)
            i2w = {v: k for k, v in w2i.items()}
            print(f"  ✅ 已加载 vocab.json ({len(w2i)} 词)")
    except:
        # 用简易词典
        chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789，。？！、：；""''（）《》—…·")
        chars.update("的了一是不是我人在有中大上这个们到说时他要来生会也子就出下去好看自家走过小得多你")
        w2i = {"<PAD>": 0, "<UNK>": 1}
        for c in sorted(chars):
            w2i[c] = len(w2i)
        i2w = {v: k for k, v in w2i.items()}
        print(f"  ⚠️ 未找到vocab.json，使用默认词典 ({len(w2i)} 词)")

    # 测试用例
    test_cases = [
        ("北京是中国首都，位于华北地区。", "中国的首都是哪里？"),
        ("长江是中国最长的河流，全长约6300公里。", "中国最长的河流是什么？"),
        ("清华大学成立于1911年，是中国著名的大学。", "清华大学成立于哪一年？"),
        ("太阳系有八大行星，最大的是木星。", "太阳系最大的行星是什么？"),
        ("珠穆朗玛峰高约8848米，是世界最高峰。", "世界最高峰是什么？"),
    ]

    print(f"\n问答演示 ({len(test_cases)} 个问题):")
    for ctx, q in test_cases:
        c_seq = torch.tensor([text_to_seq(ctx, w2i, max_context_len)]).to(device)
        q_seq = torch.tensor([text_to_seq(q, w2i, max_query_len)]).to(device)

        with torch.no_grad():
            start, end = model(c_seq, q_seq)
            pred_s = start.argmax(1).item()
            pred_e = end.argmax(1).item()

        # 抽取答案文本
        if pred_s <= pred_e and pred_e < len(ctx):
            answer = ctx[pred_s:pred_e + 1]
        else:
            answer = f"[位置异常: {pred_s}~{pred_e}]"

        print(f"\n  上下文: {ctx}")
        print(f"  问题:   {q}")
        print(f"  预测:   {answer}")


if __name__ == "__main__":
    test()
