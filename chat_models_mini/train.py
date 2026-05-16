"""
ChatModel (175M) — 防过拟合训练脚本

★ 特色 ★
- 多样化数据增强 (50+模板、随机扰动、噪声注入)
- Label Smoothing + 高Dropout + 强Weight Decay
- 验证集监控 + 早停
- LR预热 + Cosine退火
- 每epoch保存最佳模型
"""
import random
import json
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config import (
    vocab_size, batch_size, epochs, lr, device, seed,
    max_context_len, max_query_len, dropout,
    weight_decay, label_smoothing, warmup_steps,
    grad_clip, early_stop_patience, num_train,
)

# 将 model.py 中的 ChatModel import 进来
import sys
sys.path.insert(0, os.path.dirname(__file__))
from model import ChatModel, count_parameters

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# ─── 训练状态文件 ─────────────────────────────────────
STATUS_FILE = os.path.join(os.path.dirname(__file__), "training_status.json")


# ═══════════════════════════════════════════════════
#  1. 问答语料 — 50+ 多样化模板
# ═══════════════════════════════════════════════════

qa_templates = [
    # ── 人物 ──
    ("{name}出生于{year}年，是一位著名的{profession}。", "{name}出生于哪一年？", "{year}年", 1),
    ("{name}出生于{year}年，是一位著名的{profession}。", "{name}是谁？", "{name}", 1),
    ("{name}是{era}的{profession}。", "{name}是什么时期的人？", "{era}", 1),
    ("{name}在{year}年发现了{discovery}。", "{discovery}是谁发现的？", "{name}", 1),

    # ── 地理 ──
    ("{city}是{country}的首都，位于{region}。", "{country}的首都是哪里？", "{city}", 0),
    ("{river}是{country}最长的河流，全长约{length}公里。", "{country}最长的河流是什么？", "{river}", 0),
    ("{city}是{country}最大的城市，人口超过{population}。", "{country}最大的城市是哪里？", "{city}", 0),
    ("{mountain}高约{height}米，是{region}最高峰。", "{region}最高峰是什么？", "{mountain}", 0),
    ("{country}位于{continent}。", "{country}位于哪个洲？", "{continent}", 0),
    ("{country}的官方语言是{language}。", "{country}的官方语言是什么？", "{language}", 0),

    # ── 公司/组织 ──
    ("{company}总部在{location}，成立于{year}年。", "{company}的总部在哪里？", "{location}", 0),
    ("{company}由{founder}于{year}年创立。", "{company}由谁创立？", "{founder}", 0),
    ("{company}成立于{year}年。", "{company}成立于哪一年？", "{year}年", 0),

    # ── 科学 ──
    ("{field}始于{year}年，是{science}的重要分支。", "{field}始于哪一年？", "{year}年", 1),
    ("{star}系有{num}颗{obj}，其中最大的是{biggest}。", "{star}系最大的{obj}是什么？", "{biggest}", 1),
    ("人体有{num1}块骨骼，其中最长的骨是{bone}。", "人体最长的骨骼是什么？", "{bone}", 1),
    ("{substance}在{temp1}时沸腾，在{temp2}时结冰。", "{substance}在多少度时沸腾？", "{temp1}", 1),
    ("地球绕太阳公转一周需要{days}天。", "地球公转一周需要多少天？", "{days}天", 1),
    ("{planet}是地球唯一的天然卫星。", "地球唯一的天然卫星是什么？", "{planet}", 1),
    ("DNA的全称是{full_name}。", "DNA的全称是什么？", "{full_name}", 1),
    ("{element}的元素符号是{symbol}。", "{element}的元素符号是什么？", "{symbol}", 1),
    ("光速约为{speed}米每秒。", "光速约为多少？", "{speed}", 1),
    ("{animal}的寿命约为{lifespan}年。", "{animal}的寿命约为多少？", "{lifespan}年", 1),

    # ── 文学 ──
    ("《{book}》是{author}写的{classic}。", "《{book}》的作者是谁？", "{author}", 0),
    ("《{book}》是{author}写的中国古典小说。", "《{book}》的作者是谁？", "{author}", 0),
    ("《{book}》发表于{year}年。", "《{book}》发表于哪一年？", "{year}年", 0),

    # ── 历史 ──
    ("{construction}全长约{length}公里，始建于{dynasty}。", "{construction}始建于哪个朝代？", "{dynasty}", 1),
    ("{country}有{num}个民族，{major}人口最多。", "{country}有多少个民族？", "{num}个", 1),
    ("地球表面积约{area}平方公里。", "地球表面积约多少？", "{area}", 1),
    ("{process}是植物利用光能制造有机物的过程。", "{process}利用什么能量？", "光能", 1),
    ("{event}发生在{year}年。", "{event}发生在哪一年？", "{year}年", 1),
]

# 填充模板的语料库
fill_data = {
    "name": ["张三", "李四", "王五", "赵六", "钱七", "孙八", "周九", "吴十",
             "爱因斯坦", "牛顿", "居里夫人", "达尔文", "张衡", "祖冲之"],
    "year": ["1990", "1985", "1978", "1964", "1956", "1774", "1905", "1911", "1999", "1642",
             "1879", "1860", "1950", "2001", "2020", "1988", "2005", "2010", "1995", "1800"],
    "profession": ["计算机科学家", "物理学家", "化学家", "数学家", "生物学家",
                   "天文学家", "思想家", "教育家", "作家", "工程师"],
    "era": ["春秋时期", "战国时期", "东汉末年", "盛唐", "明朝", "清朝",
            "20世纪", "19世纪", "文艺复兴时期", "古代"],
    "discovery": ["万有引力定律", "相对论", "镭元素", "进化论", "地动仪",
                  "青霉素", "X射线", "电子", "放射性", "DNA双螺旋"],
    "city": ["北京", "上海", "广州", "深圳", "杭州", "成都", "武汉", "南京",
             "东京", "纽约", "伦敦", "巴黎", "柏林", "悉尼"],
    "country": ["中国", "日本", "美国", "英国", "法国", "德国", "澳大利亚",
                "俄罗斯", "加拿大", "韩国"],
    "region": ["华北地区", "华东地区", "华南地区", "华中地区", "西南地区",
               "东亚", "欧洲", "北美", "南美", "大洋洲"],
    "river": ["长江", "黄河", "珠江", "黑龙江", "亚马逊河", "长江"],
    "length": ["6300", "5464", "2320", "4440", "6400", "3000"],
    "population": ["2400万", "3000万", "2000万", "1300万", "1400万"],
    "mountain": ["珠穆朗玛峰", "泰山", "黄山", "富士山", "阿尔卑斯山", "乞力马扎罗山"],
    "height": ["8848", "1545", "1864", "3776", "4807", "5895"],
    "continent": ["亚洲", "欧洲", "北美洲", "南美洲", "非洲", "大洋洲", "南极洲"],
    "language": ["汉语", "日语", "英语", "法语", "德语", "俄语"],
    "company": ["华为", "阿里巴巴", "腾讯", "百度", "字节跳动",
                "苹果", "微软", "谷歌", "三星", "特斯拉"],
    "location": ["深圳", "杭州", "深圳", "北京", "北京", "库比蒂诺", "雷德蒙德", "山景城", "首尔", "奥斯汀"],
    "founder": ["任正非", "马云", "马化腾", "李彦宏", "张一鸣",
                "乔布斯", "比尔·盖茨", "拉里·佩奇", "李秉喆", "马斯克"],
    "field": ["人工智能", "量子物理", "基因工程", "航天技术", "纳米技术"],
    "science": ["计算机科学", "物理学", "生物学", "工程学", "材料科学"],
    "star": ["太阳", "比邻星", "天狼星", "织女星", "北斗七星"],
    "num": ["八", "56", "七", "九", "12", "24"],
    "obj": ["行星", "恒星", "卫星", "星系", "星座"],
    "biggest": ["木星", "太阳", "天狼星A", "织女星", "北斗二"],
    "num1": ["206", "207", "208", "205"],
    "bone": ["股骨", "胫骨", "肱骨", "脊椎骨"],
    "substance": ["水", "酒精", "水银", "汽油"],
    "temp1": ["100°C", "78°C", "357°C", "70°C"],
    "temp2": ["0°C", "-114°C", "-39°C", "-95°C"],
    "days": ["365", "365.25", "366"],
    "planet": ["月球", "火星", "金星"],
    "full_name": ["脱氧核糖核酸", "核糖核酸", "核糖体核糖核酸", "转移核糖核酸"],
    "element": ["氢", "氦", "氧", "碳", "铁", "金", "银", "铜"],
    "symbol": ["H", "He", "O", "C", "Fe", "Au", "Ag", "Cu"],
    "speed": ["299792458", "300000000", "3亿"],
    "animal": ["大象", "乌龟", "鲸鱼", "鹰", "狗", "猫"],
    "lifespan": ["60-70", "150", "80", "50-60", "12-15", "12-18"],
    "book": ["红楼梦", "西游记", "水浒传", "三国演义", "论语", "道德经", "诗经", "史记"],
    "author": ["曹雪芹", "吴承恩", "施耐庵", "罗贯中", "孔子", "老子", "无名氏", "司马迁"],
    "classic": ["中国古典名著", "哲学著作", "史书", "诗歌总集"],
    "construction": ["长城", "故宫", "大运河", "都江堰"],
    "dynasty": ["秦朝", "明朝", "隋朝", "战国"],
    "major": ["汉族", "壮族", "满族"],
    "area": ["5.1亿", "5.2亿", "5.0亿"],
    "process": ["光合作用", "呼吸作用", "蒸腾作用"],
    "event": ["辛亥革命", "新中国成立", "工业革命", "文艺复兴",
              "第一次世界大战", "第二次世界大战"],
}

# 随机噪声词 (注入干扰)
noise_words = ["请问", "那么", "实际上", "当然", "也就是说",
               "例如", "特别是", "尤其是", "众所周知", "一般说来"]

extra_prefixes = ["我想知道", "请问一下", "帮我查查", "告诉我",
                  "你知不知道", "你能告诉我", ""]

extra_suffixes = ["请回答", "谢谢", "帮我一下", "拜托了", ""]


def fill_template(template, q_template, a_template, answer_idx, fp):
    """填充模板并记录答案在context中的偏移"""
    # 随机选取语料
    fill = {}
    for key in set(template.split("{")).union(q_template.split("{")):
        for k in ["name", "year", "profession", "era", "discovery",
                   "city", "country", "region", "river", "length", "population",
                   "mountain", "height", "continent", "language",
                   "company", "location", "founder",
                   "field", "science", "star", "num", "obj", "biggest",
                   "num1", "bone", "substance", "temp1", "temp2", "days",
                   "planet", "full_name", "element", "symbol", "speed",
                   "animal", "lifespan", "book", "author", "classic",
                   "construction", "dynasty", "major", "area", "process", "event"]:
            if "{" + k + "}" in (template + q_template + a_template):
                fill[k] = random.choice(fp[k])

    context = template.format(**fill)
    question = q_template.format(**fill)
    answer = a_template.format(**fill)

    # 数据增强1: 随机前缀
    prefix = random.choice(extra_prefixes)
    if prefix:
        context = prefix + "，" + context

    # 数据增强2: 随机后缀
    suffix = random.choice(extra_suffixes)
    if suffix:
        context = context + suffix

    # 数据增强3: 注入噪声词
    if random.random() < 0.2:
        noise = random.choice(noise_words)
        words = list(context)
        pos = random.randint(0, max(0, len(words) - 2))
        words.insert(pos, noise)
        context = "".join(words)

    # 数据增强4: 随机删除字符 (1%概率)
    if random.random() < 0.1:
        chars = list(context)
        if len(chars) > 10:
            del_pos = random.randint(0, len(chars) - 1)
            del chars[del_pos]
            context = "".join(chars)

    # 找到答案位置
    answer_pos = context.find(answer)
    if answer_pos == -1:
        return None

    # 数据增强5: 问题模糊化 — 随机替换问题中的1个字
    if random.random() < 0.1:
        q_chars = list(question)
        if len(q_chars) > 4:
            rp = random.randint(0, len(q_chars) - 1)
            q_chars[rp] = "某"
            question = "".join(q_chars)

    return (context, question, answer, answer_pos, answer_pos + len(answer))


def generate_synthetic_data(num_samples):
    """生成多样化合成问答数据"""
    data = []
    while len(data) < num_samples:
        t_idx = random.randrange(len(qa_templates))
        template, q_template, a_template, answer_idx = qa_templates[t_idx]
        result = fill_template(template, q_template, a_template, answer_idx, fill_data)
        if result is not None:
            data.append(result)
    return data


def load_textbook_dataset(filepath=None):
    """加载教材数据集 (取代模板数据)"""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "textbook_dataset.json")
    if not os.path.exists(filepath):
        print(f"    ! 未找到教材数据集 ({filepath}), 回退到模板数据")
        return None

    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 转换为 (context, question, answer, s_char, e_char) 格式
    data = []
    for item in raw:
        ctx = item["paragraph"]
        q = item["question"]
        a = item["answer"]
        s = item["answer_start"]
        e = item["answer_end"]
        # 双重校验: 保证答案确实在段落中
        pos = ctx.find(a)
        if pos >= 0:
            data.append((ctx, q, a, s, e))
        elif s < len(ctx) and e < len(ctx):
            # fallback: 用保存的位置
            data.append((ctx, q, a, s, e))

    return data


# ═══════════════════════════════════════════════════
#  2. 数据集
# ═══════════════════════════════════════════════════

def build_vocab(all_texts):
    chars = set()
    for text in all_texts:
        chars.update(text)
    chars = sorted(c for c in chars if ord(c) < 65536)
    # 常用汉字优先
    common = "的一是不了人在有我他这中大小上个来生会也子就出下去好看自家走过得多你没有"
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for c in common:
        if c in chars and c not in word2idx:
            word2idx[c] = len(word2idx)
    for c in chars:
        if c not in word2idx and len(word2idx) < vocab_size - 2:
            word2idx[c] = len(word2idx)
    while len(word2idx) < vocab_size:
        word2idx[f"<EXTRA_{len(word2idx)}>"] = len(word2idx)
    return word2idx, {v: k for k, v in word2idx.items()}


def text_to_seq(text, w2i, max_len):
    seq = [w2i.get(c, w2i["<UNK>"]) for c in text[:max_len]]
    seq = seq + [0] * (max_len - len(seq))
    return seq


def find_answer_span(ctx_text, answer_text, max_len):
    if len(ctx_text) > max_len:
        ctx_text = ctx_text[:max_len]
    pos = ctx_text.find(answer_text)
    if pos < 0:
        pos = 0
    end = min(pos + max(1, len(answer_text)), len(ctx_text)) - 1
    return max(0, pos), max(pos, end)


class QADataset(Dataset):
    def __init__(self, data, w2i):
        self.data = data
        self.w2i = w2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ctx, q, a, s_char, e_char = self.data[idx]
        c_seq = text_to_seq(ctx, self.w2i, max_context_len)
        q_seq = text_to_seq(q, self.w2i, max_query_len)
        s, e = find_answer_span(ctx, a, max_context_len)
        return (
            torch.tensor(c_seq, dtype=torch.long),
            torch.tensor(q_seq, dtype=torch.long),
            torch.tensor(s, dtype=torch.long),
            torch.tensor(e, dtype=torch.long),
        )


# ═══════════════════════════════════════════════════
#  3. 训练主循环
# ═══════════════════════════════════════════════════

def save_status(state):
    """保存训练状态 (便于中断恢复)"""
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except:
        pass


def train():
    print("=" * 56)
    print("  ChatModel (175M) — 防过拟合训练")
    print("=" * 56)

    print(f"\n设备: {device}")

    # ── 生成数据 ──
    print(f"\n[1] 加载数据 ({num_train:,} 条, 优先使用教材数据集) ...")
    all_data = load_textbook_dataset()
    if all_data is None:
        print(f"    → 使用模板合成数据")
        all_data = generate_synthetic_data(num_train)
        # 只保留和num_train一样多
        selected = all_data[:num_train]
    else:
        # 教材数据可能不足num_train条, 用模板补齐
        print(f"    → 教材数据集: {len(all_data)} 条")
        if len(all_data) < num_train:
            extra = generate_synthetic_data(num_train - len(all_data))
            all_data = all_data + extra
            print(f"    → 模板补齐: +{len(extra)} 条")
        selected = all_data[:num_train]
    print(f"    ✓ {len(selected)} 条问答数据准备完毕")

    all_texts = [t for c, q, a, _, _ in selected for t in (c, q, a)]
    w2i, i2w = build_vocab(all_texts)
    print(f"    ✓ 词汇表大小: {len(w2i)}")

    # 划分训练/验证 (85%/15%)
    split = int(0.85 * len(selected))
    train_ds = QADataset(selected[:split], w2i)
    val_ds = QADataset(selected[split:], w2i)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"    ✓ 训练: {len(train_ds)} | 验证: {len(val_ds)}")

    # ── 模型 ──
    print(f"\n[2] 构建模型 (175M, dropout={dropout}, wd={weight_decay}, ls={label_smoothing}) ...")
    model = ChatModel().to(device)
    total_p = count_parameters(model)
    print(f"    ✓ 参数量: {total_p:,} ({total_p/1e6:.1f}M)")

    # ── 损失函数 (Label Smoothing) ──
    criterion = nn.CrossEntropyLoss(
        ignore_index=0,
        label_smoothing=label_smoothing,
    )

    # ── 优化器 (强Weight Decay) ──
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )

    # ── 余弦退火 + LR预热 ──
    total_steps = (len(train_loader) // 4) * epochs  # accum后的step数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # 预热用gradual warmup: 自己算warmup factor
    warmup_iters = min(warmup_steps, total_steps // 10)
    def get_lr_multiplier(step):
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        return 1.0

    # 梯度累积步数
    accum_steps = 4
    steps_per_epoch = (len(train_loader) + accum_steps - 1) // accum_steps

    print(f"\n[3] 开始训练 (epochs={epochs}, accum={accum_steps}, warmup={warmup_iters}steps)")
    print(f"    {'='*50}")

    best_val_loss = float('inf')
    best_epoch = -1
    no_improve = 0
    train_losses = []
    val_losses = []
    global_step = 0

    for epoch in range(epochs):
        # ── 训练 ──
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        t0 = time.time()

        for step, (c, q, s, e) in enumerate(train_loader):
            c = c.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)

            start_logits, end_logits = model(c, q)
            loss = criterion(start_logits, s) + criterion(end_logits, e)
            loss = loss / accum_steps
            loss.backward()

            if (step + 1) % accum_steps == 0:
                # LR预热 (按global step)
                if global_step < warmup_iters:
                    multiplier = global_step / max(1, warmup_iters)
                    for g in optimizer.param_groups:
                        g['lr'] = lr * multiplier

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            total_loss += loss.item() * accum_steps

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # ── 验证 ──
        model.eval()
        val_loss = 0
        correct_s = 0
        correct_e = 0
        total = 0

        with torch.no_grad():
            for c, q, s, e in val_loader:
                c = c.to(device, non_blocking=True)
                q = q.to(device, non_blocking=True)
                s = s.to(device, non_blocking=True)
                e = e.to(device, non_blocking=True)

                start_logits, end_logits = model(c, q)
                loss = criterion(start_logits, s) + criterion(end_logits, e)
                val_loss += loss.item()

                pred_s = start_logits.argmax(1).cpu()
                pred_e = end_logits.argmax(1).cpu()
                correct_s += (pred_s == s.cpu()).sum().item()
                correct_e += (pred_e == e.cpu()).sum().item()
                total += s.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        dt = time.time() - t0
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  [{epoch+1:2d}/{epochs}] "
              f"train={avg_train_loss:.4f}  "
              f"val={avg_val_loss:.4f}  "
              f"acc_s={100*correct_s/total:.1f}%  "
              f"acc_e={100*correct_e/total:.1f}%  "
              f"lr={current_lr:.2e}  {dt:.0f}s")

        # ── 早停判断 ──
        if avg_val_loss < best_val_loss - 0.001:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            no_improve = 0
            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, "chat_model_best.pth")
            print(f"           ✓ 保存最佳模型 (val_loss={avg_val_loss:.4f})")
        else:
            no_improve += 1

        # 保存训练状态
        save_status({
            'epoch': epoch,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss,
            'no_improve': no_improve,
            'train_losses': train_losses,
            'val_losses': val_losses,
        })

        # 早停
        if no_improve >= early_stop_patience:
            print(f"\n  ⚡ 早停! 验证集 {early_stop_patience} 轮未下降, 停止训练")
            print(f"    最佳 epoch: {best_epoch+1}  val_loss={best_val_loss:.4f}")
            break

        # 每5轮检查过拟合信号
        if len(train_losses) >= 3:
            recent_train = np.mean(train_losses[-3:])
            recent_val = np.mean(val_losses[-3:])
            if recent_val > recent_train * 1.5 and epoch > 5:
                print(f"  ⚠️ 过拟合信号: val/train = {recent_val/recent_train:.2f}x")

    # ── 最终 ──
    print(f"\n{'='*56}")

    # 加载最佳模型
    if os.path.exists("chat_model_best.pth"):
        checkpoint = torch.load("chat_model_best.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  加载最佳模型 (epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f})")

    # 最终验证
    model.eval()
    correct_s = 0
    correct_e = 0
    total = 0
    with torch.no_grad():
        for c, q, s, e in val_loader:
            c = c.to(device, non_blocking=True)
            q = q.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            e = e.to(device, non_blocking=True)
            start_logits, end_logits = model(c, q)
            pred_s = start_logits.argmax(1).cpu()
            pred_e = end_logits.argmax(1).cpu()
            correct_s += (pred_s == s.cpu()).sum().item()
            correct_e += (pred_e == e.cpu()).sum().item()
            total += s.size(0)

    print(f"\n  最终验证:")
    print(f"    开始位置准确率: {100 * correct_s / total:.1f}%")
    print(f"    结束位置准确率: {100 * correct_e / total:.1f}%")

    # 保存
    torch.save(model.state_dict(), "chat_model.pth")
    with open("vocab.json", "w", encoding="utf-8") as f:
        json.dump(w2i, f, ensure_ascii=False)

    print(f"\n  ✓ 模型: chat_model.pth")
    print(f"  ✓ 词典: vocab.json")
    print(f"  ✓ 最佳: chat_model_best.pth (epoch {best_epoch+1}, val_loss={best_val_loss:.4f})")
    print(f"\n  ✅ 完成! ({total_p:,} 参数)")


if __name__ == "__main__":
    train()
