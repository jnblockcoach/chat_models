# Fast textbook dataset builder - 50K items
import json, random, os
random.seed(42)

results = []
seen_qa = set()

def add(para, q, a):
    pos = para.find(a)
    if pos < 0:
        return
    key = (q[:50], a[:30])
    if key in seen_qa:
        return
    seen_qa.add(key)
    results.append({
        "paragraph": para,
        "question": q,
        "answer": a,
        "answer_start": pos,
        "answer_end": pos + len(a) - 1,
    })

# === LARGE KNOWLEDGE BASE ===

# 1. Chinese Poetry (15 poems, 4-6 QAs each = 75-90 base)
poems = [
    ("静夜思","李白","唐代","床前明月光，疑是地上霜。举头望明月，低头思故乡。"),
    ("春晓","孟浩然","唐代","春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。"),
    ("咏鹅","骆宾王","唐代","鹅鹅鹅，曲项向天歌。白毛浮绿水，红掌拨清波。"),
    ("望庐山瀑布","李白","唐代","日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。"),
    ("登鹳雀楼","王之涣","唐代","白日依山尽，黄河入海流。欲穷千里目，更上一层楼。"),
    ("绝句","杜甫","唐代","两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。"),
    ("悯农","李绅","唐代","锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。"),
    ("草","白居易","唐代","离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。"),
    ("画","王维","唐代","远看山有色，近听水无声。春去花还在，人来鸟不惊。"),
    ("山行","杜牧","唐代","远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。"),
    ("江雪","柳宗元","唐代","千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。"),
    ("游子吟","孟郊","唐代","慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。"),
    ("出塞","王昌龄","唐代","秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。"),
    ("凉州词","王翰","唐代","葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。"),
    ("敕勒歌","北朝民歌","南北朝","敕勒川，阴山下。天似穹庐，笼盖四野。天苍苍，野茫茫，风吹草低见牛羊。"),
    ("江南","汉乐府","汉代","江南可采莲，莲叶何田田。鱼戏莲叶间。鱼戏莲叶东，鱼戏莲叶西，鱼戏莲叶南，鱼戏莲叶北。"),
]
for title, author, dynasty, content in poems:
    for num in range(5):
        extra = ["意境深远","语言优美","流传千古","脍炙人口","家喻户晓"][num]
        p = title + "是" + dynasty + "诗人" + author + "的诗作。内容是：" + content + "。" + "这首诗" + extra + "。"
        add(p, title + "的作者是谁？", author)
        add(p, title + "是什么朝代的？", dynasty)
        add(p, author + "是哪个朝代的什么诗人？", dynasty + "诗人" + author[:1])
        # lines from content
        lines = [l.strip() for l in content.replace("，","，").split("。") if l.strip() and len(l.strip())>=3]
        for line in lines[:3]:
            add(p, line[:12] + "出自哪首诗？", line[:20])

# 2. Classical texts
classics = [
    ("论语","孔子弟子","春秋","儒家经典","学而时习之，不亦说乎"),
    ("孟子","孟子","战国","儒家经典","生于忧患，死于安乐"),
    ("三字经","王应麟","南宋","启蒙教材","人之初，性本善。性相近，习相远"),
    ("岳阳楼记","范仲淹","北宋","散文","先天下之忧而忧，后天下之乐而乐"),
    ("醉翁亭记","欧阳修","北宋","散文","醉翁之意不在酒，在乎山水之间也"),
]
for title, author, dynasty, cat, quote in classics:
    for n in range(3):
        extra = ["重要典籍","著名篇章","经典名作"][n]
        p = title + "是" + dynasty + "的" + cat + "。" + title + "中的名句是" + quote + "。" + "这是中国" + extra + "。"
        add(p, title + "的作者是谁？", author)
        add(p, title + "是什么朝代的？", dynasty)
        add(p, title + "属于什么类别？", cat)
        add(p, quote[:12] + "出自哪部作品？", title)

# 3. Math
math_items = [
    ("勾股定理","直角三角形中两条直角边的平方和等于斜边的平方","a2+b2=c2","周髀算经"),
    ("三角形内角和","三角形的三个内角之和等于180度","180度","几何学"),
    ("长方形面积","长方形的面积等于长乘以宽","S=ab","小学几何"),
    ("正方形面积","正方形的面积等于边长乘以边长","S=a2","小学几何"),
    ("圆面积","圆的面积等于圆周率乘以半径的平方","S=pi*r2","几何学"),
    ("圆周长","圆的周长等于圆周率乘以直径","C=2pi*r","几何学"),
    ("乘法分配律","两个数的和乘以一个数等于分别相乘再相加","(a+b)c=ac+bc","代数"),
    ("加法交换律","交换加数的位置和不变","a+b=b+a","算术"),
    ("乘法交换律","交换因数的位置积不变","axb=bxa","算术"),
    ("分数基本性质","分子分母同乘同除一个非零数分数大小不变","a/b=ka/kb","算术"),
    ("平行四边形面积","底乘以高","S=ah","几何"),
    ("梯形面积","上底加下底乘以高除以2","S=(a+b)h/2","几何"),
    ("圆柱体积","底面积乘以高","V=pir2h","立体几何"),
    ("圆锥体积","底面积乘以高除以3","V=pir2h/3","立体几何"),
    ("速度公式","路程除以时间","v=s/t","运动学"),
    ("密度公式","质量除以体积","p=m/V","密度"),
    ("概率","事件发生的可能性大小在0到1之间","0到1","统计"),
]
for name, content, formula, field in math_items:
    for n in range(3):
        extra = ["重要公式","基本概念","核心知识"][n]
        p = name + "是" + field + "中的" + extra + "。" + name + "是指" + content + "。" + "公式为" + formula + "。" + "在数学学习中经常用到。"
        add(p, name + "的公式是什么？", formula)
        add(p, name + "是指什么？", content[:10])
        add(p, name + "属于哪个领域？", field)

# 4. Physics
physics = [
    ("牛顿第一定律","惯性定律","一切物体在没有受到力的作用时总保持静止或匀速直线运动状态","力学"),
    ("牛顿第二定律","加速度定律","F=ma，加速度与力成正比与质量成反比","力学"),
    ("牛顿第三定律","作用力反作用力定律","两个物体之间的作用力和反作用力大小相等方向相反","力学"),
    ("欧姆定律","电路定律","I=U/R，电流与电压成正比与电阻成反比","电学"),
    ("阿基米德原理","浮力定律","浸在液体中的物体所受浮力等于排开液体的重力","流体力学"),
    ("光的反射定律","反射定律","反射光线入射光线和法线在同一平面内反射角等于入射角","光学"),
    ("光的折射定律","折射定律","光从一种介质斜射入另一种介质时传播方向发生偏折","光学"),
    ("能量守恒定律","能量定律","能量既不会凭空产生也不会凭空消灭只能从一种形式转化为另一种形式","热力学"),
    ("杠杆原理","杠杆定律","动力乘以动力臂等于阻力乘以阻力臂","力学"),
    ("帕斯卡原理","液压原理","加在密闭液体上的压强能够大小不变地向各个方向传递","流体力学"),
]
for name, alias, content, field in physics:
    for n in range(3):
        extra = ["基本定律","重要定律","核心定律"][n]
        p = name + "是物理学" + field + "中的" + extra + "。" + name + "又称" + alias + "。" + "其内容是：" + content + "。" + "这个定律是物理学的基石之一。"
        add(p, name + "又称什么？", alias)
        add(p, name + "的内容是什么？", content[:12])
        add(p, name + "属于哪个分支？", field)

# 5. Chemistry
chemistry = [
    ("元素周期表","门捷列夫","1869年","按原子序数排列元素"),
    ("水的电解","通电生成氢气和氧气","正极产生氧气负极产生氢气","体积比1比2"),
    ("质量守恒定律","拉瓦锡","化学反应前后总质量不变","化学基本定律"),
    ("原子","化学变化中的最小粒子","由原子核和核外电子构成","原子结构"),
    ("分子","保持物质化学性质的最小粒子","由原子构成","分子结构"),
    ("酸","电离时阳离子全是氢离子的化合物","pH小于7","酸碱盐"),
    ("碱","电离时阴离子全是氢氧根离子的化合物","pH大于7","酸碱盐"),
    ("中和反应","酸和碱反应生成盐和水的反应","放出热量","化学反应"),
]
for name, details, extra1, extra2 in chemistry:
    p = name + "是化学中的重要概念。" + details + "。" + extra1 + "。" + extra2 + "。"
    add(p, name + "属于什么领域？", "化学")
    add(p, name + "的特点是？", extra1[:10])

# 6. Biology
biology = [
    ("细胞","生物体结构和功能的基本单位","有细胞膜细胞质细胞核","生物学"),
    ("DNA","脱氧核糖核酸","遗传信息的载体","沃森和克里克发现双螺旋结构"),
    ("光合作用","绿色植物利用光能将二氧化碳和水转化为有机物并释放氧气","在叶绿体中进行","需要光"),
    ("呼吸作用","生物体吸收氧气放出二氧化碳的过程","为生命活动提供能量","线粒体中进行"),
    ("食物链","生态系统中生物之间吃与被吃的关系","草被兔子吃兔子被狐狸吃","能量流动途径"),
    ("进化论","达尔文提出","自然选择适者生存","物种起源"),
    ("生态系统","生物群落与无机环境相互作用的统一整体","具有自我调节能力","生态平衡"),
]
for name, content, extra1, extra2 in biology:
    p = name + "是" + extra2 + "中的重要概念。" + name + "是指" + content + "。" + extra1 + "。"
    add(p, name + "是指什么？", content[:15])
    add(p, name + "属于什么领域？", extra2[:10])

# 7. History
history = [
    ("秦统一六国","前221年","秦始皇嬴政","建立了郡县制统一度量衡文字货币"),
    ("汉武帝大一统","前140年","汉武帝刘彻","罢黜百家独尊儒术开辟丝绸之路"),
    ("贞观之治","627年至649年","唐太宗李世民","政治清明经济发展"),
    ("安史之乱","755年至763年","安禄山史思明","唐朝由盛转衰"),
    ("鸦片战争","1840年至1842年","英国","签订南京条约割让香港岛"),
    ("太平天国运动","1851年至1864年","洪秀全","天朝田亩制度"),
    ("洋务运动","1861年至1895年","李鸿章曾国藩","自强求富"),
    ("戊戌变法","1898年","康有为梁启超","百日维新"),
    ("辛亥革命","1911年","孙中山","推翻清朝建立民国"),
    ("五四运动","1919年5月4日","学生","新民主主义革命开端"),
    ("中国共产党成立","1921年7月","上海","中共一大"),
    ("抗日战争","1937年至1945年","全民族抗战","日本投降"),
    ("新中国成立","1949年10月1日","毛泽东","中国人民站起来了"),
    ("改革开放","1978年","邓小平","十一届三中全会经济特区"),
]
for event, time, leader, significance in history:
    for n in range(3):
        p = event + "发生在" + time + "。" + event + "由" + leader + "领导。" + event + "的意义是" + significance + "。" + "这是" + ["中国历史","世界历史","中国近代史"][n%3] + "上的重要事件。"
        add(p, event + "发生在什么时候？", time)
        add(p, event + "由谁领导？", leader)
        add(p, event + "的历史意义是什么？", significance[:12])

# 8. Geography
geography = [
    ("长江","6300公里","唐古拉山","东海","中国最长河流"),
    ("黄河","5464公里","巴颜喀拉山","渤海","中华民族母亲河"),
    ("珠江","2320公里","云南","南海","华南第一大河"),
    ("泰山","1545米","山东","五岳之首"),
    ("华山","2154米","陕西","以险著称"),
    ("珠穆朗玛峰","8848米","喜马拉雅山","世界最高峰"),
    ("亚洲","4400万平方公里","最大洲","48个国家"),
    ("非洲","3000万平方公里","第二大洲","热带草原广布"),
    ("太平洋","18100万平方公里","最大最深","平均深度约4000米"),
]

for item in geography:
    name = item[0]
    prop = item[1]
    origin = item[2] if len(item)>2 else ""
    extra = item[3] if len(item)>3 else ""
    extra2 = item[4] if len(item)>4 else ""
    p = name + prop + "。" + ("发源于" + origin + "。" if origin else "") + extra + "。" + extra2 + "。"
    add(p, name + "的特点是？", prop[:12])
    if origin:
        add(p, name + "发源于哪里？", origin)

# 9. English grammar
grammar = [
    ("一般现在时","表示经常发生的动作或存在的状态","主语加动词原形或动词三单"),
    ("现在进行时","表示正在进行的动作","am/is/are加动词ing形式"),
    ("一般过去时","表示过去发生的动作或状态","主语加动词过去式"),
    ("一般将来时","表示将要发生的动作","will加动词原形"),
    ("现在完成时","表示已经完成的动作","have/has加过去分词"),
    ("被动语态","主语是动作的承受者","be加过去分词"),
    ("比较级","比较两者差异","形容词比较级加than"),
    ("最高级","比较三者或以上","the加形容词最高级"),
    ("条件状语从句","表示条件","if加从句加主句"),
    ("定语从句","修饰名词或代词","关系代词引导"),
]
for name, usage, structure in grammar:
    p = name + "是英语语法的重要内容。" + name + "用于" + usage + "。" + "其结构是" + structure + "。" + "掌握这个语法点对英语学习很有帮助。"
    add(p, name + "用于什么情况？", usage)
    add(p, name + "的结构是什么？", structure[:15])

# === CROSS-MATCH ===
# Take all paragraphs and find all possible answer matches
paras = []
p2qs = {}
for r in results:
    if r["paragraph"] not in paras:
        paras.append(r["paragraph"])
    q = r["question"]
    a = r["answer"]
    if r["paragraph"] not in p2qs:
        p2qs[r["paragraph"]] = []
    p2qs[r["paragraph"]].append((q, a))

all_answers_set = set()
for r in results:
    all_answers_set.add(r["answer"])
all_answers = list(all_answers_set)
print(f"已有 {len(results)} 条 | 段落 {len(paras)} | 答案 {len(all_answers)}")

# Cross-match every paragraph with every answer
matched = 0
for p in paras:
    for a in all_answers:
        if len(a) < 3:
            continue
        if a in p:
            # Add with creative questions
            for q_var in [
                "什么是" + a,
                a + "是什么",
                "请解释" + a,
                a[:8] + "指什么",
                a[:6] + "是哪方面知识",
                a[:5] + "的概念",
            ]:
                add(p, q_var, a)
                matched += 1
                if matched % 1000 == 0:
                    break

print(f"交叉匹配后: {len(results)} 条")

# Final dedup and sort
random.shuffle(results)
# Keep max 50K
final = results[:50000]

with open(r"D:\models\qa_model\chat_models\textbook_dataset.json", "w", encoding="utf-8") as f:
    json.dump(final, f, ensure_ascii=False, indent=1)

sz = os.path.getsize(r"D:\models\qa_model\chat_models\textbook_dataset.json")
print(f"写入 {len(final)} 条 | {sz/1024/1024:.1f} MB")
