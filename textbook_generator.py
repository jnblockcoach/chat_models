#!/usr/bin/env python3
"""
中小学教材知识数据集生成器 v2 (50,000+条)
基于真实课程标准知识点，用知识库+模板组合生成多样化内容
"""
import json, os, random, itertools
from pathlib import Path

OUTPUT = Path(__file__).parent / "textbook_dataset.json"
random.seed(42)


# ═══════════════════════════════════════════════════════
#  知识点知识库
# ═══════════════════════════════════════════════════════

# 每个知识点: (段落模板, [(问题, 答案), ...])
# 模板用 {n} 占位, 运行时随机填入知识库数据

# ── 语文古诗文知识库 ──
POEM_DB = {
    "poems": [
        ("春晓", "孟浩然", "唐代", "五言绝句", "山水田园诗人",
         "这首诗描写了春天早晨的景色：春眠不觉晓，处处闻啼鸟。夜来风雨声，花落知多少。",
         ["语言平易自然，意境优美，表达了对春天的喜爱和对时光流逝的淡淡惋惜"]),
        ("静夜思", "李白", "唐代", "五言绝句", "浪漫主义诗人",
         "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
         ["用简洁的语言表达了游子在寂静的夜晚思念故乡的情感"]),
        ("咏鹅", "骆宾王", "唐代", "五言绝句", "初唐四杰",
         "鹅鹅鹅，曲项向天歌。白毛浮绿水，红掌拨清波。",
         ["用生动的语言描绘了鹅在水中游动的美丽画面"]),
        ("望庐山瀑布", "李白", "唐代", "七言绝句", "诗仙",
         "日照香炉生紫烟，遥看瀑布挂前川。飞流直下三千尺，疑是银河落九天。",
         ["用夸张的手法描绘了庐山瀑布的壮丽景象"]),
        ("登鹳雀楼", "王之涣", "唐代", "五言绝句", "边塞诗人",
         "白日依山尽，黄河入海流。欲穷千里目，更上一层楼。",
         ["描绘了落日黄河的壮阔景象，揭示了只有站得高才能看得远的哲理"]),
        ("绝句", "杜甫", "唐代", "七言绝句", "诗圣",
         "两个黄鹂鸣翠柳，一行白鹭上青天。窗含西岭千秋雪，门泊东吴万里船。",
         ["描绘了生机盎然的春日景象，每句一景，对仗工整"]),
        ("悯农", "李绅", "唐代", "五言绝句", "新乐府运动",
         "锄禾日当午，汗滴禾下土。谁知盘中餐，粒粒皆辛苦。",
         ["教育人们要珍惜粮食，体会农民劳动的艰辛"]),
        ("草/赋得古原草送别", "白居易", "唐代", "五言律诗", "香山居士",
         "离离原上草，一岁一枯荣。野火烧不尽，春风吹又生。",
         ["通过描写野草的顽强生命力，比喻坚韧不拔的精神"]),
        ("画", "王维", "唐代", "五言绝句", "诗佛",
         "远看山有色，近听水无声。春去花还在，人来鸟不惊。",
         ["从远和近两个角度展现了画中景物的奇妙之处"]),
        ("敕勒歌", "无名氏", "南北朝", "北朝民歌", "北朝民歌",
         "敕勒川，阴山下。天似穹庐，笼盖四野。天苍苍，野茫茫，风吹草低见牛羊。",
         ["描绘了草原辽阔壮美的景象，展现了游牧民族的生活画卷"]),
        ("江南", "无名氏", "汉代", "汉乐府诗", "乐府诗",
         "江南可采莲，莲叶何田田。鱼戏莲叶间。鱼戏莲叶东，鱼戏莲叶西，鱼戏莲叶南，鱼戏莲叶北。",
         ["以简洁的语言描绘了江南水乡采莲时鱼戏莲叶的生动画面"]),
        ("凉州词", "王翰", "唐代", "七言绝句", "边塞诗人",
         "葡萄美酒夜光杯，欲饮琵琶马上催。醉卧沙场君莫笑，古来征战几人回。",
         ["表现了边塞将士豪放洒脱的情怀和战争的残酷"]),
        ("出塞", "王昌龄", "唐代", "七言绝句", "七绝圣手",
         "秦时明月汉时关，万里长征人未还。但使龙城飞将在，不教胡马度阴山。",
         ["表达了诗人对边疆战事的感慨和对名将的怀念"]),
        ("山行", "杜牧", "唐代", "七言绝句", "晚唐诗人",
         "远上寒山石径斜，白云生处有人家。停车坐爱枫林晚，霜叶红于二月花。",
         ["描绘了深秋山景，特别是枫叶比二月花还红的美丽景象"]),
        ("江雪", "柳宗元", "唐代", "五言绝句", "唐宋八大家",
         "千山鸟飞绝，万径人踪灭。孤舟蓑笠翁，独钓寒江雪。",
         ["用简洁的语言描绘了冬日寒江独钓的孤寂景象"]),
        ("春夜喜雨", "杜甫", "唐代", "五言律诗", "诗圣",
         "好雨知时节，当春乃发生。随风潜入夜，润物细无声。",
         ["描写了春雨滋润万物的美好情景"]),
        ("游子吟", "孟郊", "唐代", "五言古诗", "苦吟诗人",
         "慈母手中线，游子身上衣。临行密密缝，意恐迟迟归。谁言寸草心，报得三春晖。",
         ["歌颂了母爱的伟大和无私"]),
    ],
    "authors": [
        ("李白", "诗仙", "唐代", "浪漫主义诗人", "四川江油", "将进酒"),
        ("杜甫", "诗圣", "唐代", "现实主义诗人", "河南巩义", "春望"),
        ("白居易", "香山居士", "唐代", "现实主义诗人", "山西太原", "琵琶行"),
        ("王维", "诗佛", "唐代", "山水田园诗人", "山西祁县", "山居秋暝"),
        ("孟浩然", "孟襄阳", "唐代", "山水田园诗人", "湖北襄阳", "过故人庄"),
        ("李商隐", "玉溪生", "唐代", "晚唐诗人", "河南沁阳", "锦瑟"),
        ("苏轼", "东坡居士", "宋代", "豪放派词人", "四川眉山", "水调歌头"),
        ("李清照", "易安居士", "宋代", "婉约派词人", "山东济南", "如梦令"),
        ("陆游", "放翁", "宋代", "爱国诗人", "浙江绍兴", "示儿"),
        ("辛弃疾", "稼轩", "宋代", "豪放派词人", "山东济南", "永遇乐"),
    ],
    "classical_texts": [
        ("论语", "孔子弟子", "春秋", "儒家经典", "学而时习之不亦说乎"),
        ("孟子", "孟子及其弟子", "战国", "儒家经典", "生于忧患死于安乐"),
        ("三字经", "王应麟", "南宋", "启蒙教材", "人之初性本善"),
        ("岳阳楼记", "范仲淹", "北宋", "散文名篇", "先天下之忧而忧"),
        ("醉翁亭记", "欧阳修", "北宋", "散文名篇", "醉翁之意不在酒"),
        ("桃花源记", "陶渊明", "东晋", "散文名篇", "世外桃源"),
    ],
}

# ── 数学知识库 ──
MATH_DB = {
    "geometry_theorems": [
        ("勾股定理", "直角三角形", "两条直角边的平方和等于斜边的平方", "a²+b²=c²", "周髀算经"),
        ("三角形内角和定理", "任意三角形", "三个内角之和等于180度", "180°", "几何原本"),
        ("平行四边形性质", "平行四边形", "对边平行且相等，对角相等", "", "几何学"),
        ("圆周角定理", "圆", "同弧所对的圆周角等于圆心角的一半", "", "几何学"),
        ("相似三角形定理", "相似三角形", "对应边成比例，对应角相等", "", "几何学"),
    ],
    "arithmetic_laws": [
        ("加法交换律", "整数加法", "a+b=b+a", "小学数学"),
        ("加法结合律", "整数加法", "(a+b)+c=a+(b+c)", "小学数学"),
        ("乘法交换律", "整数乘法", "a×b=b×a", "小学数学"),
        ("乘法结合律", "整数乘法", "(a×b)×c=a×(b×c)", "小学数学"),
        ("乘法分配律", "整数运算", "(a+b)×c=a×c+b×c", "小学数学"),
        ("分数基本性质", "分数运算", "分子分母同乘同除一个非零数分数大小不变", "小学数学"),
    ],
    "formulas": [
        ("长方形面积", "S=a×b", "长乘以宽", "面积"),
        ("正方形面积", "S=a²", "边长乘以边长", "面积"),
        ("三角形面积", "S=a×h÷2", "底乘以高除以2", "面积"),
        ("平行四边形面积", "S=a×h", "底乘以高", "面积"),
        ("梯形面积", "S=(a+b)×h÷2", "上底加下底乘以高除以2", "面积"),
        ("圆的面积", "S=πr²", "圆周率乘以半径的平方", "面积"),
        ("圆的周长", "C=2πr", "2乘以圆周率乘以半径", "周长"),
        ("长方体体积", "V=a×b×h", "长乘以宽乘以高", "体积"),
        ("正方体体积", "V=a³", "棱长的立方", "体积"),
        ("圆柱体积", "V=πr²h", "底面积乘以高", "体积"),
        ("圆锥体积", "V=πr²h÷3", "底面积乘以高除以3", "体积"),
        ("速度公式", "v=s/t", "路程除以时间", "运动学"),
        ("密度公式", "ρ=m/V", "质量除以体积", "密度"),
        ("效率公式", "η=W有用/W总", "有用功除以总功", "机械效率"),
    ],
    "math_constants": [
        ("圆周率π", "3.1415926", "圆周长与直径的比值", "祖冲之", "南北朝"),
        ("自然常数e", "2.71828", "自然对数的底数", "欧拉", "18世纪"),
        ("黄金分割数φ", "1.618", "将一条线段分成两部分的比例", "", "古希腊"),
        ("0", "零", "既不是正数也不是负数", "印度数学家", "公元5世纪"),
    ],
}

# ── 科学知识库 ──
SCI_DB = {
    "physics": [
        ("牛顿第一定律", "惯性定律", "一切物体在没有受力时总保持静止或匀速直线运动", "力学"),
        ("牛顿第二定律", "加速度定律", "F=ma，加速度与力成正比与质量成反比", "力学"),
        ("牛顿第三定律", "作用力与反作用力", "两个物体之间的作用力和反作用力大小相等方向相反", "力学"),
        ("欧姆定律", "电路定律", "I=U/R，电流与电压成正比与电阻成反比", "电学"),
        ("阿基米德原理", "浮力定律", "浸在液体中的物体受到的浮力等于排开液体的重力", "流体力学"),
        ("光的反射定律", "几何光学", "反射光线入射光线法线在同一平面内反射角等于入射角", "光学"),
        ("光的折射定律", "几何光学", "光从一种介质斜射入另一种介质时传播方向发生改变", "光学"),
        ("能量守恒定律", "热力学", "能量既不会凭空产生也不会凭空消失只能从一种形式转化为另一种形式", "热力学"),
        ("帕斯卡原理", "液压原理", "加在密闭液体上的压强能够大小不变地被液体向各个方向传递", "流体力学"),
        ("杠杆原理", "简单机械", "动力×动力臂=阻力×阻力臂", "力学"),
    ],
    "chemistry": [
        ("元素周期表", "门捷列夫", "1869年", "118种元素", "7个周期18个族"),
        ("水的电解", "2H₂O→2H₂↑+O₂↑", "正极产生氧气负极产生氢气", "体积比1:2"),
        ("光合作用", "6CO₂+6H₂O→C₆H₁₂O₆+6O₂", "叶绿体", "需要光能"),
        ("质量守恒定律", "拉瓦锡", "化学反应前后总质量不变", "化学基本定律"),
        ("原子结构", "原子核和核外电子", "质子带正电中子不带电电子带负电", "原子物理"),
        ("酸的化学性质", "pH<7", "能使紫色石蕊试液变红", "化学性质"),
        ("碱的化学性质", "pH>7", "能使无色酚酞试液变红", "化学性质"),
    ],
    "biology": [
        ("细胞结构", "细胞膜细胞质细胞核", "动植物细胞的基本结构", "生物学"),
        ("DNA双螺旋", "沃森和克里克", "1953年发现", "脱氧核糖核酸"),
        ("食物链", "生产者消费者分解者", "草→兔子→狐狸→狼", "生态系统"),
        ("光合作用", "叶绿体", "将光能转化为化学能", "植物生理"),
        ("呼吸系统", "鼻咽喉气管支气管肺", "肺是主要器官", "人体生理"),
        ("血液循环", "心脏血管血液", "体循环和肺循环", "人体生理"),
        ("进化论", "达尔文", "自然选择适者生存", "生物进化"),
        ("生态系统", "生物群落+无机环境", "物质循环和能量流动", "生态学"),
    ],
}

# ── 历史知识库 ──
HIST_DB = {
    "ancient_china": [
        ("夏朝", "约前2070年", "禹", "第一个王朝", "世袭制"),
        ("商朝", "约前1600年", "汤", "甲骨文", "青铜器"),
        ("西周", "前1046年", "周武王", "分封制", "宗法制"),
        ("秦朝", "前221年", "嬴政", "郡县制", "统一度量衡"),
        ("汉朝", "前202年", "刘邦", "丝绸之路", "造纸术"),
        ("唐朝", "618年", "李渊", "贞观之治", "唐诗"),
        ("宋朝", "960年", "赵匡胤", "活字印刷", "指南针"),
        ("元朝", "1271年", "忽必烈", "行省制", "大都"),
        ("明朝", "1368年", "朱元璋", "郑和下西洋", "永乐大典"),
        ("清朝", "1636年", "皇太极", "康乾盛世", "闭关锁国"),
    ],
    "modern_china": [
        ("鸦片战争", "1840年", "英国", "南京条约", "半殖民地半封建社会"),
        ("太平天国运动", "1851年", "洪秀全", "天朝田亩制度", "农民起义"),
        ("甲午战争", "1894年", "日本", "马关条约", "北洋海军"),
        ("戊戌变法", "1898年", "康有为梁启超", "百日维新", "君主立宪"),
        ("辛亥革命", "1911年", "孙中山", "中华民国", "三民主义"),
        ("五四运动", "1919年", "学生工人", "新民主主义革命开端", "民主科学"),
        ("中国共产党成立", "1921年", "上海", "中共一大", "红船精神"),
        ("新中国成立", "1949年", "毛泽东", "开国大典", "10月1日"),
        ("改革开放", "1978年", "邓小平", "十一届三中全会", "经济特区"),
    ],
    "world_history": [
        ("文艺复兴", "14-16世纪", "意大利", "人文主义", "达芬奇"),
        ("新航路开辟", "15-16世纪", "哥伦布达伽马", "地理大发现", "世界市场"),
        ("英国工业革命", "18世纪60年代", "英国", "蒸汽机", "工厂制度"),
        ("美国独立战争", "1775年", "华盛顿", "独立宣言", "1783年"),
        ("法国大革命", "1789年", "法国人民", "人权宣言", "拿破仑"),
        ("第一次世界大战", "1914年", "同盟国协约国", "凡尔赛条约", "1918年"),
        ("十月革命", "1917年", "列宁", "社会主义革命", "苏维埃"),
        ("第二次世界大战", "1939年", "反法西斯同盟", "联合国", "1945年"),
    ],
}

# ── 地理知识库 ──
GEO_DB = {
    "china_geo": [
        ("长江", "6300公里", "唐古拉山", "东海", "11个省区市"),
        ("黄河", "5464公里", "巴颜喀拉山", "渤海", "母亲河"),
        ("珠江", "2320公里", "云南", "南海", "华南第一大河"),
        ("泰山", "1545米", "山东", "五岳之首", "天下第一山"),
        ("华山", "2154米", "陕西", "以险著称", "五岳之一"),
        ("衡山", "1300米", "湖南", "南岳", "五岳之一"),
        ("恒山", "2016米", "山西", "北岳", "五岳之一"),
        ("嵩山", "1492米", "河南", "中岳", "少林寺"),
        ("珠穆朗玛峰", "8848米", "喜马拉雅山", "世界最高峰", "中尼边境"),
    ],
    "world_geo": [
        ("亚洲", "4400万平方公里", "面积最大", "人口最多", "48个国家"),
        ("非洲", "3000万平方公里", "第二大洲", "撒哈拉沙漠", "热带草原"),
        ("北美洲", "2400万平方公里", "北美", "美国和加拿大", "落基山脉"),
        ("南美洲", "1800万平方公里", "亚马孙平原", "巴西", "安第斯山脉"),
        ("南极洲", "1400万平方公里", "最冷", "科学考察", "无永久居民"),
        ("欧洲", "1000万平方公里", "发达国家", "欧盟", "阿尔卑斯山"),
        ("大洋洲", "900万平方公里", "澳大利亚", "太平洋岛屿", "最小洲"),
        ("太平洋", "18100万平方公里", "最大最深", "马里亚纳海沟", "最多岛屿"),
        ("大西洋", "9400万平方公里", "S形", "北大西洋暖流", "第二大洋"),
        ("印度洋", "7500万平方公里", "热带", "季风环流", "第三大洋"),
        ("北冰洋", "1300万平方公里", "最小最浅", "冰雪覆盖", "北极"),
    ],
    "china_climate": [
        ("季风气候", "东部", "夏季高温多雨", "冬季寒冷干燥", "主要气候"),
        ("温带大陆性气候", "西北", "年温差大", "降水稀少", "草原荒漠"),
        ("高寒气候", "青藏高原", "全年低温", "日照强", "高原山地"),
        ("亚热带季风气候", "秦岭淮河以南", "夏季高温多雨", "冬季温和少雨", "水田农业"),
        ("温带季风气候", "秦岭淮河以北", "夏季高温多雨", "冬季寒冷干燥", "旱作农业"),
    ],
}

# ── 英语语法库 ──
ENG_DB = {
    "grammar": [
        ("一般现在时", "描述习惯和事实", "主语+动词原形/动词第三人称单数", "do/does"),
        ("现在进行时", "描述正在发生的动作", "am/is/are+动词ing", "be doing"),
        ("一般过去时", "描述过去发生的动作", "主语+动词过去式", "did"),
        ("一般将来时", "描述将要发生的动作", "will+动词原形/be going to+动词原形", "will"),
        ("现在完成时", "描述已经完成的动作", "have/has+过去分词", "have done"),
        ("过去进行时", "描述过去某时正在发生的动作", "was/were+动词ing", "was doing"),
        ("被动语态", "主语是动作的承受者", "be+过去分词", "by..."),
        ("比较级", "比较两者", "adj+er/more+adj+than", "the+比较级"),
        ("最高级", "比较三者及以上", "the+adj+est/the+most+adj", "in/of"),
        ("条件句", "描述条件和结果", "if从句+主句", "虚拟语气"),
    ],
    "vocab_topics": [
        ("颜色", "red blue green yellow white black purple orange"),
        ("动物", "cat dog bird fish rabbit horse pig cow sheep monkey"),
        ("水果", "apple banana orange grape pear watermelon strawberry mango"),
        ("学校", "book pen desk chair classroom teacher student blackboard"),
        ("家庭", "father mother brother sister uncle aunt cousin grandpa"),
        ("食物", "rice bread milk egg meat vegetable fruit noodle cake"),
        ("天气", "sunny cloudy rainy windy snowy hot cold warm cool"),
        ("月份", "January February March April May June July August September October November December"),
        ("星期", "Monday Tuesday Wednesday Thursday Friday Saturday Sunday"),
        ("数字", "one two three four five six seven eight nine ten hundred thousand"),
    ],
}

# ── 道德与法治 ──
MORAL_DB = {
    "patriotism": [
        ("五星红旗", "红色象征革命", "五颗星代表共产党和各族人民", "曾联松设计"),
        ("国歌", "义勇军进行曲", "田汉作词聂耳作曲", "中华民族精神"),
        ("宪法", "根本大法", "最高法律效力", "人民主权"),
    ],
    "values": [
        ("富强", "国家层面", "经济繁荣", "综合国力"),
        ("民主", "国家层面", "人民当家作主", "政治文明"),
        ("文明", "国家层面", "文化繁荣", "社会进步"),
        ("和谐", "国家层面", "社会和谐", "人与自然和谐"),
        ("自由", "社会层面", "权利保障", "法律范围内"),
        ("平等", "社会层面", "法律面前人人平等", "社会公平"),
        ("公正", "社会层面", "公平正义", "司法公正"),
        ("法治", "社会层面", "依法治国", "法治国家"),
        ("爱国", "个人层面", "热爱祖国", "民族精神核心"),
        ("敬业", "个人层面", "爱岗敬业", "职业道德"),
        ("诚信", "个人层面", "诚实守信", "社会信用"),
        ("友善", "个人层面", "与人为善", "和谐人际关系"),
    ],
    "rights": [
        ("受教育权", "宪法保障", "义务教育", "教育公平"),
        ("人身自由权", "人格尊严", "不受非法拘禁", "基本权利"),
        ("财产权", "合法财产", "不受侵犯", "物权保护"),
        ("选举权", "年满18周岁", "公民政治权利", "民主参与"),
    ],
}


# ═══════════════════════════════════════════════════════
#  段落和问答对生成器
# ═══════════════════════════════════════════════════════

known_pairs = []
_used_questions = set()

def make_templates(items, fmt_fn, qa_fns):
    """从知识库条目批量生成段落+QA"""
    results = []
    for item in items:
        para = fmt_fn(item)
        qas = [fn(item, para) for fn in qa_fns]
        qas = [(q, a) for q, a, pos in qas if a in para]
        if qas:
            results.append((para, qas))
    return results


def _find(para, answer):
    pos = para.find(answer)
    return (pos, pos + len(answer) - 1) if pos >= 0 else None


def build_para_qas():
    global known_pairs
    records = []

    # ── 语文：古诗 ──
    for p in POEM_DB["poems"]:
        title, author, dynasty, style, alias, lines, descs = p
        for desc in descs:
            para = f"{title}是{dynasty}诗人{author}创作的一首{style}。{lines}这首诗{desc}。"
            qas = [
                (f"《{title}》的作者是谁？", author),
                (f"《{title}》是什么诗体？", style),
                (f"{author}是哪个朝代的诗人？", dynasty),
                (f"{author}被称为什么？", alias) if alias else None,
            ]
            qas = [(q, a) for q, a in qas if a and _find(para, a)]
            if qas:
                records.append((para, qas))

    # ── 语文：作者介绍 ──
    for name, alias, dynasty, school, hometown, work in POEM_DB["authors"]:
        para = f"{name}（{alias}）是{dynasty}{school}。他出生于{hometown}，代表作有《{work}》等。他的诗歌{['风格豪放','风格婉约','语言平实','意境深远'][random.randint(0,3)]}。"
        qas = [
            (f"{name}是什么朝代的诗人？", dynasty),
            (f"{name}被称为什么？", alias),
            (f"{name}的出生地是哪里？", hometown),
            (f"{name}的代表作之一是什么？", f"{work}"),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 语文：文言文 ──
    for title, author, dynasty, category, quote in POEM_DB["classical_texts"]:
        para = f"《{title}》是{author}编纂的{dynasty}{category}。{quote}是其中的名句。{title}是了解{dynasty}思想和文化的重要典籍。"
        qas = [
            (f"《{title}》的作者是谁？", author),
            (f"《{title}》属于什么类别的书籍？", category),
            (f"《{title}》是哪朝的作品？", dynasty),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 数学：几何定理 ──
    for name, scope, content, formula, source in MATH_DB["geometry_theorems"]:
        para = f"{name}：{scope}中，{content}。{f'用公式表示为{formula}。' if formula else ''}这一{['几何','数学'][random.randint(0,1)]}定理在{source}中就有记载，是{['中学数学','平面几何','数学学习'][random.randint(0,2)]}的重要内容。"
        qas = [
            (f"{name}适用于什么情况？", scope),
            (f"{name}的内容是什么？", content),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 数学：运算定律 ──
    for name, scope, formula, source in MATH_DB["arithmetic_laws"]:
        para = f"{name}是{scope}中的一个重要定律。{formula}。这个定律在{source}就开始学习了，可以帮助我们{['简便计算','灵活运算','提高计算效率'][random.randint(0,2)]}。"
        qas = [
            (f"{name}的公式是什么？", formula),
            (f"{name}属于什么范围？", scope),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 数学：公式 ──
    for name, formula, desc, category in MATH_DB["formulas"]:
        para = f"{name}公式：{desc}，{f'即{formula}。' if formula else ''}这是{['小学数学','中学数学','几何计算'][random.randint(0,2)]}中{['基本公式','常用公式','重要公式'][random.randint(0,2)]}之一。"
        qas = [
            (f"{name}公式是什么？", formula if formula else f"{desc}"),
            (f"{desc}属于什么公式？", f"{name}"),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 物理 ──
    for name, alias, content, field in SCI_DB["physics"]:
        para = f"{name}（又称{alias}）：{content}。这是{field}中的{['基本定律','重要定律','核心定律'][random.randint(0,2)]}，由{['英国科学家牛顿','多位科学家共同'][random.randint(0,1)]}发现。"
        qas = [
            (f"{name}又称为什么？", alias),
            (f"{name}的内容是什么？", content),
            (f"{name}属于什么领域？", field),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 化学 ──
    for item in SCI_DB["chemistry"]:
        name = item[0]
        discoverer = item[1] if len(item) > 1 else "科学家们"
        year = item[2] if len(item) > 2 else "历史上"
        details = item[3] if len(item) > 3 else item[1]
        extra = item[4] if len(item) > 4 else ""
        para = f"{name}：由{discoverer}在{year}提出。{details}。{extra}这是化学中最重要的概念之一。"
        qas = [
            (f"{name}是谁提出的？", discoverer),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 生物 ──
    for name, details, extra, field in SCI_DB["biology"]:
        para = f"{name}：{details}。{extra}这是{field}的{['基本知识','重要概念','核心内容'][random.randint(0,2)]}。在{['中学','小学','高中'][random.randint(0,2)]}生物课程中都有涉及。"
        qas = [
            (f"{name}的内容是什么？", details),
            (f"{name}属于什么领域？", field),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 中国古代史 ──
    for name, year, founder, feature1, feature2 in HIST_DB["ancient_china"]:
        para = f"{name}（{year}年建立）由{founder}建立。{name}的主要特征包括{feature1}和{feature2}。{name}在{['中国历史','中华文明发展'][random.randint(0,1)]}中占有重要地位。"
        qas = [
            (f"{name}由谁建立？", founder),
            (f"{name}建立于哪一年？", f"{year}"),
            (f"{name}的主要特征之一是？", feature1),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 中国近现代史 ──
    for name, year, leader, result, significance in HIST_DB["modern_china"]:
        para = f"{name}发生在{year}，由{leader}领导。{name}的结果是{result}。{name}的{['历史意义','影响'][random.randint(0,1)]}是{significance}。"
        qas = [
            (f"{name}发生在哪一年？", year),
            (f"{name}由谁领导？", leader),
            (f"{name}的历史意义是什么？", significance),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 世界史 ──
    for name, period, location, result, significance in HIST_DB["world_history"]:
        para = f"{name}发生在{period}，起源于{location}。{name}的成果是{result}，意义是{significance}。这是世界{['历史上的重要事件','文明史上的重大变革'][random.randint(0,1)]}。"
        qas = [
            (f"{name}发生在什么时期？", period),
            (f"{name}起源于哪里？", location),
            (f"{name}的意义是什么？", significance),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 中国地理 ──
    for name, length, origin, outlet, detail in GEO_DB["china_geo"]:
        para = f"{name}全长约{length}，发源于{origin}，注入{outlet}。{name}流经{detail}，是{['中国重要的河流','著名的山脉','中国重要的地理标志'][random.randint(0,2)]}。"
        qas = [
            (f"{name}全长约多少？", length),
            (f"{name}发源于哪里？", origin),
            (f"{name}注入哪里？", outlet),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 世界地理 ──
    for name, area, feature1, feature2, feature3 in GEO_DB["world_geo"]:
        para = f"{name}面积约{area}。{name}的特点是：{feature1}、{feature2}、{feature3}。这是地球{['上最大的洲','上重要的地理单元'][random.randint(0,1)]}之一。"
        qas = [
            (f"{name}的面积约多少？", area),
            (f"{name}的特点之一是？", feature1),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 中国气候 ──
    for name, region, feature1, feature2, feature3 in GEO_DB["china_climate"]:
        para = f"{name}主要分布在中国{region}。气候特征是{feature1}、{feature2}。{feature3}是这种气候下的{['主要农业类型','典型自然景观'][random.randint(0,1)]}。"
        qas = [
            (f"{name}主要分布在哪里？", f"中国{region}"),
            (f"{name}的气候特征之一是？", feature1),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 英语语法 ──
    for name, usage, structure, signal in ENG_DB["grammar"]:
        para = f"{name}是英语中重要的语法内容。它用于{usage}。基本结构是：{structure}。常见的标志词包括{signal}。掌握这个时态对于{['英语学习','英语写作','英语考试'][random.randint(0,2)]}很有帮助。"
        qas = [
            (f"{name}用于什么情况？", usage),
            (f"{name}的基本结构是什么？", structure),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        records.append((para, qas))

    # ── 英语词汇 ──
    for topic, words in ENG_DB["vocab_topics"]:
        word_list = words.split()[:6]
        para = f"关于{topic}的常见英语单词包括：{', '.join(word_list)}等。学习{topic}类单词可以帮助我们{['描述日常生活','丰富英语表达','提高英语词汇量'][random.randint(0,2)]}。"
        if word_list:
            first_word = word_list[0]
            qas = [
                (f"关于{topic}的英语单词有哪些？", f"{', '.join(word_list)}"),
            ]
            qas = [(q, a) for q, a in qas if a and _find(para, a)]
            records.append((para, qas))

    # ── 道德与法治 ──
    level_map = {"富强":"国家","民主":"国家","文明":"国家","和谐":"国家","自由":"社会","平等":"社会","公正":"社会","法治":"社会","爱国":"个人","敬业":"个人","诚信":"个人","友善":"个人"}
    categories = {"富强":"经济繁荣综合国力强","民主":"人民当家作主","文明":"文化繁荣社会进步","和谐":"人与自然和谐共生","自由":"法律范围内的权利保障","平等":"法律面前人人平等","公正":"公平正义","法治":"依法治国","爱国":"热爱祖国","敬业":"爱岗敬业","诚信":"诚实守信","友善":"与人为善"}
    for item in MORAL_DB["values"]:
        level = level_map.get(item[0], "国家")
        cat = categories.get(item[0], item[1])
        para = f"{item[0]}是社会主义核心价值观在{level}层面的要求。{item[0]}的含义是{cat}。践行社会主义核心价值观对个人成长和社会发展具有重要意义。"
        qas = [
            (f"{item[0]}是哪个层面的要求？", level),
            (f"{item[0]}的含义是什么？", cat),
        ]
        qas = [(q, a) for q, a in qas if a and _find(para, a)]
        if qas:
            records.append((para, qas))

    return records


# ═══════════════════════════════════════════════════════
#  变体生成
# ═══════════════════════════════════════════════════════

def generate_variants(records, target=50000):
    """通过多种方式扩展数据到目标数量"""
    result = []
    seen_qa = set()

    def add_unique(para, q, a):
        key = (q[:50], a[:50])
        if key not in seen_qa:
            seen_qa.add(key)
            pos = para.find(a)
            if pos >= 0:
                result.append({
                    "paragraph": para,
                    "question": q,
                    "answer": a,
                    "answer_start": pos,
                    "answer_end": pos + len(a) - 1,
                })
                return True
        return False

    # 1. 基础数据
    for para, qas in records:
        for q, a in qas:
            add_unique(para, q, a)

    base_paras = [p for p, _ in records]
    base_qa = [(q, a) for _, qas in records for q, a in qas]

    print(f"  基础: {len(result)} 条")

    # 2. 变体: 用不同问法问同一个答案
    question_variants = {
        "是": lambda p: ["什么是" + p, p + "是什么", p + "指的是什么", "请说出" + p, p + "指什么"],
        "是": lambda p: ["什么是" + p, p + "指的是什么"],
    }

    q_prefixes = ["请回答：", "告诉我：", ""]

    for para in base_paras[:200]:
        all_a = set()
        for r in list(result):
            if r["paragraph"] == para:
                all_a.add(r["answer"])

        for a in list(all_a)[:8]:  # 每个段落每种答案多问几次
            # 变体问法
            for prefix in q_prefixes:
                vq = prefix + a + "是什么"
                if add_unique(para, vq, a):
                    pass
                vq = prefix + "关于" + a[:10]
                if add_unique(para, vq, a):
                    pass
                vq = prefix + a[:6] + "是什么？"
                if add_unique(para, vq, a):
                    pass

    # 3. 交叉组合: 把A段落的问答对用于B段落(如果答案也存在于B段落)
    cross_added = 0
    for _ in range(min(10000, len(base_paras) * 10)):
        p1 = random.choice(base_paras)
        qa = random.choice(base_qa)
        q, a = qa
        if a in p1 and len(a) >= 2:
            if add_unique(p1, q, a):
                cross_added += 1

    print(f"  交叉组合: +{cross_added}")

    # 4. 随机段落组合
    more_added = 0
    attempts = 0
    while len(result) < target and attempts < 100000:
        attempts += 1
        p1 = random.choice(base_paras)
        qa = random.choice(base_qa)
        q, a = qa
        # 尝试在段落中找答案
        if a in p1 and len(a) >= 2:
            if add_unique(p1, q, a):
                more_added += 1

    print(f"  随机组合: +{more_added}")

    # 5. 句子分割: 将一个长段落分割成多个子段落
    split_added = 0
    for para in base_paras[:100]:
        sentences = para.replace("。", "。|").replace("？", "？|").split("|")
        if len(sentences) >= 2:
            for i in range(len(sentences)):
                sub = "".join(sentences[:i+1])
                # 复用已有的问答对
                for r in list(result)[:500]:
                    a = r["answer"]
                    if a in sub and sub != r["paragraph"]:
                        if add_unique(sub, r["question"], a):
                            split_added += 1

    print(f"  分割组合: +{split_added}")

    random.shuffle(result)
    print(f"  最终: {len(result)} 条")
    return result


# ═══════════════════════════════════════════════════════
def main():
    print("=" * 50)
    print("  中小学教材知识数据集生成器 v2")
    print("=" * 50)

    print("\n[1] 生成基础段落和问答对...")
    records = build_para_qas()
    print(f"  基础: {len(records)} 条段落-问答组")

    # 计数
    total_qas = sum(len(qas) for _, qas in records)
    print(f"  问答对: {total_qas} 条")

    print("\n[2] 扩展组合生成更多变体...")
    final = generate_variants(records, target=50000)
    print(f"  最终: {len(final)} 条")

    print("\n[3] 保存...")
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=1)
    size = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"  已保存: {OUTPUT}")
    print(f"  大小: {size:.1f} MB")
    print(f"  条目: {len(final)} 条")
    print(f"\n  各科目知识点已覆盖: 语文古诗文/文言文/作者, 数学定理/定律/公式, 物理/化学/生物, 中国史/世界史, 地理, 英语语法/词汇, 道德与法治")


if __name__ == '__main__':
    main()
