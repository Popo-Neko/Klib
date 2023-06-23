class Subtitle:
    def __init__(self, *labels, name):
        assert all(isinstance(label, Labels) for label in labels), "Not all variables are Labels"
        self.keywords = [words for label in labels for words in label]
        self.labels = {label.name: label for label in labels}
        self.name = name
        self.keys = list(self.labels.keys())
        self.index = 0

    def __len__(self):
        return len(self.labels)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.keys):
            raise StopIteration
        key = self.keys[self.index]
        value = self.labels[key]
        self.index += 1
        return value

    def see_labels(self):
        return [i.name for i in self.labels]

    def count_keywords(self):
        return len(self.keywords)

    def __getitem__(self, key):
        return self.labels[key]

    def __setitem__(self, key, value):
        self.labels[key] = value

    def add_label(self, label):
        assert type(label) == Labels, "the added label must be the Class Labels"
        self.labels[label.name] = label
        self.keywords.extend(label.keywords)


class Labels:
    def __init__(self, keywords, name):
        assert all(isinstance(word, str) for word in keywords), "Not all keywords are str"
        self.keywords = keywords
        self.name = name
        self.index = 0

    def __len__(self):
        return len(self.keywords)

    def __getitem__(self, index):
        return self.keywords[index]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.keywords):
            raise StopIteration
        value = self.keywords[self.index]
        self.index += 1
        return value

    def add(self, iterable):
        if isinstance(iterable, str):
            self.keywords.append(iterable)
        else:
            for i in iterable:
                assert type(i) == str, "the added keywords are not all str"
                self.keywords.append(i)


pregnant_key_words = ['周', '+', '孕', '阴道炎', '烧心', '耻骨', '子宫', '宫颈', '白带', 'NT', '唐', '唐氏',
                      '无创', '排畸', '二维', '四维', '甲供三项', '糖耐', '胎', 'ABO溶血', '宫缩', '羊水', '会阴',
                      '妊娠', '甲状腺', '游离', '叶酸', 'B超']

baby_key_words = ['宝宝', '岁', '产后', '小孩', '娃', '母乳', '奶粉', '孩子', '女儿', '儿子', 'BB', '男', '女', '奶量',
                  '哺乳', '翻身', '粑粑', '月', '出生', '小朋友', '新生儿', '宝']

pic_key_words = ['图', '看看', '如图', '报告单', '照片', '图片', '看', '这样']

nutrition_key_words = ['营养', '补钙', '缺钙', '补铁', '维生素', '零食', 'dha', 'DHA', '食欲']

# 叶酸
YeSuan = ['叶酸']
YeSuan = Labels(YeSuan, '叶酸')

# 营养
nutrition = ['营养', '荤', '素', '零食', '少食多餐', '食欲']
nutrition = Labels(nutrition, '营养')

# 补钙
Ca = ['钙']
Ca = Labels(Ca, '补钙')

# 补铁
Fe = ['铁']
Fe = Labels(Fe, '补铁')

# 维生素
vitamin = ['维生素', '维他命', '多维胶囊', '爱乐维', '黄金素', '复合维生素', '胡萝卜素', '维A', '维D', '维B', '维a', '维d',
           '维b']
vitamin = Labels(vitamin, '维生素')

# 饮食药品禁忌
food_taboo = ['忌口', '能吃', '可以吃', '要吃', '擦了', '涂抹']
food_taboo = Labels(food_taboo, '饮食药品禁忌')

# DHA
DHA = ['dha', 'DHA']
DHA = Labels(DHA, 'DHA')

# Subtitle: nutrition
Nutrition = Subtitle(YeSuan, nutrition, Ca, Fe, vitamin, food_taboo, DHA, name='营养与饮食')

# 阴道私处
vagina = ['阴道', '私处', '白带', '外阴', '会阴', '阴道流血', '小便', '阴唇']
vagina = Labels(vagina, '阴道私处')

# 贫血
anemia = ['贫血', '血小板', '红细胞', '血红蛋白', '红细胞', '血清铁', '铁蛋白']
anemia = Labels(anemia, '妊娠期贫血')

# 糖尿病
diabetes = ['血糖', '糖尿', '尿糖', '葡萄糖耐性', '控糖']
diabetes = Labels(diabetes, '孕期糖尿病')

# 高血压
hypertension = ['血压']
hypertension = Labels(hypertension, '孕期高血压')

# 甲状腺疾病
thyroid_gland = ['甲状腺', '甲亢', '甲减']
thyroid_gland = Labels(thyroid_gland, '甲状腺疾病')

# 静脉曲张
varicose_veins = ['静脉曲张', '青筋', '蜘蛛痣']
varicose_veins = Labels(varicose_veins, '静脉曲张(下肢青筋)')

# 湿疹
eczema = ['湿疹']
eczema = Labels(eczema, '湿疹')

# 痔疮
piles = ['痔疮']
piles = Labels(piles, '痔疮')

# 感冒
cold = ['感冒', '着凉', '风寒', '流感']
cold = Labels(cold, '孕期感冒')

# 孕吐
sickness = ['恶心', '孕吐', '呕吐', '吐']
sickness = Labels(sickness, '孕吐')

# 腹部胀痛
stomach = ['腹', '肚子']
stomach = Labels(stomach, '腹部胀痛')

# 孕期烧心
heartburn = ['烧心', '灼烧感', '反胃']
heartburn = Labels(heartburn, '烧心')

# 皮肤与妊娠纹
skin = ['妊娠纹', '皮肤']
skin = Labels(skin, '皮肤与妊娠纹')

# 腰疼
lumbar_pain = ['腰']
lumbar_pain = Labels(lumbar_pain, '腰疼')

# 耻骨疼
pubic_bone = ['耻骨', '耻骨']
pubic_bone = Labels(pubic_bone, '耻骨疼')

# 关节疼
joint = ['关节']
joint = Labels(joint, '关节疼')

# 便秘
constipation = ['便秘']
constipation = Labels(constipation, '便秘')

# 水肿
swelling = ['水肿']
swelling = Labels(swelling, '水肿')

# 抽筋
cramp = ['抽筋']
cramp = Labels(cramp, '抽筋')

# 先兆流产
miscarriage = lumbar_pain.keywords + stomach.keywords + vagina.keywords + ['分泌物', '腹部压力', '流产', '早产']
miscarriage = Labels(miscarriage, '先兆流产')

# 大小便
stool_urine = ['大便', '小便', '尿频', '失禁', '尿']
stool_urine = Labels(stool_urine, '大小便')

# 瘢痕子宫
scarred_uterus = ['剖宫产', '瘢痕', '子宫肌瘤', '内膜异位']
scarred_uterus = Labels(scarred_uterus, '瘢痕子宫')

# 宫颈疾病
cervix = ['宫颈', '脱垂']
cervix = Labels(cervix, '宫颈疾病')

# Subtitle：illness
illness = Subtitle(vagina, anemia, diabetes, hypertension, thyroid_gland, varicose_veins, eczema, piles,
                   cold, sickness, stomach, heartburn, skin, lumbar_pain, pubic_bone, joint, constipation,
                   swelling, cramp, miscarriage, stool_urine, scarred_uterus, cervix, name='疾病、不适与身体变化')


illness_key_words = ['痛', '疼', '难受', '不适', '阴道', '贫血', '糖尿病', '胰岛素', '高血压', '甲状腺疾病',
                     '静脉曲张', '湿疹', '感冒', '异味', '孕吐', '腹胀', '腹痛', '烧心', '腰痛', '耻骨', '关节',
                     '便秘', '水肿', '抽筋', '私处', '分泌物', '瘙痒', '搔痒', '流血', '流产', '先兆', '子痫',
                     '尿频', '尿失禁', '瘢痕', '子宫', '牙疼', '疱疹', '中耳炎', '结缔组织', '宫颈', '肛门',
                     '息肉', '阴唇', '胸闷', '气短', '头晕', '皮疹', '宫缩', '痒', '水泡', '黄色', '白色', '肝吸虫',
                     '手麻', '感染', '辐射', '心慌', '血丝', '咳嗽', '手麻', '上火', '出血', '感染', '发紧', '燥热',
                     '白带', '反胃', '甲亢', '大便', '痔疮', '血窦', '红斑狼疮']

daily_key_words = ['乳房', '乳头', '乳晕', '乳腺', '火车', '高铁', '打车', '长途', '飞机', '性欲', '性生活',
                   '散步', '运动', '跑步', '瑜伽', '体重', '心情', '抑郁', '压抑', '烦躁', '烦心', '烦', '胎教',
                   '化妆品']

fetus_key_words = ['胎儿', '头围', '腹围', '双顶径', '脑侧室', '侧脑室', '胎心', '胎动', '鼻骨', '心室', '绕颈']

production_key_words = ['破水', '见红', '顺产', '剖腹产', '分娩', '早产', '待产包', '栓塞', '入盆']

test_key_words = ['NT', 'nt', '唐氏', '唐筛', '无创', 'DNA', 'dna', 'NIPT', 'nipt', '排畸', '穿刺', 'B超', '二维', '三维', '四维',
                  '甲功三项', '糖耐', '胎心监测', '监测', '染色体', '地贫', '筛查', 'ABO', 'abo', '溶血', '心电图',
                  '产检', '基因', '检测', '脐带绕颈', '羊水', '胎盘', '孕酮', 'hcg', 'HCG', 'B族链球菌',
                  '孕囊', '促甲状腺素', '甲胎蛋白', '血小板', '血红蛋白', '脉络丛囊肿', '肾盂分离', '肠管回声增强',
                  '肠管', '单脐动脉', '胎位', '优生', '宫腔积液', '子宫肌瘤', '胎膜', '凝血', '偏高', '偏低', '血压',
                  '辐射', '酶', '分析', '甲状腺', 'b超', 'B超', '结果', '单子', '血糖', '尿检', '弓形虫', '抽血']

# 产检
NT = ['nt', 'NT', 'Nt', 'nT']
NT = Labels(NT, 'NT检查')

# 唐氏筛查
Downs = ['唐氏', '血清学', '唐筛', '唐', 'hcg', 'HCG', 'hCG', 'PAPP-A', 'β-hCG']
Downs = Labels(Downs, '唐氏筛查')

# 无创DNA
NIPT = ['无创', 'DNA', 'nipt', 'NIPT', '游离DNA']
NIPT = Labels(NIPT, '无创DNA')

# 大小排畸
Deformity = ['排畸']
Deformity = Labels(Deformity, '大小排畸')

# 羊水穿刺
amniocentesis = ['羊水穿刺', '羊穿']
amniocentesis = Labels(amniocentesis, '羊水穿刺')

# B超
ultrasound = ['b超', '超声', 'B超', '彩超']
ultrasound = Labels(ultrasound, 'B超')

# 二维
twodimen = ['二维', '2维']
twodimen = Labels(twodimen, '二维')

# 三维
threedimen = ['三维', '3维']
threedimen = Labels(threedimen, '三维')

# 四维
fourdimen = ['四维', '4维']
fourdimen = Labels(fourdimen, '四维')

# 甲功三项
JiaGong = ['甲状腺', '甲功']
JiaGong = Labels(JiaGong, '甲功三项')

# 糖耐
sugar_tolerance = ['糖耐']
sugar_tolerance = Labels(sugar_tolerance, '糖耐')

# 胎心监测
TaiXin = ['胎心']
TaiXin = Labels(TaiXin, '胎心监测')

# 染色体检查
chromosome = ['染色体']
chromosome = Labels(chromosome, '染色体')

# 地中海贫血
DiPing = ['地中海贫血', '地贫']
DiPing = Labels(DiPing, '地中海贫血')

# ABO溶血
ABO = ['abo', 'ABO', '溶血']
ABO = Labels(ABO, 'ABO溶血')

# 脐带绕颈
QiDai = ['脐带绕颈', '绕颈', '脐带']
QiDai = Labels(QiDai, '脐带绕颈')

# 羊水
SheepWater = ['羊水']
SheepWater = Labels(SheepWater, '羊水')

# 胎盘
TaiPan = ['胎盘']
TaiPan = Labels(TaiPan, '胎盘')

# 孕酮
YunTong = ['孕酮']
YunTong = Labels(YunTong, '孕酮')

# 孕囊
YunNang = ['孕囊']
YunNang = Labels(YunNang, '孕囊')

# 血检
blood = ['红细胞', '血红蛋白', '血小板']
blood = Labels(blood, '血检')

# 脉络丛囊肿
MaiLuo = ['脉络丛囊肿']
MaiLuo = Labels(MaiLuo, '脉络丛囊肿')

# 肾盂分离
ShenYu = ['肾盂分离']
ShenYu = Labels(ShenYu, '肾盂分离')

# 单脐动脉
DanQi = ['单脐动脉']
DanQi = Labels(DanQi, '单脐动脉')

# 胎位
TaiWei = ['胎位']
TaiWei = Labels(TaiWei, '胎位')

# 优生检查(torch)
torch = ['优生五项', 'torch', '巨细胞病毒', '弓形虫', '风疹病毒', '单纯胞疹病毒', 'IgM', 'igm', 'IgG', 'igg', 'hsv', 'HSV']
torch = Labels(torch, '优生检查(torch)')

# 宫腔积液
JiYe = ['宫腔积液']
JiYe = Labels(JiYe, '宫腔积液')

# 子宫肌瘤
JiLiu = ['肌瘤']
JiLiu = Labels(JiLiu, '子宫肌瘤')

# Subtitle: test
test = Subtitle(JiLiu, JiYe, torch, TaiWei, DanQi, ShenYu, MaiLuo, blood, YunNang, YunTong, TaiPan, SheepWater,
                QiDai, ABO, DiPing, chromosome, TaiXin, sugar_tolerance, JiaGong, fourdimen, threedimen, twodimen,
                ultrasound, amniocentesis, Deformity, NIPT, Downs, NT, name='孕检专题')

# 胎儿头围
head = ['头围', '头顶']
head = Labels(head, '胎儿头围')

# 胎儿腹围
FuWei = ['腹围']
FuWei = Labels(FuWei, '胎儿腹围')

# 双顶径
ShuangDingJing = ['双顶径']
ShuangDingJing = Labels(ShuangDingJing, '双顶径')

# 侧脑室
CeNaoShi = ['侧脑', '脑侧']
CeNaoShi = Labels(CeNaoShi, '侧脑室')

# 胎动
TaiDong = ['胎动']
TaiDong = Labels(TaiDong, '胎动')

# 鼻骨
BiGu = ['鼻骨']
BiGu = Labels(BiGu, '鼻骨')

# 胎儿心脏
heart = ['心脏', '心室', '双室']
heart = Labels(heart, '胎儿心脏')

# Subtitle: fetus
fetus = Subtitle(heart, BiGu, TaiDong, CeNaoShi, ShuangDingJing, FuWei, head, name='胎儿发育')

# 临产征兆
pre_production = ['临产', '入盆']
pre_production = Labels(pre_production, '临产征兆')

# 破水
water = ['破水']
water = Labels(water, '破水')

# 宫缩
GongSuo = ['宫缩']
GongSuo = Labels(GongSuo, '宫缩')

# 顺产
ShunChang = ['顺产']
ShunChang = Labels(ShunChang, '顺产')

# 剖腹产
PouFu = ['剖腹']
PouFu = Labels(PouFu, '剖腹')

# 无痛分娩
WuFen = ['无痛分娩']
WuFen = Labels(WuFen, '无痛分娩')

# 早产
ZaoChan = ['早产']
ZaoChan = Labels(ZaoChan, '早产')

# 待产包
DaiChanBao = ['待产包']
DaiChanBao = Labels(DaiChanBao, '待产包')

# 分娩
FenMian = ['分娩']
FenMian = Labels(FenMian, '分娩')

# 栓塞
ShuanSe = ['栓塞']
ShuanSe = Labels(ShuanSe, '栓塞')

# Subtitle : production
production = Subtitle(ShuanSe, FenMian, DaiChanBao, ZaoChan, WuFen, PouFu, ShunChang, GongSuo, water,
                      pre_production, name='临产与分娩')


