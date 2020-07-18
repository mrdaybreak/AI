import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SOS_token = 0
EOS_token = 1


class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {0: '<SOS>', 1: '<EOS>', -1: '<UNK>'}
        self.idx = 2

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def __call__(self, word):
        if not word in self.word2idx:
            return -1
        return self.word2idx[word]

    def __len__(self):
        return self.idx


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # input_size代表输入语言的所有单词的数量，hidden_size代表网络的隐藏层节点数
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        # seq_len = 1, batch = 1
        embedded = self.embedding(input).view(1, 1, self.hidden_size)
        output, hidden = self.gru(embedded, hidden)
        return hidden

    def initHidde(self):
        return torch.zeros(1, 1, self.hidden_size, device=devices)

    def sample(self, seq_list):
        word_inds = torch.LongTensor(seq_list).to(devices)
        h = self.initHidde()
        for word_tensor in word_inds:
            h = self.forward(word_tensor, h)
        return h


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.maxlen = 10
        # output_size是输出语言的所有单词的数量，hidden_size是GRU网络的隐藏层节点数
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        # Linear的作用是将前面的GRU的输出结果变成目标语言的单词的长度
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, seq_input, hidden):
        output = self.embedding(seq_input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def sample(self, pre_hidden):
        input = torch.tensor([SOS_token], device=devices)
        hidden = pre_hidden
        res = [SOS_token]
        for i in range(self.maxlen):
            output, hidden = self.forward(input, hidden)
            topv, topi = output.topk(1)
            print(topi)
            if topi.item() == EOS_token:
                res.append(EOS_token)
                break
            else:
                res.append(topi.item())
            print(res)
            # 将生成的单词作为下一时刻的输入， sueeze去掉纬度为1的纬度， detach保证梯度不传导
            input = topi.squeeze().detach()
        return res


def sentence2tensor(lang, sentence):
    indexes = [lang(word) for word in sentence.split()]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=devices).view(-1, 1)


def pair2tensor(pair):
    input_tensor = sentence2tensor(lan1, pair[0])
    target_tensor = sentence2tensor(lan2, pair[1])
    return (input_tensor, target_tensor)

lan1 = Vocabulary()
lan2 = Vocabulary()
data = [['四海网讯，近日，有媒体报道称：章子怡真怀孕了!报道还援引知情人士消息称，“章子怡怀孕大概四五个月，预产期是年底前后，现在已经不接工作了。”这到底是怎么回事?消息是真是假?针对此消息，23日晚8时30分，华西都市报记者迅速联系上了与章子怡家里关系极好的知情人士，这位人士向华西都市报记者证实说：“子怡这次确实怀孕了。她已经36岁了，也该怀孕了。章子怡怀上汪峰的孩子后，子怡的父母亲十分高兴。子怡的母亲，已开始悉心照料女儿了。子怡的预产期大概是今年12月底。”当晚9时，华西都市报记者为了求证章子怡怀孕消息，又电话联系章子怡的亲哥哥章子男，但电话通了，一直没有人<Paragraph>接听。有关章子怡怀孕的新闻自从2013年9月份章子怡和汪峰恋情以来，就被传N遍了!不过，时间跨入2015年，事情却发生着微妙的变化。2015年3月21日，章子怡担任制片人的电影《从天儿降》开机，在开机发布会上几张合影，让网友又燃起了好奇心：“章子怡真的怀孕了吗?”但后据证实，章子怡的“大肚照”只是影片宣传的噱头。过了四个月的7月22日，《太平轮》新一轮宣传，章子怡又被发现状态不佳，不时深呼吸，不自觉想捂住肚子，又觉得不妥。然后在8月的一天，章子怡和朋友吃饭，在酒店门口被风行工作室拍到了，疑似有孕在身!今年7月11日，汪峰本来在上海要举行演唱会，后来因为台风“灿鸿”取消了。而消息人士称，汪峰原来打算在演唱会上当着章子怡的面宣布重大消息，而且章子怡已经赴上海准备参加演唱会了，怎知遇到台风，只好延期，相信9月26日的演唱会应该还会有惊喜大白天下吧。'],
        ['中新社西宁11月22日电<Paragraph>(赵凛松)青海省林业厅野生动植物和自然保护区管理局高级工程师张毓22日向中新社记者确认：“经过中国林业科学院、中科院新疆生态与地理研究所和青海省林业厅的共同认定，出现在青海省海西州境内的三只体型较大的鸟为世界极度濒危的红鹳目红鹳科红鹳属的大红鹳。”11月18日，青海省海西州可鲁克湖—托素湖国家级陆生野生动物疫源疫病监测站在野外监测巡护过程中，在可鲁克湖西南岸入水口盐沼滩发现三只体型较大的鸟类。张毓说：“此前在该区域从未发现过这种体型的鸟类。”可鲁克湖—托素湖位于青海省柴达木盆地东北部，海拔2800米，水域湿地环境内的优势种动物主要是水禽，共有30余种。根据拍摄的照片以及视频，张毓根据动物学体型得出了初步结论，然后会同中国林业科学院和中科院新疆生态与地理研究所的相关专家，确认了这三只鸟为红鹳目红鹳科红鹳属的大红鹳。大红鹳也称为大火烈鸟、红鹤等，三只鸟类特征为大红鹳亚成体。根据世界自然保护联盟、世界濒危动物红色名录，该鸟主要分布于非洲、中亚、南亚等区域，分布广、种群数量较大，无威胁因子，以往在中国并无分布。但1997年在新疆野外首次发现并确定该鸟在中国境内有分布，为中国鸟类新纪录，2012年在四川也发现一只该鸟亚成体。此次野外发现在中国属第三次。“我们现在还无法判断这三只鸟从何而来。不过我个人倾向于是从中亚国家迁徙至此。”张毓强调说，该种鸟国内也有人工饲养，因此也有人判断为从动物园逃逸。“我们对这三只鸟进行了详尽的记录，如果明年这个时间还在此地出现这种鸟，那就能肯定是迁徙的鸟类，而不是从动物园里跑出来的。”由于目前可鲁克湖—托素湖已开始结冰，鸟类采食困难，不排除三只鸟由于无法获得能量补给而进行远距离迁飞的可能。青海省林业厅野生动物行政主管部门将随时做好野外救护的各项准备工作。(完)', '知情人透露章子怡怀孕后，父母很高兴。章母已开始悉心照料。据悉，预产期大概是12月底','青海首次野外发现濒危大火烈鸟 尚不清楚具体来源'],
        ['本报讯(记者<Paragraph>陈雪<Paragraph>实习生<Paragraph>王健)日前，在我省公安机关公布的数据显示，在用药水泡制豆芽的犯罪行为的专项打击中，全省共抓获犯罪嫌疑人270名，查获生产窝点259个，打掉团伙94个。受此影响，延安的蔬菜市场上，已鲜见豆芽出售，即使是有个别商贩出售，价格已涨到每斤4元。查处严格商贩干脆不卖豆芽了10月份，延安已有些凉意。在这个小城里，大家已经达到了一个共识：不愿吃豆芽，豆芽也不容易买到。在位于二道街的市供销农贸市场里，大部分蔬菜摊都没有豆芽销售，只有个别摊位把少量豆芽放在隐蔽处卖。在一个蔬菜摊位前的货架底下，上下相扣的俩盆子里，装有在水里浸泡的豆芽。摊主表示，现在不让卖豆芽，不敢摆上货架，只好放得隐蔽些。据了解，该摊位大概有豆芽七斤左右，每斤卖4元钱。据悉，原来延安每斤豆芽2元左右。记者走访了延安其他几大农贸市场。在延安东关街东盛农贸市场内，也几乎找不到一家卖豆芽的摊位。只有一位摊主，周边人都知道他那儿有几斤绿豆大豆芽，市民需要打听方才能够得知。一位摊主告诉记者，他卖蔬菜好几年，一直有豆芽，但是现在查处严格，生产豆芽必须有许可证。“有关部门来查，必须清楚来源，否则就不让卖，这样给我们带来很多麻烦。干脆就不卖了。”这位摊主说，以前很多个人在自家就能做豆芽，现在流程要求严格，很多人也怕麻烦不做了。此外，一直有“毒豆芽”被查处公布，市民们也就不敢吃豆芽了，前来询问的也很少。如何辨别“毒豆芽”不长毛根据悉，所谓“毒豆芽”，是指在豆芽里添加化学药品“6-苄基腺嘌呤”，俗名无根水、无根豆芽素。9月29日，延长县质监局联合工商局和刑警大队，对七里村镇王良沟、苏家沟和西河子沟三处生产销售毒豆芽黑窝点进行查处。对西河子沟农贸市场销售的豆芽进行抽样检测，经省质量技术监督局鉴定，该批豆芽含无根水，这是一种植物生长调节剂，能提高豆芽产量。但这种被添加“无根水”的豆芽，长期食用后却会致癌，导致胎儿畸形等危害。与传统方法培育的豆芽不同，这种毒豆芽芽秆粗壮、饱满晶莹，外形非常好看。延长县质监局副局长郭景峰称，这次联合执法，抽出豆芽样品送到西安，经检验豆芽添加无根素高达2300多倍，对人体危害非常大，这种豆芽不长毛根，生产下来又长又脆。据涉案人员交代，不放无根水的豆芽，产量少且外形不好看，利润也很低。这种豆芽比较白、比较嫩，卖得也好，可以增产30%。此外，近日子长县、吴起县也捣毁了一些豆芽黑窝点。日前，黄陵首例生产销售毒豆芽案件开庭审理。一对夫妻将无根水添加到豆芽中，16万余斤毒豆芽流入市场，二人因此领刑。此外，二人也被禁止在两年内从事生产销售食品类行业及相关活动。','延安豆芽现已疯涨至每斤四元 查处严格后商贩不卖豆芽'],
        ['【环球网综合报道】据韩国纽西斯通讯社6月14日报道，韩国水原市地方法院14日审理了一起金条走私案，犯罪嫌疑人边某被判处有期徒刑1年6个月，累计罚款55亿韩元(约合人民币3080万元)。审判部表示，被告人在衣服内藏有的金条总价值55亿韩元，犯罪情节严重，但考虑到边某反省态度端正，且走私中获得的私利不多，因此判处其有期徒刑1年零6个月，缓期三年执行。2013年8月17日，边某将4kg的金条藏在衣内从山东威海出发前往京畿道平泽市，之后又通过同样的方法走私金条102kg。(实习编译：李婷婷<Paragraph>审稿：李小飞)','韩国一女子从中国走私106kg金条获刑1年6个月，这批金条总价值55亿韩元。'],
        ['错漏百出的教科书。授课老师挑出《计算机应用基础》百多处错误、不妥之处反被处分<Paragraph>引出教科书名利之争一本高职学生使用的计算机教材，仅前三章已被授课老师挑出了68处错误。他向学校反映后，反而遭到了处分。事情发生在广东外语艺术职业学院，事情匪夷所思的背后，其实隐藏着复杂的利益纠葛。而这本出现多处“硬伤”的教科书，至今仍在该校学生手中继续使用。学校方面表示，已经组织教学委员会及校外专家对教材展开鉴定，若发现重大知识性错误，将立即停用。错得太离谱：授课老师边上课边纠错广东外语艺术职业学院副教授叶克江的桌面上，摆着一本《计算机应用基础》教材，里面密密麻麻地布满红色笔圈出的修订。“我前后看了不下3次，错漏触目惊心”，他说，自己一边一字不漏地阅读，一边对比实际操作的软件，并对读其他出版社同类教材。仅前三章，他声称已挑出68个错误和87个不妥之处。他将错误分成知识性错误和语句表达的不妥之处。不妥之处像错字漏字方面，比如“计算精度越来越高”，中间的“越”字不见了，“英文字符”也缺“文”；知识描述方面同样明显，如“单击”变“双击”，“右侧”成“左侧”，“代码”成“代表”，常人一眼就看得出确实是硬伤。最关键的是，该教科书以windows7操作系统为主要工作环境，但其中有的内容仍以WindowsXP系统进行表述。结果讲到窗口的“工具栏”时，WindowsXP中的确存在，但Windows7系统已更新了，学生看书时一头雾水。目前，全校一年级新生共订购了2700多本该教材，而叶克江上课时，一边讲解一边纠错。不少学生觉得改起来麻烦，甚至连教科书都不带了。对于文中一些“语句不通之处”，叶克江甚至放话：“学校有能读懂这一句话的，请与我联系，本人砸锅卖铁提供10万元悬赏奖金。”挑错后被处分：“盗用书号+私印”PK胡编乱造这本教材，其实是叶克江同校教师、学校计算机公共教研室主任杨伟杰主编的。后者在旁听叶克江授课时，叶克江当着他的面严厉予以纠正，多名学生和听课老师面面相觑。叶克江纠错时不仅执着，还很高调。去年10月初，他向学校教务处反映该书的这些错误，同时联系上出版该书的高等教育出版社，但没等到调查结果。同月21日，他将一篇名为“‘国家级垃圾教材’是如何炼成的”的邮件，通过校内邮件系统发给了全校所有老师。洋洋洒洒3000字中，叶克江言辞激烈，称该书主编杨伟杰利用手中“权力”选用自编的教材，才致使“垃圾”教材“误人子弟”，并表示“对本文的一切描述负完全责任”。此举招致学校极快的反应。去年11月1日，学校给予叶克江通报批评处分，称他“群发邮件，混淆是非，作出不实指责，对学院和相关人员的声誉造成不良影响。”在叶克江的角度，他是为捍卫教科书质量而战；但有人认为，这是一场同校老师间的教科书之争。这缘起于学校从去年9月起，叶克江自己编写的计算机基础教材在被连续使用4年后，被同事的杨伟杰版取而代之。2013年5月，学校对计算机教材的选用工作中，同时收到了两人送审的样书。校方表示，之所以“弃叶选杨”，是因为叶克江送审的教材，盗用了别的书号，自己再私自印刷了几百本来送审，这是不正当的。“不是盗用，是借用”，对此，叶克江感到十分不忿。过去4年，学校使用他的书作教材时，已经默认他先“借”书号、后印制教材的方式。而学校解释给他处分时还重提此事，说要继续调查，他认为是避重就轻，“反而不直接鉴定新教材是否有质量问题，就是想扯皮。”校方：最终鉴定结果未出来出版社：错误率“难以理解”一本的确存在错漏的教科书，目前仍在学校内使用。教务处相关负责人说，按学校的教材选用管理办法，课程教材的选用由课程负责人组织任课教师集体研究讨论后提出，教研室进行初审，系（部）主管教学的领导复审确定后报教务处。教材一经选定不得随意更改。他说，之所以处分叶克江，是因为他采取不当方式闹得沸沸扬扬。“学校也是错误教材的受害者，而教材有错就要向出版社纠正。”学校宣传部江老师则表示，学校之所以至今尚未表态和处理，是因为最终鉴定结果尚未形成。如果鉴定结果是本书存在重大的知识性错误，经教学委员会确认后，学校将立即停止使用；若存在校对上的错漏，学校将追究出版社的责任。对此，出版《计算机应用基础》的高等教育出版社表示，教材的编写错误率一般控制在万分之一，如果真的是前三章就挑出了68个错误，实在是“难以理解”。出版社会向此书的责任编辑反映，挑错者可以将书中错误一一列出寄给他们，以便改正。教材名利场：出书容易、能抽版税、“职称”加分一些高职教师说，编写教科书是不少老师的“兵家必争之地”。据广州市轻工高级职业技术学校一位老师透露，教科书编写相对简单，却带给老师不少利益，基本能拿到15%左右的版税，对教师评职称也很有帮助。“编教科书的难度是业内公认最容易的，同类型教材很多，不涉及有难度的学术问题，左抄抄、右抄抄就可以出本书。”他说，有些错误不可避免，因为电脑更新得比书本快，所以一般是边上课边纠错。但基础性知识性错误又另当别论。广东机电职业技术学院的一名老师也认为，编写教科书可以说是“一家便宜两家着”，不仅出版社获取经济利益，老师也能为“职称”加分，他也曾收到不少出版社编写教科书的邀约。这导致一些老师“七抄八抄”拼凑出教材，教材质量良莠不齐。据了解，杨伟杰编写的这本《计算机应用基础》属于广东省教育厅推动的课题“高等学校大学计算机公共课程教学改革”项目。他在编写完教材后，还获得了学校颁发的教学优秀二等奖。不过他不愿意回答记者任何关于此本教材的问题。', '广州一教师因挑出教科书68处错误受处分 学校称其混淆是非']]

for i, j in data:
    lan1.add_sentence(i)
    lan2.add_sentence(j)

print(len(lan1), len(lan2))

lr = 0.001
hidden_size = 256

encoder = EncoderRNN(len(lan1), hidden_size).to(devices)
decoder = DecoderRNN(hidden_size, len(lan2)).to(devices)

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(params, lr=lr)
criterion = nn.NLLLoss()
loss = 0
turn = 200
print_every = 20
print_loss_total = 0
training_pairs = [pair2tensor(random.choice(data)) for pair in range(turn)]

for tur in range(turn):
    # print(tur)
    optimizer.zero_grad()
    loss = 0
    x, y = training_pairs[tur]
    input_length = x.size(0)
    target_length = y.size(0)
    h = encoder.initHidde()
    for i in range(input_length):
        h = encoder(x[i], h)
    decoder_input = torch.LongTensor([SOS_token]).to(devices)

    for i in range(target_length):
        decoder_output, h = decoder(decoder_input, h)
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss = loss + criterion(decoder_output, y[i])
        if decoder_input.item() == EOS_token:
            break
    print_loss_total = print_loss_total + loss.item() / target_length
    if (tur+1) % print_every == 0:
        print('loss:', print_loss_total/print_every)
        print_loss_total = 0

    loss.backward()
    optimizer.step()


def translate(s):
    t = [lan1(i) for i in s.split()]
    print(t)
    t.append(EOS_token)
    f = encoder.sample(t)
    # print(f)
    s = decoder.sample(f)
    print(s)
    r = [lan2.idx2word[i] for i in s]
    return '/'.join(r)

print(translate('I try'))