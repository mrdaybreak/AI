# coding:utf-8

'Thinking1 常用的文本分类方法都有哪些'
print('English--NLTK, Chinese--jieba')

'Thinking2 RNN为什么会出现梯度消失'
print('因为tanh函数会随着时间的加大无限接近于0.')


import keras
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import torch.utils.data as Data

def read_txt(txt_dir):
    with open(txt_dir, 'r', encoding='utf-8') as f:
        words = [a.strip() for a in f.readlines()]
        words_id = dict(zip(words, range(len(words))))
    print(words_id)
    return words, words_id


def read_category():
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
    cate_id = dict(zip(categories, range(len(categories))))
    print(cate_id)
    return categories, cate_id

categories, cate_id = read_category()
# print(cate_id)
words, words_id = read_txt('cnews.vocab.txt')
# print(len(words))
# print(words_id)


def process_file(filename, words_id, cate_id, maxlength=600):
    contents, labels = [], []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            # print(line.strip().split('\t')) # \t空格
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(content)
                    labels.append(label)
            except:
                pass
    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([words_id[x] for x in contents[i] if x in words_id])
        label_id.append(cate_id[labels[i]])
    # print(len(data_id))
    # print('---')
    # print(label_id)
    x_pad = keras.preprocessing.sequence.pad_sequences(data_id, maxlength)
    y_pad = keras.utils.to_categorical(label_id, num_classes=len(cate_id))
    # print(x_pad)
    # print(y_pad)
    return x_pad, y_pad

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(5000, 64)  # 5000个64维的词嵌入
        self.rnn = nn.GRU(input_size=64, hidden_size=128, num_layers=2, bidirectional=True) # 特征64，隐藏层神经元自己定义，叠加层数自己定，双向RNN
        self.f1 = nn.Sequential(nn.Linear(256, 128),
                                nn.Dropout(0.5),
                                nn.ReLU())
        self.f2 = nn.Sequential(nn.Linear(128, 10),
                                nn.Softmax(dim=1))
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.rnn(x)
        # print(x)
        x = F.dropout(x, p=0.5)
        # print(x[:, -1, :])
        x = self.f1(x[:, -1, :]) # ??
        return self.f2(x)


class RnnModel:
    def __init__(self):
        self.categories, self.cate_id = categories, cate_id
        self.words, self.words_id = words, words_id
        self.model = TextRNN()
        self.model.load_state_dict(torch.load('model_params.pkl'))

    def predict(self, message):
        content = message
        data = [self.words_id[x] for x in content if x in self.words_id]
        data = keras.preprocessing.sequence.pad_sequences([data], 600)
        data = torch.LongTensor(data)
        y_pred_cls = self.model(data)
        class_index = torch.argmax(y_pred_cls[0]).item()
        return self.categories[class_index]


def train():
    model = TextRNN() # TextRNN().cuda()
    Loss = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_acc = 0

    for epoch in range(1000):
        print(epoch)
        for step, (x_batch, y_batch) in enumerate(train_loader):
            x = x_batch # x_batch.cuda()
            y = y_batch # y_batch.cuda()
            out = model(x)
            loss = Loss(out, y)
            print('loss=', loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            accury = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
            print('acc=', accury)

            if (epoch+1)%5 == 0:
                for step, (x_batch, y_batch) in enumerate(val_loader):
                    x = x_batch
                    y = y_batch
                    out = model(x)
                    accury = np.mean((torch.argmax(out, 1) == torch.argmax(y, 1)).cpu().numpy())
                    print('acc=', accury)

                    if accury > best_val_acc:
                        torch.save(model.state_dict(), 'model_params.pkl')
                        best_val_acc = accury
                        print(accury)

x_train, y_train = process_file('cnews.val.txt', words_id, cate_id)
x_val, y_val = process_file('cnews.test.txt', words_id, cate_id)
x_train, y_train, x_val, y_val = torch.LongTensor(x_train), torch.Tensor(y_train), torch.LongTensor(x_val), torch.Tensor(y_val)
train_data = Data.TensorDataset(x_train, y_train)
train_loader = Data.DataLoader(dataset=train_data, batch_size=1000, shuffle=True, num_workers=2)
val_data = Data.TensorDataset(x_val, y_val)
val_loader = Data.DataLoader(dataset=val_data, batch_size=1000, shuffle=True, num_workers=2)

if __name__ == '__main__':
    train()
    model = RnnModel()
    test_demo = ['《时光重返四十二难》恶搞唐增取经一款时下最热门的动画人物：猪猪侠，加上创新的故事背景，震撼的操作快感，成就了这部恶搞新作，现正恶搞上市，玩家们抢先赶快体验快感吧。游戏简介：被时光隧道传送到208年的猪猪侠，必须经历六七四十二难的考验，才能借助柯伊诺尔大钻石的力量，开启时光隧道，重返2008年。在迷糊老师、菲菲公主的帮助下，猪猪侠接受了挑战，开始了这段充满了关心和情谊的旅程。    更多精彩震撼感觉，立即下载该款游戏尽情体验吧。玩家交流才是王道，讯易游戏玩家交流中心 QQ群：6306852-----------------生活要有激情，游戏要玩多彩(多彩游戏)。Colourfulgame (多彩游戏)，让你看看快乐游戏的颜色！精品推荐：1：《钟馗传》大战无头关羽，悲壮的剧情伴随各朝英灵反攻地府！2：《中华群英》将和赵云，项羽，岳飞等猛将作战，穿越各朝代抗击日寇。良品推荐：1：《赌王争霸之斗地主》易飞会在四角恋中会选择谁？是否最终成赌神呢？2：勇者后裔和魔王紧缠一起，前代恩怨《圣火伏魔录》将为您揭示一切。  3：颠覆传统概念，恶搞+非主流？！誓必弄死搞残为止《爆笑飞行棋》。4：《中国象棋残局大师》快棋和人机模式让畅快对弈！一切“多彩游戏”资讯，点击Colourfulgame官网http://www.colourfulgame.com一切“多彩游戏”感言，交流Colourfulgame论坛http://121.33.203.124/forum/【客服邮箱】：xunyiwangluo@126.com">xunyiwangluo@126.com">xunyiwangluo@126.com【客服热线】：020-87588437']
    for i in test_demo:
        print(model.predict(i))
