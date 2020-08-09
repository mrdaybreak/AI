# encoding:utf-8
from gensim.models import fasttext
import pandas as pd
import jieba

data = pd.read_excel('zhyuliao.xls', encoding='gb18030')
sentences = data.values.tolist()
segment_sen = []
for i in sentences:
    segment_sen.append(jieba.lcut(i[0]))

model1 = fasttext.FastText(segment_sen, min_count=1, iter=20)
model1.save('fast_text.model')
model1_read = fasttext.FastText.load('fast_text.model')
word2 = model1_read.wv.most_similar('太阳')
print(word2)