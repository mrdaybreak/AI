










# import re
#
# print(re.match('www', 'www.baidu.com').span())
# print(re.search('www', 'www.baidu.com').span())
# print(re.search('com', 'www.baidu.com').span())
#
# phone = "001-609-7267#"
# num = re.sub('#.*$', '', phone)
# print(num)
# num2 = re.sub('\D', '', phone)
# print(num2)
#
# def find_phone(phone):
#     a = re.compile('^1[35678]\d{9}')
#     b = a.match(phone)
#     print(b.group())
# find_phone('13560357290$')
#
# m = re.match('^(\d{3})-(\d{3,8})$', '010-12345')
# print(m.group(2))

# import pandas as pd
# import re
# from gensim import corpora
# import gensim
# import jieba
#
# def clean_email_text(text):
#     text = text.replace('\n', " ")
#     text = re.sub("-", " ", text)
#     text = re.sub("\d+/\d+/\d+", "", text)
#     text = re.sub("[0-2]?[0-9]:[0-6][0-9]", "", text)
#     text = re.sub("[\w]+@[\.\w]+", "", text)
#     text = re.sub("/[a-zA-Z]*[:\//\]*[A-Za-z0-9\-_]+\.+[A-Za-z0-9\.\/%&=\?\-_]+/i", "", text)
#     pure_text = ''
#     # 以防还有其他特殊字符（数字）等等，我们直接把他们loop一遍，过滤掉
#     for letter in text:
#         # 只留下字母和空格
#         if letter.isalpha() or letter == ' ':
#             pure_text += letter
#     text = ' '.join(word for word in pure_text.split() if len(word) > 1)
#     return text
#
# df = pd.read_csv("Emails.csv")
# df = df[['Id', 'ExtractedBodyText']].dropna()
# doc = df['ExtractedBodyText']
# docs = doc.apply(lambda s:clean_email_text(s))
# # print(docs)
#
# doclist = docs.values
#
# texts = [[word for word in jieba.cut(doc)] for doc in doclist]
# # print(texts)
#
# dictionary = corpora.Dictionary(texts)
# # print(dictionary)
# corpus = [dictionary.doc2bow(text) for text in texts]
# # print(corpus)
# lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)
# print(lda.print_topics(num_topics=10, num_words=10))
# for topic in lda.print_topics(num_words=5):
#     print(topic)
# print(corpus[0])
# print(lda.get_document_topics(corpus[0]))
# print(lda.get_document_topics(corpus[1]))
# print(lda.get_document_topics(corpus[2]))

# from snownlp import SnowNLP
# from textrank4zh import TextRank4Keyword, TextRank4Sentence

# text = '王者荣耀典韦连招是使用一技能+大招+晕眩+二技能+普攻，这套连招主要用于先手强开团，当发现对面走位失误或撤退不及时，我们就可以利用一技能的加速，配合大招减速留住对手，协同队友完成击杀。\
# 当对方站位较集中时，我们同样可以利用“一技能+大招+晕眩”进行团控和吸收伤害。\
# 在吸收伤害的同时我们还可以利二技能打出不错的输出。这套连招重要的是把握时机，要有一夫当关，万夫莫开之势。\
# 缺点是一技能的强化普攻和解除控制的效果会被浪费。\
# 连招二：大招+晕眩+二技能+普攻+一技能+普攻。\
# 这套连招用于偷袭对手后排很是好用，利用草丛埋伏。\
# 大招跳到对面身上。迅速晕眩对手，接着二技能继续减速对手，二技能命中后会提升典韦到极限攻速，这时不断普攻，接下来一般会遇到两种情况，当对手继续逃跑时，我们利用一技能加速追击对手，强化普攻击杀对手。\
# 当对手用技能控住我们我们可以利用一技能解除控制，追击并完成击杀。'

# snow = SnowNLP(text)
# print(snow.keywords(20))
# print(snow.summary(3))
# print(snow.sentiments)

# tr4w = TextRank4Keyword()
# tr4w.analyze(text=text, lower=TextRank4Keyword, window=3)
# for item in tr4w.get_keywords(20, word_min_len=2):
#     print(item.word, item.weight)
# tr4s = TextRank4Sentence()
# tr4s.analyze(text=text, lower=True, source='all_filters')
# for item in tr4s.get_key_sentences(num=3):
#     print(item.index, item.weight, item.sentence)

import jieba
import jieba.analyse
import jieba.posseg as pseg
# seg_list = jieba.cut(text, cut_all=False)
# print(seg_list)
# words = pseg.cut(text)
# for word, flag in words:
#     print(word, flag)
# keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=('n', 'nr', 'ns'))
# keywords2 = jieba.analyse.textrank(text, topK=20, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
# for item in keywords:
#     print(item)
# for item in keywords2:
#     print(item)

import pandas as pd
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
pd.options.display.max_columns = 30
import matplotlib.pyplot as plt
import re

plt.rcParams['font.sans-serif'] = ['SimHei']
df = pd.read_csv('Seattle_Hotels.csv', encoding='latin-1')
print(df)

def print_description(index):
    example = df[df.index == index][['desc', 'name']].values[0]
    if len(example) > 0:
        print(example)

def get_top_n_words(corpus, n=1, k=None):
    vec = CountVectorizer(ngram_range=(n, n), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    print(vec.get_feature_names())
    print(bag_of_words.toarray())
    sum_words = bag_of_words.sum(axis=0)
    words_freg = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freg, key=lambda x: x[1], reverse=True)
    return words_freg[:k]
common_words = get_top_n_words(df['desc'], 3, 20)
print(common_words)
df1 = pd.DataFrame(common_words, columns = ['desc' , 'count'])
df1.groupby('desc').sum()['count'].sort_values().plot(kind='barh', title='去掉停用词后，酒店描述中的Top20单词')
plt.show()

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
# 对文本进行清洗
def clean_text(text):
    # 全部小写
    text = text.lower()
    # 用空格替代一些特殊符号，如标点
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    # 移除BAD_SYMBOLS_RE
    text = BAD_SYMBOLS_RE.sub('', text)
    # 从文本中去掉停用词
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)
    return text
# 对desc字段进行清理
df['desc_clean'] = df['desc'].apply(clean_text)
print(df['desc_clean'])

df.set_index('name', inplace=True)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0.01, stop_words='english')
tfidf_matrix = tf.fit_transform(df['desc_clean'])
print(tfidf_matrix)

cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)
print(cosine_similarities)
indices = pd.Series(df.index)

def recomendations(name, cosine_similarities=cosine_similarities):
    recommend_hotels = []
    idx = indices[indices==name].index[0]
    print(idx)
    score_series = pd.Series(cosine_similarities[idx]).sort_values()
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending=False)
    # 取相似度最大的前10个（除了自己以外）
    top_10_indexes = list(score_series.iloc[1:11].index)
    # 放到推荐列表中
    for i in top_10_indexes:
        recommend_hotels.append(list(df.index)[i])
    return recommend_hotels


print(recomendations('Hilton Seattle Airport & Conference Center'))
print(recomendations('The Bacon Mansion Bed and Breakfast'))
# print(result)



