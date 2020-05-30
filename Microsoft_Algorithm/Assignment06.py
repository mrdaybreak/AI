import numpy as np
import pandas as pd
import jieba
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import editdistance

# 读取停用词和新闻内容
with open('chinese_stopwords.txt', 'r', encoding='utf-8') as file:
    stopwords = [i[:-1] for i in file.readlines()]
news = pd.read_csv('sqlResult.csv', encoding='gb18030').dropna(subset=['content'])

# print(news[news.content.isna()])


def split_text(text):
    # text = text.replace(' ', '').replace('\n', '')
    # print(text)
    text2 = jieba.cut(text.strip())
    result = '/'.join([w for w in text2 if w not in stopwords])
    return result

# print(news.iloc[0].content)
# print(split_text(news.iloc[0].content))


if not os.path.exists('corpus.pkl'):
    # 内容分词集合模型
    corpus = list(map(split_text, [str(i) for i in news.content]))
    with open('corpus.pkl', 'wb') as file:
        pickle.dump(corpus, file)
else:
    with open('corpus.pkl', 'rb') as file:
        corpus = pickle.load(file)

# CountVectorizer是属于常见的特征数值计算类，是一个文本特征提取方法
countvectorizer = CountVectorizer(encoding='gb18030', min_df=0.015)
# 用 CountVectorizer 类向量化之后再调用 TfidfTransformer 类进行预处理
tfidftransformer = TfidfTransformer()
# CountVectorizer会将文本中的词语转换为词频矩阵，它通过fit_transform函数计算各个词语出现的次数
countvector = countvectorizer.fit_transform(corpus)
# TF-IDF倾向于过滤掉常见的词语，保留重要的词语
tfidf = tfidftransformer.fit_transform(countvector)
# 创建一个新华标签为1，其他为0的list
label = list(map(lambda source: 1 if '新华' in str(source) else 0, news.source))
# print('tfidf:', tfidf.toarray())
# print('label:', label)
# 切分训练集，训练集x是词频array，y是label
x_train, x_test, y_train, y_test = train_test_split(tfidf.toarray(), label, test_size=0.3, random_state=33)
# 贝叶斯多样式分类
clf = MultinomialNB()
clf.fit(x_train, y_train)
y_predict = clf.predict(tfidf.toarray())
labels = np.array(label)
# 创建一个预测值跟真实值的dataframe
compare_news_index = pd.DataFrame({'prediction':y_predict, 'labels':labels})
# print(compare_news_index)
# 生成一个预测是新华社但label不是新华社的的列表
copy_news_index = compare_news_index[(compare_news_index['prediction'] == 1) & (compare_news_index['labels'] == 0)].index
# print(copy_news_index)
xinhuashe_news_index = compare_news_index[compare_news_index['labels'] == 1].index
print('可疑文章数: ', len(copy_news_index))

# 预处理
# normalizer = Normalizer()
# print('tfidf',tfidf.toarray())
# scaled_array = normalizer.fit_transform(tfidf.toarray())
# print('预处理',scaled_array)

if not os.path.exists('label.pkl'):
    # KMeans预测数据模型
    kmeans = KMeans(n_clusters=25)
    k_labels = kmeans.fit_predict(tfidf.toarray())
    print('k_labels: ', k_labels)
    with open('label.pkl', 'wb') as file:
        pickle.dump(k_labels, file)
else:
    with open('label.pkl', 'rb') as file:
        k_labels = pickle.load(file)
        print('k_labels: ', k_labels)

if not os.path.exists('id_class.pkl'):
    # 创建k_labels字典
    id_class = {index:class_ for index, class_ in enumerate(k_labels)}
    with open('id_class.pkl', 'wb') as file:
        pickle.dump(id_class, file)
        print('id_calss:', id_class)
else:
    with open('id_class.pkl', 'rb') as file:
        id_class = pickle.load(file)


if not os.path.exists('class_id.pkl'):
    # 创建labels里面加确定是新华社的index的字典
    class_id = defaultdict(set)
    for index, class_ in id_class.items():
        if index in xinhuashe_news_index.tolist():
            class_id[class_].add(index)
    with open('class_id.pkl', 'wb') as file:
        pickle.dump(class_id, file)
else:
    with open('class_id.pkl', 'rb') as file:
        class_id = pickle.load(file)
        print('class_id', class_id)


def find_similar_text(cpindex, top=10):
    dist_dict = {i:cosine_similarity(tfidf[cpindex], tfidf[i]) for i in class_id[id_class[cpindex]]}
    return sorted(dist_dict.items(), key=lambda x:x[1][0], reverse=True)

cpindex = 3352
similar_list = find_similar_text(cpindex)
print(similar_list)
print('怀疑抄袭：\n', news.iloc[cpindex].content)
similar2 = similar_list[0][0]
print('相似原文\n', news.iloc[similar2].content)
print('编辑距离', editdistance.eval(corpus[cpindex], corpus[similar2]))









