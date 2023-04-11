import jieba  # 导入分词模块
import pandas as pd  # 导入Pandas模块
from nltk import FreqDist
import numpy as np
from nltk.text import TextCollection
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import openpyxl
import os


def import_data():  # 导入数据
    comments_df = pd.read_excel('data2.xlsx')  # 读入数据
    print(50*"-")
    print("数据载入成功")
    print('数据的维度是：', comments_df.shape)
    len(comments_df)
    print("数据前5行：", comments_df.head())  # 查看数据的前5行
    print(50 * "-")
    comments_df["主题+内容"] = comments_df["主题+内容"].apply(clean_data)
    comments_df2 = comments_df[comments_df["主题+内容"] != "nan"]
    comments_df2["主题+内容"] = comments_df["主题+内容"].apply(filter_stopwords)  # 过滤停用词后的文本
    # print(comments_df2["主题+内容"].head())
    return comments_df, comments_df2  # comments_df未经处理，comments_df2经过数据清洗与词过滤


def clean_data(txt):  # 清洗数据, 删除空的数据
    txt = str(txt) if txt is not None else ""
    if len(txt) == 0:
        return None
    else:
        return txt


def filter_stopwords(txt):  # 过滤停用词
    with open('stopwords.txt', "r", encoding="utf8") as f:
        stopwords_list = [word.strip() for word in f.read()]
    sent = jieba.lcut(txt)
    words = []
    for word in sent:
        if word in stopwords_list:
            continue
        else:
            if word == ' ':
                continue
            else:
                words.append(word)
    return words


def word_frequency(comments_df2):  # 统计词频
    # 把所有词和对应的词频放在一个list里
    all_words = []
    for comment in comments_df2["主题+内容"]:
        all_words.extend(comment)
    len(all_words)
    fdisk = FreqDist(all_words)
    TOP_COMMON_WORDS = 1000
    most_common_words = fdisk.most_common(TOP_COMMON_WORDS)
    print(most_common_words[:10])  # 词频最高的10个词
    return most_common_words


def extract_tfidf(texts, targets, text_collection, common_words):  # 提取特征
    """
    提取文本的tf-idf.
        texts: 输入的文本.
        targets: 对应的评价.
        text_collection: 预先初始化的TextCollection.
        common_words: 输入的前N个词作为特征进行计算.
    """
    # 得到行向量的维度
    n_sample = len(texts)
    # 得到列向量的维度
    n_feat = len(common_words)
    # 初始化X矩阵, X为最后要输出的TF-IDF矩阵
    X = np.zeros([n_sample, n_feat])
    y = np.zeros(n_sample)
    for i, text in enumerate(texts):
        # if i % 5000 == 0:
        #     print("已经完成{}个样本的特征提取.".format(i))
        # 每一行对应一个文档, 计算这个文档中的词的tf-idf, 没出现的词则为0
        feature_vector = []
        for word in common_words:
            if word in text:
                tf_idf = text_collection.tf_idf(word, text)
            else:
                tf_idf = 0.0
            feature_vector.append(tf_idf)
        X[i, :] = np.array(feature_vector)
        y[i] = targets.iloc[i]
    return X, y


def extraction_integrate():
    comments_df, comments_df2 = import_data()  # comments_df未经处理，comments_df2经过数据清洗与词过滤
    X, y = comments_df2["主题+内容"], comments_df2["类别编号"]  # X处理文本, y类别
    most_common_words = word_frequency(comments_df2)
    tfidf_generator = TextCollection(X.values.tolist())
    cleaned_X, cleaned_y = extract_tfidf(X, y, tfidf_generator, dict(most_common_words).keys())
    # 将特征与输出合并
    processed_data = np.concatenate((cleaned_X, cleaned_y.reshape(len(cleaned_y), 1)), axis=1)
    # 将features保存到csv文件中
    df_features = pd.DataFrame(processed_data)
    df_features.to_csv("features.csv", header=None, index=None)
    return processed_data


def data_fitting(a):  # one-hot 输入数据格式调整
    b = []
    for i in range(len(a)):
        temp = ""
        for j in range(len(a[i])):
            if len(a[i]) - j == 1:
                space = ""
            else:
                space = " "
            temp = temp + a[i][j] + " "
        b.append(temp)
    return b


def one_hot():
    comments_df, comments_df2 = import_data()  # comments_df未经处理，comments_df2经过数据清洗与词过滤
    X, y = comments_df2["主题+内容"], comments_df2["类别编号"]  # X处理文本, y类别
    X = data_fitting(X)
    one_hot_vectorizer = CountVectorizer(binary=True)
    features = one_hot_vectorizer.fit_transform(X).toarray()
    y = np.array(y)
    processed_data = np.concatenate((features, y.reshape(len(y), 1)), axis=1)
    # 将features保存到csv文件中
    df_features = pd.DataFrame(processed_data)
    df_features.to_csv("features_onehot.csv", header=None, index=None)
    return processed_data


if __name__ == "__main__":
    one_hot()
    # corpus = [
    #     '这是 第一个 文档',
    #     '这是 第二个 文档',
    #     '这是 最后 一个 文档'
    # ]
    # one_hot_vectorizer = CountVectorizer(binary=True)
    # one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
    # print(one_hot)
    # # sns.heatmap(one_hot, annot=True, char=False, yticklabels=['Sentence 1', 'Sentence 2'])
    # # plt.show()
