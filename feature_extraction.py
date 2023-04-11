import jieba  # 导入分词模块
import pandas as pd  # 导入Pandas模块
import openpyxl
import os

# 避免工作台输出显示不完全，输出太长有省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 10000)
# 表头
text = "内容"
label = "分类后"
# 1. 导入数据
comments_df = pd.read_excel(open('data_cleaned.xlsx', 'rb'), sheet_name='Sheet2')  # 读入数据
print('数据的维度是：', comments_df.shape)
print("导入数据的前五行：")
print(comments_df.head())  # 查看数据的前5行


# 2. 清洗数据, 删除空的数据
def clean_sents(txt):
    txt = str(txt) if txt is not None else ""
    if len(txt) == 0:
        return None
    else:
        return txt


stopwords_file = "stopwords.txt"
with open('stopwords.txt', "r", encoding="utf8") as f:
    stopwords_list = [word.strip() for word in f.read()]


def filter_stopwords(txt):
    """过滤停用词"""
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


comments_df[text] = comments_df[text].apply(clean_sents)
comments_df2 = comments_df[comments_df[text] != "nan"]
len(comments_df2)
# 运行结果
# 58117
comments_df2[text] = comments_df2[text].apply(filter_stopwords)
print(comments_df2.head())

# 3. 切分训练集和验证集和测试集
from sklearn.model_selection import train_test_split

X, y = comments_df2[text], comments_df2[label]


# 4. 统计词频
from nltk import FreqDist

# 把所有词和对应的词频放在一个list里
all_words = []

for comment in comments_df2[text]:
    all_words.extend(comment)

len(all_words)

fdisk = FreqDist(all_words)

TOP_COMMON_WORDS = 200

most_common_words = fdisk.most_common(TOP_COMMON_WORDS)

print(most_common_words[:10])

# 5. 提取特征
import numpy as np
from nltk.text import TextCollection

tfidf_generator = TextCollection(X.values.tolist())


def extract_tfidf(texts, targets, text_collection, common_words):
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
        if i % 5000 == 0:
            print("已经完成{}个样本的特征提取.".format(i))
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


cleaned_X, cleaned_y = extract_tfidf(X, y, tfidf_generator, dict(most_common_words).keys())
features = np.concatenate((cleaned_X, cleaned_y.reshape(len(cleaned_y), 1)), axis=1)
print(len(features))
df_features = pd.DataFrame(features)
df_features.to_csv("features_cleaned_2000.csv", header=None, index=None)


