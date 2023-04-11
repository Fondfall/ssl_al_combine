"""
将data_lite这种excel改成csv格式，并且每行分别为标号、文本、标签
"""
import xlrd2
import pandas as pd
import opencc
import numpy as np
filter = [
    "\\n", "\\xa0", "~", ".", "。", "'", "\"", "?", "？", "!", "！",
    "(", ")", "（", "）", "…", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "+", "-",
    "^", "/", "ω", "`", "´", "＾", "﹡", "*", "、",  "∀", "↑", "=", "＞",
    "<", ">", "＜", "°", "】", "【", "[", "]", "ˆ", "๑", "•", ":", "：",
    ";", "；", "￣", "⊂", " ", "，，", "ㅍ", "_", "＝", "“", "“", "点击httpspinyincne查看表情",
    "◕", "ˇ", "#", "ﾟ", "Д"
    # "，", ",",
    ]


def extract_text(content):
    data = xlrd2.open_workbook(content, encoding_override='utf-8')
    table = data.sheets()[3]  # 选定表
    n_rows = table.nrows  # 获取行号
    text = []
    cc = opencc.OpenCC('t2s')  # 繁到简
    for i in range(1, n_rows):  # 第0行为表头
        tex = table.row_values(i)  # 循环输出excel表中每一行，即所有数据
        result = tex[1]  # 取出表中列数据
        result = cc.convert(result)
        text.append(result)
    for i in range(len(text)):
        text[i] = text[i].replace(",", "，")
        for x in range(2):
            for j in range(len(filter)):
                text[i] = text[i].replace(str(filter[j]), "")
    return text


def extract_label(content):
    data = xlrd2.open_workbook(content, encoding_override='utf-8')
    table = data.sheets()[3]  # 选定表
    n_rows = table.nrows  # 获取行号
    label = []
    for i in range(1, n_rows):  # 第0行为表头
        tex = table.row_values(i)  # 循环输出excel表中每一行，即所有数据
        result = tex[2]  # 取出表中列数据
        label.append(int(result))
    return label


# 对数据进行格式处理形成新的pd数据
def dataProcess(content):
    text = extract_text(content)
    label = extract_label(content)
    tf = pd.DataFrame({
        'text': text,
        'label': label,
    })
    return tf


def Process(before, after):
    tf = dataProcess(before)
    dataframe = pd.DataFrame(tf)
    dataframe.to_csv(after,  sep=',')
    return None


def spilt():
    content = "data_lite_big.xlsx"
    text = extract_text(content)
    label = extract_label(content)
    rng = np.random.RandomState(5)  # 随机数种子
    indices = np.arange(len(text))
    rng.shuffle(indices)  # 将索引打乱
    print(indices[0])
    train_set = indices[:2795]
    test_set = indices[2795:]
    train_text = np.array(text)[train_set]
    train_label = np.array(label)[train_set]
    test_text = np.array(text)[test_set]
    test_label = np.array(label)[test_set]
    tf_train = pd.DataFrame({
        'text': train_text,
        'label': train_label,
    })
    tf_test = pd.DataFrame({
        'text': test_text,
        'label': test_label,
    })
    dataframe = pd.DataFrame(tf_train)
    dataframe.to_csv('data_lite_big_train.csv', sep=',')
    dataframe_1 = pd.DataFrame(tf_test)
    dataframe_1.to_csv('data_lite_big_test.csv', sep=',')


if __name__ == '__main__':
    before = "data_cleaned.xlsx"
    after = 'data_cleaned_5000.csv'
    Process(before, after)
    # # tf = dataProcess(before)
    # # print(tf)
    # spilt()
