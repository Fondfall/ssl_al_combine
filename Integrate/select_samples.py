import numpy as np
import torch
import pandas as pd


def select(features, targets, k, category):
    """
    :param features: 输入样本的特征，x_i的特征为一维行向量
    :param targets: 样本的标签，与features一一对应
    :param k: KNN的参数值
    :param category: 类别，格式为[0,1,2,3]
    :return select_label_features, select_label_index_order, noise_label_features, noise_label_index_order:
    """
    # 假设已标记样本的标签的特征是一维的
    N = len(features)  # 用字母N表示输入的样本数
    similar = torch.zeros(N, N)  # 初始化相似度矩阵，大小为N*N
    '''
    step1 计算所有样本两两之间的相似度，保存在相似度矩阵中
    '''
    for i in range(N):
        x_i = features[i]
        for j in range(N):
            x_j = features[j]
            # 计算余弦相似度
            if i == j:
                similar[i, j] = -1  # 自身的相似度设为-1
            else:
                similar[i, j] = torch.dot(x_i, x_j)
    # print(similar)
    '''
    step2 利用KNN生成伪标签
    '''
    y_hat = torch.zeros(N)  # 初始化伪标签向量
    y = targets  # 输入的标签用字母y表示
    K_Neighbors = torch.zeros(N, k)  # 用来存放最相似k个样本的索引
    for row in range(N):
        # 对每行的数据按降序排列，value里保存值，index里保存该值在原始数据中的索引
        value, index = torch.sort(similar[row], descending=True)
        top_k_index = index[:k]  # 相似度最大的k个索引
        K_Neighbors[row] = top_k_index  # 存放在矩阵中
        y_k = y[top_k_index]  # top_k的标签存放在y_k中
        y_k_list = y_k.tolist()  # 将张量转为list，因为下面用到的count函数在张量里没有这个属性
        # 将最相似k个样本中出现最多的标签来创建伪标签
        y_hat[row] = max(y_k_list, key=y_k_list.count)
    # print(K_Neighbors)
    '''
    step3 用伪标签代替真实标签来近似后验概率
    '''
    class_num = len(category)  # 总共分为几类
    q = torch.zeros(class_num, N)  # 后验概率用字母q表示，q=P(y|x),q_ij代表x_j的情况下类别为i的概率
    for i in range(class_num):
        for j in range(N):
            for k_temp in range(k):
                y_hat_index = K_Neighbors[j, k_temp].long()  # tensor的索引必须是long,byte,bool等类型
                # print(y_hat_index)
                if y_hat[y_hat_index] == i:
                    q[i, j] = q[i, j] + 1
            q[i, j] = q[i, j] / k  # 归一化
    '''
    step4 用交叉熵来挑选样本
    '''
    select_num = torch.zeros(class_num)  # 每类挑选的置信样本数
    cross_entropy_vector = torch.zeros(N)  # 存放交叉熵
    for i in range(class_num):
        for j in range(N):
            if y_hat[j] == y[j] and y[j] == i:
                select_num[i] = select_num[i] + 1
    # 根据后验概率得到输入的预测标签，值代表概率，形式上就是上式q的转置
    y_p = q.t()  # 预测标签
    y_t = torch.zeros(N, class_num)  # 真实标签，即输入的标签，转为向量形式
    select_label_index = []
    noise_label_index = []
    for i in range(N):
        for j in range(class_num):
            if j == y[i]:
                y_t[i, j] = 1
    for i in range(N):
        cross_entropy_vector[i] = cross_entropy(y_t[i], y_p[i])  # 计算交叉熵
    for i in range(class_num):
        y_i_index = torch.nonzero(y == category[i]).squeeze()  # 输入数据中类别i对应输入的索引
        cross_entropy_i = cross_entropy_vector[y_i_index]  # 建立一个临时向量，值分别为第i类的各样本交叉熵
        _, index_temp = torch.sort(cross_entropy_i, descending=False)  # 将属第i类的交叉熵按从小到大排列
        min_entropy_index = index_temp[:select_num[i].long()].long()  # 交叉熵较小的样本索引，样本数为select_num[i]
        select_label_index = select_label_index + y_i_index[min_entropy_index].tolist()
        # min_entropy_index在y_i_index相应位置处即为所挑选的样本
        noise_label_index = noise_label_index + y_i_index[index_temp[select_num[i].long():]].long().tolist()
    select_label_index = torch.tensor(select_label_index).reshape(len(select_label_index))
    noise_label_index = torch.tensor(noise_label_index).reshape(len(noise_label_index))
    # print(select_label_index)
    # print(noise_label_index)
    select_label_index_order, _ = torch.sort(select_label_index, descending=False)  # 按从小到大排列
    noise_label_index_order, _ = torch.sort(noise_label_index, descending=False)  # 按从小到大排列
    select_label_features = features[select_label_index_order]
    select_label = targets[select_label_index_order]
    noise_label_features = features[noise_label_index_order]
    return select_label_features, select_label, select_label_index_order, noise_label_features, noise_label_index_order


def cross_entropy(y_true, y_pred):
    C = 0
    # one-hot encoding
    for col in range(y_true.shape[-1]):
        y_pred[col] = y_pred[col] if y_pred[col] < 1 else 0.99999
        y_pred[col] = y_pred[col] if y_pred[col] > 0 else 0.00001
        C += y_true[col] * np.log(y_pred[col]) + (1 - y_true[col]) * np.log(1 - y_pred[col])
    return -C


def selection_integrate(processed_data, k):
    data = processed_data.copy()
    x = data[:, : -1]  # 输入
    y = data[:, -1]  # 输出
    # 将numpy.ndarray类型的数据转为tensor类型
    features = torch.from_numpy(x)
    targets = torch.from_numpy(y)
    category = torch.tensor(range(4))
    select_features, select_label, select_index, _, noise_index = select(features, targets, k, category)
    noise_rate = len(noise_index) / len(features)
    print("k_num:", k, "len_noise:", len(noise_index),
          "len_select:", len(select_index), "samples_num:", len(processed_data))
    print("noise rate:", noise_rate)
    # select_data = data[select_index]
    # noise_data = data[noise_index]
    return select_index, noise_index


def main():  # 测试用
    data = np.loadtxt(open("features.csv", "rb"), delimiter=",", skiprows=0)
    rng = np.random.RandomState(2)  # 随机数种子
    indices = np.arange(len(data))
    rng.shuffle(indices)  # 将索引打乱
    x = data[:, : -1]  # 输入
    y = data[:, -1]  # 输出
    # 将numpy.ndarray类型的数据转为tensor类型
    features = torch.from_numpy(x)
    label = torch.from_numpy(y)
    k = 100
    category = torch.tensor(range(4))
    _, select_index, _, noise_index = select(features, label, k, category)
    print(noise_index)
    noise_rate = len(noise_index)/len(features)
    print("k_num:", k, "len_noise:", len(noise_index), "len_select:", len(select_index), "samples_num:", len(features))
    print("noise rate:", noise_rate)
    return None


if __name__ == '__main__':
    main()
