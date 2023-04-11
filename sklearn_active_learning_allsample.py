import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from select_samples import select


def standard_1(data_before):  # 归一化至-1 - 1
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    data = min_max_scaler.fit_transform(data_before)  # 归一化后的结果
    # print(data[0])
    return data


def standard_0(data_before):  # 归一化至0 - 1
    scaler = MinMaxScaler()
    scaler.fit(data_before)
    data = scaler.transform(data_before)
    # print(data[0])
    return data


def select_indices(features, label):
    features = torch.from_numpy(features)
    label = torch.from_numpy(label)
    _, sel, _, un_sel = select(features, label)
    sel = sel.tolist()
    un_sel = un_sel.tolist()

    rng = np.random.RandomState(5)  # 随机数种子
    rng.shuffle(sel)  # 将索引打乱

    if len(sel) >= 300:
        unlabeled_indices = sel[300:] + un_sel
        unlabeled_indices = [int(i) for i in unlabeled_indices]
        rng.shuffle(unlabeled_indices)  # 将索引打乱
    else:
        unlabeled_indices = [int(i) for i in un_sel]
        rng.shuffle(unlabeled_indices)  # 将索引打乱
    return unlabeled_indices


def parameter(X, y, y_train, unlabeled_set):
    num = 30
    gammas = np.logspace(-2, 1, num=num)  # 10^-2到10的等比数列
    score_in = 0
    for i in range(num):
        lp_model = LabelSpreading(gamma=gammas[i], max_iter=200, kernel='rbf')
        lp_model.fit(X, y_train)
        predicted_labels = lp_model.transduction_[unlabeled_set]
        true_labels = y[unlabeled_set]
        score = accuracy_score(true_labels, predicted_labels)
        # print(i, gammas[i], score)
        if score > score_in:
            score_in = score
            gamma_fin = round(gammas[i], 4)
            # gamma_fin = gammas[i]
    # print(gamma_fin)
    return gamma_fin


# data = np.loadtxt(open("w2v_big.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("w2v_big_low_dim.csv", "rb"), delimiter=",", skiprows=0)
data = np.loadtxt(open("w2v_all_mean.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("w2v_big_low_dim_mean.csv", "rb"), delimiter=",", skiprows=0)
X = data[:, : -1]  # 输入

# 特征归一化处理
X = standard_1(X)
y = data[:, -1]  # 标签
X_train = X[:3664]
y_train = y[:3664]
# print(y_train)
# print(len(y_train))

######################################
rng = np.random.RandomState(5)  # 随机数种子
indices = np.arange(len(y_train))
rng.shuffle(indices)  # 将索引打乱
n_total_samples = len(y_train)
n_labeled_points = 300
max_iterations = 500
unlabeled_indices = indices[n_labeled_points:]
######################################
# unlabeled_indices = select_indices(X, y)
# n_total_samples = len(y)
# n_labeled_points = 300
# max_iterations = 500
######################################
for i in range(max_iterations):
    # break
    if len(unlabeled_indices) == 0:
        print("No unlabeled items left to label.")
        break
    y1_train = np.copy(y_train)
    y1_train[unlabeled_indices] = -1

    # lp_model = LabelSpreading(gamma=0.09, max_iter=200)  # 768  big  0.09&0.77
    # lp_model = LabelSpreading(gamma=0.625, max_iter=200)  # 300  low_dim 0.91&0.81  0.625&0.82
    lp_model = LabelSpreading(gamma=0.24, max_iter=200)  # 768  big_mean 0.43&0.81  0.693&0.79
    # lp_model = LabelSpreading(gamma=1.6, max_iter=200)  # 300  low_dim_mean  1.6&0.8
    # lp_model = LabelSpreading(kernel='knn', n_neighbors=10, max_iter=100)
    #####################################################################
    # gamma = parameter(X, y, y_train, unlabeled_indices)
    # lp_model = LabelSpreading(gamma=gamma, max_iter=200)  # 自适应参数
    #####################################################################
    lp_model.fit(X_train, y1_train)

    predicted_labels = lp_model.transduction_[unlabeled_indices]
    true_labels = y_train[unlabeled_indices]

    cm = confusion_matrix(true_labels, predicted_labels,
                          labels=lp_model.classes_)

    print("Iteration %i %s" % (i, 70 * "_"))
    # print("Label Spreading model: %d labeled & %d unlabeled (%d total)  gamma=%a"
    #       % (n_labeled_points, n_total_samples - n_labeled_points,
    #          n_total_samples, gamma))
    print("Label Spreading model: %d labeled & %d unlabeled (%d total)"
          % (n_labeled_points, n_total_samples - n_labeled_points,
             n_total_samples))

    print(classification_report(true_labels, predicted_labels, digits=4))

    print("Confusion matrix")
    print(cm)

    # 计算转导标签分布的熵
    pred_entropies = stats.distributions.entropy(
        lp_model.label_distributions_.T)

    # 选择分类器最不确定的最多5位数字示例
    uncertainty_index = np.argsort(pred_entropies)[::-1]
    uncertainty_index = uncertainty_index[
        np.in1d(uncertainty_index, unlabeled_indices)][:5]

    # 跟踪我们获得标签的索引
    delete_indices = np.array([], dtype=int)

    for index, image_index in enumerate(uncertainty_index):
        # 标记5点，远离标记集
        delete_index, = np.where(unlabeled_indices == image_index)
        delete_indices = np.concatenate((delete_indices, delete_index))

    unlabeled_indices = np.delete(unlabeled_indices, delete_indices)
    n_labeled_points += len(uncertainty_index)
final_labeled = np.setdiff1d(indices, unlabeled_indices)
lp_model = LabelSpreading(gamma=0.24, max_iter=200)
y2_train = np.copy(y)
y2_train[unlabeled_indices] = -1
lp_model.fit(X, y2_train)
y_p = lp_model.transduction_[3664:]
print(y_p,len(y_p))
print(y_train,len(y_train))
y_last = np.hstack((y_train, y_p))  # 左右合并
tf = pd.DataFrame(y_last)
tf.to_excel("predict.xlsx")
