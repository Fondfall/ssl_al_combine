import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report, accuracy_score

# data = np.loadtxt(open("features.csv", "rb"), delimiter=",", skiprows=0)
# rng = np.random.RandomState(5)  # 随机数种子
# indices = np.arange(len(data))
# rng.shuffle(indices)  # 将索引打乱


# data = np.loadtxt(open("w2v_big.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("d2v_cleaned_2000.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("d2v.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("w2v_all_mean.csv", "rb"), delimiter=",", skiprows=0)
data = np.loadtxt(open("features_cleaned_2000.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("w2v_cleaned_mean_shuffle_2000.csv", "rb"), delimiter=",", skiprows=0)
data = data[:3664]
X = data[:, : -1]  # 输入
y = data[:, -1]  # 输出
rng = np.random.RandomState(9)  # 随机数种子
indices = np.arange(len(data))
rng.shuffle(indices)  # 将索引打乱
# print(X[0])
# images = digits.images[indices[:340]]  # 这个不知道干啥用的

n_total_samples = len(y)
# n_labeled_points = 2795  # 已标记/样本数
n_labeled_points = 1600  # 已标记样本数1260 1440

unlabeled_set = indices[n_labeled_points:]

y_train = np.copy(y)
y_train[unlabeled_set] = -1  # 未标记样本集y设为-1

num=100
accuracy = []
gammas = np.logspace(-2, 3, num=num)
# gammas = range(100)
score_in = 0
for i in range(num):
    lp_model = LabelSpreading(gamma=gammas[i], max_iter=30, kernel='rbf')
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[unlabeled_set]
    true_labels = y[unlabeled_set]
    score = accuracy_score(true_labels, predicted_labels)
    # if score > score_in:
    #     gamma_fin = gammas[i]
    accuracy.append(score)
    print(i,gammas[i],score)
    print(classification_report(true_labels, predicted_labels, digits=4))

# lp_model = LabelSpreading(gamma=0.24, max_iter=200, kernel='rbf')# bert2000
lp_model = LabelSpreading(gamma=1.9, max_iter=200, kernel='rbf')# tf-idf2000
# lp_model = LabelSpreading(kernel='knn', n_neighbors=9, max_iter=100)
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]
print(classification_report(true_labels, predicted_labels, digits=4))



# print(
#     "Label Spreading model: %d labeled & %d unlabeled points (%d total)"
#     % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
# )
# print(classification_report(true_labels, predicted_labels))
# print(true_labels)
# print(predicted_labels)

