import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report

data = np.loadtxt(open("features.csv", "rb"), delimiter=",", skiprows=0)
# rng = np.random.RandomState(5)  # 随机数种子
# indices = np.arange(len(data))
# rng.shuffle(indices)  # 将索引打乱

# X = data[:, : -1]  # 输入
y = data[:, -1]  # 输出
data = np.loadtxt(open("w2v.csv", "rb"), delimiter=",", skiprows=0)
X = data  # 输入
print(X[0])
# images = digits.images[indices[:340]]  # 这个不知道干啥用的

n_total_samples = len(y)
n_labeled_points = 300  # 已标记样本数

indices = np.arange(n_total_samples)  # 新的索引

unlabeled_set = indices[n_labeled_points:]

y_train = np.copy(y)
y_train[unlabeled_set] = -1  # 未标记样本集y设为-1


lp_model = LabelSpreading(gamma=0.3, max_iter=200, kernel='rbf')
lp_model.fit(X, y_train)
predicted_labels = lp_model.transduction_[unlabeled_set]
true_labels = y[unlabeled_set]

print(
    "Label Spreading model: %d labeled & %d unlabeled points (%d total)"
    % (n_labeled_points, n_total_samples - n_labeled_points, n_total_samples)
)
print(classification_report(true_labels, predicted_labels))
print(true_labels)
print(predicted_labels)

