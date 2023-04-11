import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# data = np.loadtxt(open("w2v_big.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("d2v_2.csv", "rb"), delimiter=",", skiprows=0)
# data = np.loadtxt(open("w2v_cleaned_mean_shuffle_2000.csv", "rb"), delimiter=",", skiprows=0)
data = np.loadtxt(open("features_cleaned_2000.csv", "rb"), delimiter=",", skiprows=0)
X = data[:, : -1]  # 输入
y = data[:, -1]  # 输出
rng = np.random.RandomState(3)  # 随机数种子
indices = np.arange(len(data))
rng.shuffle(indices)  # 将索引打乱
# print(X[0])

n_total_samples = len(y)
n_labeled_points = 1600  # 已标记样本数

unlabeled_set = indices[n_labeled_points:]
labeled_set = indices[:n_labeled_points]
X_train = X[labeled_set]
y_train = y[labeled_set]
X_test = X[unlabeled_set]
y_true = y[unlabeled_set]

model = RandomForestClassifier(n_estimators=20)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
print(classification_report(y_true, y_predict, digits=4))

# 设置弱学习器数量为10
# for i in range(1,100):
#     model = RandomForestClassifier(n_estimators=i)
#     model.fit(X_train, y_train)
#     y_predict = model.predict(X_test)
#     print(i, classification_report(y_true, y_predict, digits=4))
