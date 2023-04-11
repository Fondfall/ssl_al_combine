import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import classification_report


def classification(features, targets, labeled_index, test_index, unlabeled_index):
    X = features  # 输入
    y = targets  # 输出
    n_total_samples = len(y)
    unlabeled_set = np.concatenate((test_index, unlabeled_index), axis=0)
    y_train = np.copy(y)
    y_train[unlabeled_set] = -1  # 未标记样本集y设为-1
    labeled_points = len(labeled_index)
    lp_model = LabelSpreading(gamma=0.25, max_iter=100)
    lp_model.fit(X, y_train)
    predicted_labels = lp_model.transduction_[test_index]  # 标签预测
    true_labels = y[test_index]  # 测试集

    print(
        "Label Spreading model: %d labeled & %d unlabeled points (%d total)"
        % (labeled_points, n_total_samples - labeled_points, n_total_samples)
    )
    print(classification_report(true_labels, predicted_labels))
    # print(predicted_labels)
    return None
