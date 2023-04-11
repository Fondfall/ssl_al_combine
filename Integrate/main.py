import numpy as np

from feature_extraction import extraction_integrate
from feature_extraction import one_hot
from select_samples import selection_integrate
from label_spreading import classification


if __name__ == "__main__":
    # k = 150  # KNN参数
    # labeled_points = 3000  # 已标记的训练样本数
    # test_points = 500  # 测试集数量
    k = 20  # KNN参数
    labeled_points = 300  # 已标记的训练样本数
    test_points = 100  # 测试集数量
    processed_data = extraction_integrate()  # 提取tf-idf特征
    # processed_data = one_hot()  # 提取one-hot特征
    labeled = processed_data[: labeled_points + test_points]  # 已标记样本
    unlabeled = processed_data[labeled_points + test_points:]  # 未标记样本
    # X = features[:, : -1]  # 输入
    # y = features[:, -1]  # 输出
    rng = np.random.RandomState(np.random.randint(0, 99))  # 随机数种子
    indices = np.arange(len(labeled))
    rng.shuffle(indices)  # 将索引打乱
    data = processed_data.copy()
    data[: len(labeled)] = labeled[indices]
    train = labeled[indices[: labeled_points]]  # 训练样本
    test = labeled[indices[labeled_points + 1:]]  # 测试样本
    # X_train = X[indices[: labeled_points, :]]
    # y_train = y[indices[: labeled_points, :]]
    # X_test = X[indices[labeled_points + 1:, :]]
    # y_test = y[indices[labeled_points + 1:, :]]
    select_index, noise_index = selection_integrate(train, k)  # 挑选样本
    features = data[:, : -1]
    targets = data[:, -1]
    index = np.arange(len(features))
    labeled_index = index[: labeled_points]  # 已标记的训练集
    test_index = indices[labeled_points:]
    unlabeled_index = index[labeled_points + test_points:]
    # 未筛选
    classification(features, targets, labeled_index, test_index, unlabeled_index)
    # 已筛选
    labeled_index_new = select_index
    unlabeled_index_new = np.concatenate((noise_index, unlabeled_index), axis=0)
    classification(features, targets, labeled_index_new, test_index, unlabeled_index_new)

