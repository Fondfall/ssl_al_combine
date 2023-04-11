# ***************************************************************************
# *
# * Description: label propagation
# * Author: Zou Xiaoyi (zouxy09@qq.com)
# * Date:   2015-10-15
# * HomePage: http://blog.csdn.net/zouxy09
# *
# **************************************************************************

import time
import math
import numpy as np
from label_propagation import labelPropagation
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    import matplotlib.pyplot as plt


    for i in range(Mat_Label.shape[0]):  # shape[0]输出行数
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dr')
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Db')
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], 'Dy')

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'or')
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'ob')
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], 'oy')

    plt.xlabel('X1');
    plt.ylabel('X2')
    plt.xlim(0.0, 1.25)
    plt.ylim(0.0, 1.25)
    plt.show()


def show2(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    import matplotlib.pyplot as plt
    random_matrix = np.random.rand(768, 2)
    dim = np.dot(Mat_Label, random_matrix)
    for i in range(Mat_Label.shape[0]):  # shape[0]输出行数
        dim_sig = dim[i]
        if int(labels[i]) == 0:
            plt.plot(dim_sig[0], dim_sig[1], 'or')
        elif int(labels[i]) == 1:
            plt.plot(dim_sig[0], dim_sig[1], 'oc')
        elif int(labels[i]) == 2:
            plt.plot(dim_sig[0], dim_sig[1], 'og')
        else:
            plt.plot(dim_sig[0], dim_sig[1], 'oy')
    dim2 = np.dot(Mat_Unlabel, random_matrix)
    for i in range(Mat_Unlabel.shape[0]):  # shape[0]输出行数
        dim2_sig = dim2[i]
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(dim2_sig[0], dim2_sig[1], '^r')
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(dim2_sig[0], dim2_sig[1], '^c')
        elif int(unlabel_data_labels[i]) == 2:
            plt.plot(dim2_sig[0], dim2_sig[1], '^g')
        else:
            plt.plot(dim2_sig[0], dim2_sig[1], '^y')
    plt.xlabel('X1 red-0 cyan-1 green-2 yellow-3')
    plt.ylabel('X2 o-label ^-unlabel')
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    plt.show()


def sigmoid(x):
    return 1.0/(1+np.exp(-x))


def loadCircleData(num_data):
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data = []
    theta = 0.0
    for i in range(num_inner):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2

    theta = 0.0
    for i in range(num_outer):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1

    Mat_Label = np.zeros((2, 2), np.float32)
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])
    labels = [0, 1]
    Mat_Unlabel = np.vstack(data)
    return Mat_Label, labels, Mat_Unlabel


def loadBandData(num_unlabel_samples):
    # Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    # labels = [0, 1]
    # Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])

    Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    labels = [0, 1]
    num_dim = Mat_Label.shape[1]
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)
    Mat_Unlabel[:num_unlabel_samples / 2, :] = (np.random.rand(num_unlabel_samples / 2, num_dim) - 0.5) * np.array(
        [3, 1]) + Mat_Label[0]
    Mat_Unlabel[num_unlabel_samples / 2: num_unlabel_samples, :] = (np.random.rand(num_unlabel_samples / 2,
                                                                                   num_dim) - 0.5) * np.array([3, 1]) + \
                                                                   Mat_Label[1]
    return Mat_Label, labels, Mat_Unlabel


def loadEduData(num_unlabel_samples):
    # data = np.loadtxt(open("features.csv", "rb"), delimiter=",", skiprows=0)
    bert_feature = np.loadtxt(open("w2v_big.csv", "rb"), delimiter=",", skiprows=0)
    X = bert_feature[:, : -1]  # 输入
    y = bert_feature[:, -1]  # 输出
    # X = data[:, : -1]  # 输入td-idf
    rng = np.random.RandomState(3)  # 随机数种子
    indices = np.arange(len(bert_feature))
    rng.shuffle(indices)  # 将索引打乱
    num_total = len(X)
    labeled_num = num_total - num_unlabel_samples
    # Mat_Label = X[:labeled_num]
    # labels = y[:labeled_num]
    # Mat_Unlabel = X[labeled_num:num_total]
    # unlabeled_true_label = y[labeled_num:num_total]
    Mat_Label = X[indices[:labeled_num]]
    labels = y[indices[:labeled_num]]
    Mat_Unlabel = X[indices[labeled_num:num_total]]
    unlabeled_true_label = y[indices[labeled_num:num_total]]
    return Mat_Label, labels, Mat_Unlabel, unlabeled_true_label


# main function
if __name__ == "__main__":
    num_unlabel_samples = 2500
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    # Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)
    Mat_Label, labels, Mat_Unlabel, true_labels = loadEduData(num_unlabel_samples)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf',
                                           # rbf_sigma = 2.04, max_iter=500)
    unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='knn', knn_num_neighbors=10,
                                           max_iter=300)
    # num=20
    # accuracy = []
    # gammas = np.logspace(-2, 2, num=num)
    # for i in range(num):
    #     unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type='rbf', rbf_sigma=gammas[i])
    #     score = accuracy_score(true_labels, unlabel_data_labels)
    #     accuracy.append(score)
    #     print(i,gammas[i],score)
    show2(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)
    print(classification_report(true_labels, unlabel_data_labels))
    # print(gammas)
    # print(accuracy)