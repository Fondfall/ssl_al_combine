import numpy as np
import torch
from collections import Counter

# for i in range(4):
#     rng = np.random.RandomState(i)
#     print(rng.rand(2, 3))

# rng = np.random.RandomState(2)  # 随机数种子
# indices = np.arange(10)
# print("1:", indices)
# rng.shuffle(indices)
# print("2:", indices)

import os
import pandas as pd
# os.getcwd()
# a = np.array([[1, 2], [3, 4]])
# b = np.array([5, 6]).reshape(2, 1)
# c = np.append(a, b, axis=1)
# print(a, b, c)
# s = pd.DataFrame(c)
# print(s)
# s.to_csv("test.csv", header=None, index=None)


import csv

# data = np.loadtxt(open("test.csv", "rb"), delimiter=",", skiprows=0)
# # with open("test.csv") as f:
# #     f_csv = csv.reader(f)
# #     for row in f_csv:
# #         data.append(row)
# print(data)
# print(len(data))
# print(data[1, 1])
# X = data[:, : -1]
# y = data[:, -1]
# print("X:", X)
# print("y:", y)
# print(len(y))

# X=torch.range(1,4)
# dot = torch.dot(X,X)
# print(X,dot)

# a=torch.tensor([[1,2,3,2,1],[1,2,3,4,3]])
# print(a)
# # x=a[1]
# # print(x)
# # # print(x[1,1])
# # a,index = torch.sort(x,descending=True)
# # print(a,index)
# # index = index[:3]
# # print(x[index],index)
# # l=x[index].tolist()
# # print(l)
# # print(max(index,key=index.count))
# b=a[1]
# c=torch.tensor([0,1,2,3])
# print(b[c])
# index = torch.nonzero(a[1]==3).squeeze()
# print(index)
#
# a=[]
# a.append(1)
# print(a)
# b=[1,1,1]
# print(b[a[0]])

# a=torch.tensor([[1,2],[3,4,5]])
# b=a.numel()
# c=a.reshape(b)
# print(a,b,c)
#
# a=torch.tensor([[1,2,3],[4,5]])
# print(a.numel())

# a=[]
# b=[1,2,3]
# c=[4,5,6,7]
# d=a+b+c
# print(d)
# print(len(d))

# a=[[1,2,3],[4,5,6]]
# b=[[7,8,9]]
# c=np.concatenate((a,b),axis=0)
# print(c)

# a=np.array(range(100))
# b=np.array(range(10))
# print(np.concatenate((a,b),axis=0))
# a = [
#     ["我", "和", "你"],
#     ["心", "连", "心"]
# ]


# a = [
#     ['我', '和', '你'],
#     ['心', '连']
# ]
# print(a[1][1])
# b = []
# for i in range(len(a)):
#     temp = ""
#     for j in range(len(a[i])):
#         print(a[i][j])
#         if len(a[i]) - j == 1:
#             space = ""
#         else:
#             space = " "
#         temp = temp + a[i][j] + space
#     b.append(temp)
# print(b)

import re
# a=["ss(\\n)","bc"]
# c=[]
# for i in range(len(a)):
#     print(a[i])
#     a[i]=a[i].replace('\\n', '')
# print(type(a))
# print(a)
# # print(c)
# s = "123\n456"
# print(s)
# s=s.replace("\n", "")
# print(s)

# filter = [
#     "\\n", "\\xa0", "~", ".", "。", "'", '"', "?", "？", "!", "！",
#     "(", ")", "（", "）", "…", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "+", "-",
#     "^", "/", "ω", "`", "´", "＾", "﹡", "*"
#     ]
# print(type(filter[2]))
# a = ['已经知道了，谢谢了,0',"abc'ss'"]
# print(a[0])
# a[0]=a[0].replace('，',"")
# print(a)
import xlrd
#
# data = xlrd.open_workbook("data_lite.xlsx", encoding_override='utf-8')
# table = data.sheets()[0]  # 选定表
# n_rows = table.nrows  # 获取行号
# text = []
# for i in range(1, n_rows):  # 第0行为表头
#     tex = table.row_values(i)  # 循环输出excel表中每一行，即所有数据
#     result = tex[1]  # 取出表中列数据
#     text.append(result)
# print(text[0:5])

# a = np.logspace(-2, 2, num=10)
# print(a)

# def sigmoid(x):
#     return 1.0/(1+np.exp(-x))
# a= np.array([1,2,3,4])
# b=sigmoid(a)
# print(b)

# n_total_samples = 330
# n_labeled_points = 40
# max_iterations = 5
#
# unlabeled_indices = np.arange(n_total_samples)[n_labeled_points:]
# print(unlabeled_indices)

# a=3
# print(a**0.5)

# a=[1.0,2.0,3.0]
# b=[3,4]
# c=a+b
# c=[int(i) for i in c]
# print(c)

# a = torch.tensor([1,2,3])
# b=torch.dot(a,a)
# print(b.item()**2)
a=np.array(range(10))
print(a[:5])
print(a[5:])
