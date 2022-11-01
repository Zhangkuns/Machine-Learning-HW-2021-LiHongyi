import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# For data preprocess
import numpy as np
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pprint as pp

# For plotting
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

tr_path = 'covid.train.csv'  # path to training data
tt_path = 'covid.test.csv'  # path to testing data

data_tr = pd.read_csv(tr_path)  # 读取训练数据
data_tt = pd.read_csv(tt_path)  # 读取测试数据
data_tr.drop(['id'], axis=1, inplace=True)  # 由于id列用不到，删除id列
data_tt.drop(['id'], axis=1, inplace=True)
cols = list(data_tr.columns)  # 拿到特征列名称
WI_index = cols.index('WI')  # WI列是states one-hot编码最后一列，取值为0或1，后面特征分析时需要把states特征删掉
data_tr.iloc[:, 40:].describe()  # 从上面可以看出wi 列后面是cli, 所以列索引从40开始， 并查看这些数据分布
data_tt.iloc[:, 40:].describe()  # 查看测试集数据分布，并和训练集数据分布对比，两者特征之间数据分布差异不是很大
data_tr.iloc[:, 40:].corr()  # 上面手动分析太累，还是利用corr方法自动分析
# 锁定上面相关性矩阵最后一列，也就是目标值列，每行是与其相关性大小
data_corr = data_tr.iloc[:, 40:].corr()
target_col = data_corr['tested_positive.2']
feature = target_col[target_col > 0.8]  # 在最后一列相关性数据中选择大于0.8的行，这个0.8是自己设的超参，大家可以根据实际情况调节
feature_cols = feature.index.tolist()  # 将选择特征名称拿出来
feature_cols.pop()  # 去掉test_positive标签
pp.pprint(feature_cols)  # 得到每个需要特征名称列表
feats_selected = [cols.index(col) for col in feature_cols]  # 获取该特征对应列索引编号，后续就可以用feats + feats_selected作为特征值

