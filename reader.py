import numpy as np
from core.utils import vars_viz
import torch

path1 = 'results/stereo/stereo/est/file/000004.npy'
path2 = 'results/stereo/stereo/gt/file/000004.npy'

data1, data2 = np.load(path1), np.load(path2)
data1, data2 = data1.flatten(), data2.flatten()
print(data1.shape, data2.shape)

# data1,2 的频率分布柱状图,合并在一张图上
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
print(data1.max(), data1.min(), data2.max(), data2.min())
ax.hist(data1, bins=100, alpha=0.5, label='data1', color='r')
ax.hist(data2, bins=100, alpha=0.5, label='data2', color='b')
ax.legend()
plt.show()
