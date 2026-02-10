# coding=utf-8
import heapq
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import windows,stft
plt.rcParams['font.sans-serif'] = ['sans-serif']  # 如果要显示中文字体,则在此处设为：SimHei
plt.rcParams['axes.unicode_minus'] = False  # 显示负号
import math
from sklearn.preprocessing import MinMaxScaler
import time
import librosa.core as lc

sns.set_theme(font='sans-serif', font_scale=2.0)

def plot_line(A, B, C, D, label):
    x = np.array(range(0, max(len(A), len(B), len(C), len(D)), 5))

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1
    plt.figure(figsize=(7, 5))
    plt.grid(linestyle="-")  # 设置背景网格线为虚线
    ax = plt.gca()
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['right'].set_visible(False)  # 去掉右边框

    plt.plot(A, color="cornflowerblue", label=f"Original Model", linewidth=1.5)
    plt.plot(B, color="red", label=f"Without Reverse", linewidth=1.5)
    plt.plot(C, color="olivedrab", label=f"Without Rotation", linewidth=1.5)
    plt.plot(D, color="darkorange", label=f"Without Rotation&Reverse", linewidth=1.5)
    plt.title('Train Accuracy', fontsize=15)
    group_labels = range(0, max(len(A), len(B), len(C), len(D)), 5)  # x轴刻度的标识
    plt.xticks(x, group_labels, fontsize=15, fontweight='bold')  # 默认字体大小为10
    plt.yticks(fontsize=15, fontweight='bold')
    plt.xlabel("Epoch", fontsize=15, fontweight='bold')
    if label == 'loss':
        plt.ylabel("Loss", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(1.0, 1.5)
    else:
        plt.ylabel("Accuracy", fontsize=15, fontweight='bold')
        plt.xlim(0, 85)  # 设置x轴的范围
        plt.ylim(0.40, 0.7)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12, fontweight='bold')  # 设置图例字体的大小和粗细

    plt.savefig(f'./Train {label}.svg', format='svg')  # 建议保存为svg格式,再用inkscape转为矢量图emf后插入word中
    plt.show()


def plot_modulations():
    data = pd.read_pickle('../dataset/RML2016.10a_dict.pkl')
    vis = []
    for item in data.items():
        (label, SNR), samples = item
        if SNR < 18:
            continue
        vis.append([label, samples[25]])
    plt.subplot(341)
    plt.plot(vis[0][1][0], color="cornflowerblue")
    plt.plot(vis[0][1][1], color="lightcoral")
    plt.title(vis[0][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(342)
    plt.plot(vis[1][1][0], color="cornflowerblue")
    plt.plot(vis[1][1][1], color="lightcoral")
    plt.title(vis[1][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(343)
    plt.plot(vis[2][1][0], color="cornflowerblue")
    plt.plot(vis[2][1][1], color="lightcoral")
    plt.title(vis[2][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(344)
    plt.plot(vis[3][1][0], color="cornflowerblue")
    plt.plot(vis[3][1][1], color="lightcoral")
    plt.title(vis[3][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(345)
    plt.plot(vis[4][1][0], color="cornflowerblue")
    plt.plot(vis[4][1][1], color="lightcoral")
    plt.title(vis[4][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(346)
    plt.plot(vis[5][1][0], color="cornflowerblue")
    plt.plot(vis[5][1][1], color="lightcoral")
    plt.title(vis[5][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(347)
    plt.plot(vis[6][1][0], color="cornflowerblue")
    plt.plot(vis[6][1][1], color="lightcoral")
    plt.title(vis[6][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(348)
    plt.plot(vis[7][1][0], color="cornflowerblue")
    plt.plot(vis[7][1][1], color="lightcoral")
    plt.title(vis[7][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(349)
    plt.plot(vis[8][1][0], color="cornflowerblue")
    plt.plot(vis[8][1][1], color="lightcoral")
    plt.title(vis[8][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 10)
    plt.plot(vis[9][1][0], color="cornflowerblue")
    plt.plot(vis[9][1][1], color="lightcoral")
    plt.title(vis[9][0])
    plt.xticks([])
    plt.yticks([])
    plt.subplot(3, 4, 11)
    plt.plot(vis[10][1][0], color="cornflowerblue")
    plt.plot(vis[10][1][1], color="lightcoral")
    plt.title(vis[10][0])
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f'visualize_of_modulations.svg', format='svg', dpi=450)
    plt.show()

def center_f(data):
    sum_f = []
    for i in range(data.shape[0]):
        f_sum = sum(data[i])
        print(f_sum)
        sum_f.append(f_sum)
    max_number = heapq.nlargest(2,sum_f)
    max_idx = []
    for t in max_number:    
        index = sum_f.index(t)
        max_idx.append(index)
        sum_f[index] = 0
    print(max_idx)
    idx = np.array(max_idx).mean()
    return int(idx)

def rotation_2d(x):
    x_aug1 = np.empty(x.shape)
    x_aug2 = np.empty(x.shape)
    x_aug3 = np.empty(x.shape)    
    x_aug1[0, :] = -x[1, :]
    x_aug1[1, :] = x[0, :]
    x_aug2 = -x
    x_aug3[0, :] = x[1, :]
    x_aug3[1, :] = -x[0, :]
    return x_aug1,x_aug2,x_aug3
