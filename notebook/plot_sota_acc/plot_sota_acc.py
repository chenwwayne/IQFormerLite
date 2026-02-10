import pandas as pd
import matplotlib.pyplot as plt
import math

dataset_name = 'RML201610b'
# 读取 Excel 文件
df = pd.read_csv(f'{dataset_name}.csv')

# 只取SNR>=0的数据用于绘制
df_positive = df[df['SNR'] >= 0]

# 要绘制的列名
# columns_to_plot = ['1DCNN-PF','CLDNN','CNN1','DAE','DenseNet','IC-AMCNet','LSTM2','FT-T','KAFT-T','DyTKAFT-T']
columns_to_plot = ['MCLDNN', 'MCFormer', 'PET-CGDNN', 'AMC-Net', 'FEA-T', 'IQFormer', 'IQFormerLite']

# ---------- 图例显示名映射 ----------
legend_map = {
    'DyTKAFT-T': 'DyTKAFT-T (Ours)',
    # 其余不改的列名默认保持原样
}

# 设置图像大小
fig, ax = plt.subplots(figsize=(8, 6))

# 颜色    浅色 HEX   深色 HEX
# 蓝色    '#ADD8E6','#00008B'
# 绿色    '#90EE90','#006400'
# 红色    '#FFB6C1','#8B0000'
# 黄色    '#FFFFE0','#FFD700'
# 青色    '#E0FFFF','#00CED1'
# 紫色    '#E6E6FA','#4B0082'
# 橙色    '#FFE4B5','#FF8C00'
# 棕色    '#F5DEB3','#8B4513'
# 灰色    '#D3D3D3','#696969'
# 粉色    '#FFCCE5','#C71585'
# 橄榄绿  '#F0FFF0','#556B2F'
# 海蓝    '#B0E0E6','#4682B4'
# 酒红    '#FFE4E1','#800000'
# 金色    '#FFF8DC','#B8860B'
# 靛青    '#F0F8FF','#191970'

base_colors = [
    '#ADD8E6',  
    '#90EE90',  
    '#FFB6C1',  
    '#FFCCE5',  
    '#B0E0E6',  
    '#E6E6FA',
    '#FFE4B5'
]

# 标记样式
markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', '*', 'h', 'H', 'x', 'd', '+', '1']

for idx, column in enumerate(columns_to_plot):
    if column not in df.columns:
        print(f"警告：列名 '{column}' 不在 Excel 文件中，已跳过。")
        continue

    # 默认按交替浅深颜色
    color = base_colors[idx % len(base_colors)]
    label_txt = legend_map.get(column, column)   # 关键：映射图例文字

    # 特殊处理指定曲线
    if column == 'IQFormerLite':
        color = '#800000'  # 亮黄色 (Yellow)
    # elif column == 'FKAFT-T':
    #     color = '#FFFF00'  # 亮黄色 (Yellow)

    ax.plot(
        df['SNR'],
        df[column],
        label=label_txt,
        marker=markers[idx % len(markers)],
        color=color,
        linewidth=2,
        markersize=10
    )

# 主图设置
ax.set_xlim(-22, 20)
ax.set_ylim(0, 1)
ax.grid(True, which='both', linestyle='--', alpha=0.6)
ax.set_xticks(range(-20, 21, 5))
ax.set_yticks([i/10 for i in range(0, 11)])
ax.tick_params(labelsize=12)
ax.set_title(f'{dataset_name}', fontsize=18)
ax.set_xlabel('SNR (dB)', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)

# 添加大图的图例
ax.legend(fontsize=10, loc='upper left', ncol=1)

# 添加小图（放大局部，SNR>0部分）
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

axins = inset_axes(ax, width="45%", height="45%", loc='lower right')  # 小图位置

for idx, column in enumerate(columns_to_plot):
    if column not in df.columns:
        continue

    color = base_colors[idx % len(base_colors)]
    if column == 'IQFormerLite':
        color = '#800000'  # 浅黄色
    # elif column == 'FKAFT-T':
    #     color = '#FFFF00'  # 亮黄色

    axins.plot(
        df_positive['SNR'],
        df_positive[column],
        marker=markers[idx % len(markers)],
        color=color,
        linewidth=2,
        markersize=6
    )

axins.set_xlim(-1, 20)
ymin = math.floor(df_positive.iloc[:, 1:].min().min() * 100) / 100 - 0.01
ymax = math.floor(df_positive.iloc[:, 1:].max().max() * 100) / 100 + 0.01
axins.set_ylim(ymin, ymax)
axins.grid(True, linestyle='--', alpha=0.5)
axins.set_xticks(range(0, 21, 5))
axins.tick_params(labelsize=8)

# 调整整体布局
plt.tight_layout()
plt.savefig(f'result/{dataset_name}_sota_acc.png', dpi=300)
plt.savefig(f'result/{dataset_name}_sota_acc.svg', format="svg")
