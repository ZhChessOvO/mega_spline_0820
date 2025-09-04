import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from PIL import Image

# 创建保存图像的目录
os.makedirs('./image', exist_ok=True)

# 加载NPZ文件
npz_path = '/home/czh/code/mega-sam/outputs_cvd/Balloon1_sgd_cvd_hr.npz'
data = np.load(npz_path)

# 提取深度图数据
depth_maps = data['depths']  # 形状为(120, 512, 384)
print(f"加载的深度图形状: {depth_maps.shape}")

# 每10张选取一张，共12张
selected_indices = range(0, 12)
selected_maps = [depth_maps[i] for i in selected_indices]
print(f"选中的深度图数量: {len(selected_maps)}")

# 确定子图排列方式 (3行4列)
rows, cols = 3, 4
fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
fig.subplots_adjust(hspace=0.3, wspace=0.1)

# 用于归一化的最大最小值
all_values = np.concatenate(selected_maps)
vmin, vmax = all_values.min(), all_values.max()

# 绘制每张深度图
for i, (ax, depth_map) in enumerate(zip(axes.flatten(), selected_maps)):
    # 使用热图样式显示深度图
    im = ax.imshow(depth_map, cmap=cm.viridis, vmin=vmin, vmax=vmax)
    ax.set_title(f"Frame {selected_indices[i]}", fontsize=10)
    ax.axis('off')  # 关闭坐标轴

# 添加颜色条
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax)

# 保存合并后的大图
output_path = './image/depth_maps_combined2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"合并后的深度图已保存至: {output_path}")

# 可选：单独保存每张选中的深度图
# for i, idx in enumerate(selected_indices):
#     depth_map = selected_maps[i]
#     # 归一化到0-255范围以便保存为图像
#     normalized = ((depth_map - vmin) / (vmax - vmin) * 255).astype(np.uint8)
#     # 转换为伪彩色图像
#     colored = cm.viridis(normalized)
#     colored_image = Image.fromarray((colored * 255).astype(np.uint8))
#     colored_image.save(f'./image/depth_map_{idx}.png')

print("所有选中的深度图已单独保存至./images目录")
