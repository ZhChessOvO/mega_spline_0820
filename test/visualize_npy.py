import numpy as np
import matplotlib.pyplot as plt
import cv2

# 1. 读取 .npy 文件（关键：深度图通常为单通道，形状可能是 (H, W) 或 (H, W, 1)）
file_path = "/share/czh/nvidia_moge/Truck/uni_depth/000.npy"
depth_data = np.load(file_path)

# 2. 预处理：若有多余通道，压缩为单通道（如 (H,W,1) → (H,W)）
if depth_data.ndim == 3:
    depth_data = np.squeeze(depth_data)  # 去除维度为1的轴
print(f"深度图形状：{depth_data.shape}，数值范围：{depth_data.min()} ~ {depth_data.max()}")  # 查看数据基本信息

# 3. 可视化（两种常用风格：灰度图+伪彩色图，更直观区分深度差异）
plt.figure(figsize=(12, 5))  # 设置画布大小

# 子图1：灰度图（数值越小越黑，越大越白，直接反映深度原始值）
plt.subplot(1, 2, 1)
plt.imshow(depth_data, cmap="gray")
plt.title("Depth Map (Grayscale)", fontsize=12)
plt.axis("off")  # 隐藏坐标轴
plt.colorbar(label="Depth Value")  # 颜色条：显示数值与颜色对应关系

# 子图2：伪彩色图（用彩色梯度区分深度，更适合人眼识别差异，如Jet色系）
plt.subplot(1, 2, 2)
im = plt.imshow(depth_data, cmap="jet")  # "jet" 是深度图常用色系，也可尝试 "viridis" "plasma"
plt.title("Depth Map (Pseudocolor - Jet)", fontsize=12)
plt.axis("off")
plt.colorbar(im, label="Depth Value")

# 保存图像（可选，保存为 PNG 格式，路径可自定义）
plt.tight_layout()  # 自动调整子图间距
plt.savefig("/home/czh/code/SplineGS/test/image/moge_test.png", dpi=300, bbox_inches="tight")
# plt.show()