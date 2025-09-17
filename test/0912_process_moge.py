import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

# 定义场景列表
scenes = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]

# 处理每个场景
for scene in scenes:
    print(f"处理场景: {scene}")
    
    # 定义输入和输出路径
    input_dir = f"/share/czh/nvidia_moge/{scene}/uni_depth"
    output_dir = f"/share/czh/nvidia_moge/{scene}/depth_anything"
    vis_path = f"/share/czh/nvidia_moge/{scene}_fixed_moge.png"
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 存储所有处理后的深度图用于可视化
    processed_depths = []
    all_values = []  # 收集所有值以确定可视化的全局范围
    
    # 处理12张深度图
    for i in range(12):
        # 读取原始npy文件
        input_path = os.path.join(input_dir, f"{i:03d}.npy")
        depth_data = np.load(input_path)
        
        # 转换形状 (H, W, 1) -> (H, W)
        depth_2d = depth_data.squeeze(axis=2)
        
        # 计算当前深度图的最小值和最大值
        current_min = np.min(depth_2d)
        current_max = np.max(depth_2d)
        
        # 反转深度值（保持当前npy文件的数值范围）
        inverted_depth = current_max + current_min - depth_2d
        
        # 保存处理后的npy文件
        output_path = os.path.join(output_dir, f"{i:03d}.npy")
        np.save(output_path, inverted_depth)
        
        # 收集数据用于可视化
        processed_depths.append(inverted_depth)
        all_values.append(inverted_depth)
    
    # 计算全局的最小值和最大值用于可视化的统一颜色范围
    all_values = np.concatenate(all_values)
    global_min = np.min(all_values)
    global_max = np.max(all_values)
    
    # 创建可视化结果
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f"Depth Maps for {scene}", fontsize=20)
    
    # 使用viridis colormap
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=global_min, vmax=global_max)
    
    # 显示12张深度图
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(processed_depths[i], cmap=cmap, norm=norm)
        ax.set_title(f"Frame {i:03d}")
        ax.axis('off')
    
    # 添加颜色图例
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Depth Value', rotation=270, labelpad=20)
    
    # 调整布局并保存
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为颜色条留出空间
    plt.savefig(vis_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"场景 {scene} 处理完成\n")

print("所有场景处理完毕！")
    