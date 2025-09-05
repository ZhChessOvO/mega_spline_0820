import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

def visualize_depth_maps(scene_name, input_dir, output_path, n_rows=3, n_cols=4):
    """
    将12个深度图可视化并拼接成一张图
    
    参数:
        scene_name: 场景名称（用于标题）
        input_dir: 包含深度图的目录
        output_path: 拼接图像的保存路径
        n_rows: 行数
        n_cols: 列数
    """
    # 创建画布
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))
    fig.suptitle(f"Depth Maps - {scene_name}", fontsize=16)
    
    # 遍历12个深度图
    for i in range(12):
        # 计算子图位置
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        # 构建文件路径
        file_name = f"{i:03d}.npy"
        file_path = os.path.join(input_dir, file_name)
        
        try:
            # 加载深度图 (形状应为 (270, 480, 1))
            depth_map = np.load(file_path)
            
            # 如果有通道维度，移除它以便可视化
            if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
                depth_map = depth_map.squeeze(-1)
            
            # 应用颜色映射
            cmap = cm.get_cmap('viridis')  # 使用viridis颜色映射，适合深度可视化
            normalized_depth = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            colored_depth = cmap(normalized_depth)
            
            # 显示图像
            ax.imshow(colored_depth)
            ax.set_title(f"Frame {i:03d}")
            ax.axis('off')  # 关闭坐标轴
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            ax.text(0.5, 0.5, f"Error loading {i:03d}", 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    transform=ax.transAxes)
            ax.axis('off')
    
    # 调整子图间距
    plt.tight_layout()
    
    # 保存拼接后的图像
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存拼接图像至: {output_path}")

if __name__ == "__main__":
    # 配置参数
    SCENE_NAME = "Umbrella"  # 可以修改为其他场景名称
    BASE_DIR = "/share/czh/nvidia_megasam"
    INPUT_DIR = os.path.join(BASE_DIR, SCENE_NAME, "uni_depth")
    OUTPUT_IMAGE = f"/share/czh/nvidia_megasam/{SCENE_NAME}_depth_visualization.png"  # 输出图像文件名

    # 执行可视化
    visualize_depth_maps(
        scene_name=SCENE_NAME,
        input_dir=INPUT_DIR,
        output_path=OUTPUT_IMAGE
    )
    
    # 如果需要批量处理所有场景，可以使用以下代码
    """
    SCENES = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]
    for scene in SCENES:
        input_dir = os.path.join(BASE_DIR, scene, "uni_depth")
        output_image = f"{scene}_depth_visualization.png"
        visualize_depth_maps(scene, input_dir, output_image)
    """
    