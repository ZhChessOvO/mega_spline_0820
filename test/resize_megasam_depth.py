import numpy as np
import os
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from matplotlib import cm

# 定义场景列表
SCENES = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]
# 基础目录
BASE_DIR = "/share/czh/nvidia_megasam"
# 目标尺寸
TARGET_SHAPE = (270, 480)
# 原始尺寸要求
REQUIRED_SHAPE = (328, 584)

def visualize_depth(depth_map, output_path):
    """将深度图可视化并保存为PNG文件"""
    # 确保深度图是2D数组
    if depth_map.ndim == 3 and depth_map.shape[-1] == 1:
        depth_map = depth_map.squeeze()
    
    # 归一化深度值到0-1范围以便可视化
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    if depth_max > depth_min:  # 避免除以零
        depth_normalized = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth_map)
    
    # 应用VIRIDIS颜色映射
    cmap = cm.get_cmap('viridis')
    colored_depth = cmap(depth_normalized)
    
    # 保存为PNG文件（去掉alpha通道）
    plt.imsave(output_path, colored_depth[:, :, :3])
    print(f"已保存可视化图像: {output_path}")

def process_scene(scene_name):
    """处理单个场景的深度图文件，包括调整尺寸和可视化"""
    # 构建目录路径
    depth_dir = os.path.join(BASE_DIR, scene_name, "uni_depth")
    
    # 检查目录是否存在
    if not os.path.exists(depth_dir):
        print(f"错误: 目录不存在 - {depth_dir}")
        return False
    
    # 处理12个文件
    for i in range(12):
        file_name = f"{i:03d}.npy"
        file_path = os.path.join(depth_dir, file_name)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"错误: 文件不存在 - {file_path}")
            return False
        
        # 加载深度图
        try:
            depth_map = np.load(file_path)
            print(f"加载 {file_path}，数据类型: {depth_map.dtype}，形状: {depth_map.shape}")
        except Exception as e:
            print(f"错误: 加载文件 {file_path} 失败 - {str(e)}")
            return False
        
        # 检查形状是否符合要求
        if depth_map.shape != REQUIRED_SHAPE:
            print(f"错误: {file_path} 形状不符合要求，实际为 {depth_map.shape}，需要 {REQUIRED_SHAPE}")
            return False
        
        # 转换数据类型为float32以支持缩放操作
        try:
            depth_float = depth_map.astype(np.float32)
        except Exception as e:
            print(f"错误: 转换 {file_path} 数据类型失败 - {str(e)}")
            return False
        
        # 计算缩放因子
        zoom_factor = (
            TARGET_SHAPE[0] / depth_float.shape[0],
            TARGET_SHAPE[1] / depth_float.shape[1]
        )
        
        # 缩放深度图（使用双线性插值）
        try:
            resized = zoom(depth_float, zoom_factor, order=1)  # order=1 表示双线性插值
        except Exception as e:
            print(f"错误: 缩放 {file_path} 失败 - {str(e)}")
            return False
        
        # 检查缩放后的形状
        if resized.shape != TARGET_SHAPE:
            print(f"错误: {file_path} 缩放后形状不正确，得到 {resized.shape}，需要 {TARGET_SHAPE}")
            return False
        
        # 调整形状为 (270, 480, 1)
        resized_reshaped = resized.reshape(*TARGET_SHAPE, 1)
        
        # 覆盖保存文件（使用原始数据类型或保持float32）
        try:
            # 尝试使用原始数据类型保存，如果失败则使用float32
            np.save(file_path, resized_reshaped.astype(depth_map.dtype))
        except:
            np.save(file_path, resized_reshaped)
            
        print(f"已处理: {file_path} -> 新形状 {resized_reshaped.shape}，数据类型: {resized_reshaped.dtype}")
        
        # 生成可视化图像并保存
        viz_filename = f"{i:03d}_resized.png"
        viz_path = os.path.join(depth_dir, viz_filename)
        visualize_depth(resized_reshaped, viz_path)
    
    return True

if __name__ == "__main__":
    # 确保matplotlib不使用GUI后端
    plt.switch_backend('Agg')
    
    # 遍历所有场景并处理
    for scene in SCENES:
        print(f"\n开始处理场景: {scene}")
        success = process_scene(scene)
        if success:
            print(f"场景 {scene} 处理完成")
        else:
            print(f"场景 {scene} 处理失败，请检查错误信息")
    
    print("\n所有场景处理完毕")