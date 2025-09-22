import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def replace_depth_maps(npz_path, target_dir):
    # 检查目标目录是否存在
    if not os.path.exists(target_dir):
        raise ValueError(f"目标目录不存在: {target_dir}")
    
    # 加载NPZ文件
    print(f"加载NPZ文件: {npz_path}")
    data = np.load(npz_path)
    
    # 检查是否包含depths字段
    if 'depths' not in data:
        raise ValueError("NPZ文件中不包含'depths'字段")
    
    depths = data['depths']
    
    # 检查深度图数量是否为12
    if depths.shape[0] != 12:
        raise ValueError(f"深度图数量应为12，实际为{depths.shape[0]}")
    
    print(f"找到{depths.shape[0]}个深度图，形状为{depths.shape}")
    
    # 遍历每个深度图并替换对应的文件
    for i in range(12):
        # 格式化索引为3位数字（000到011）
        idx_str = f"{i:03d}"
        
        # 获取当前深度图
        depth_map = depths[i]
        
        # 保存为NPY文件
        npy_path = os.path.join(target_dir, f"{idx_str}.npy")
        np.save(npy_path, depth_map)
        print(f"已保存NPY文件: {npy_path}")
        
        # 归一化深度图以便保存为PNG
        # 注意：这里假设深度图值为非负，实际应用中可能需要根据数据范围调整
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        
        if depth_max > depth_min:
            # 归一化到0-255
            depth_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
        else:
            # 如果所有值都相同，创建一个全灰图像
            depth_normalized = np.ones_like(depth_map, dtype=np.uint8) * 128

        # 应用VIRIDIS颜色映射
        cm = plt.cm.get_cmap('viridis')
        depth_colored = cm(depth_normalized)

        # 转换为8位RGB图像
        depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
        
        # 保存为PNG文件
        png_path = os.path.join(target_dir, f"{idx_str}.png")
        Image.fromarray(depth_colored).save(png_path)
        print(f"已保存PNG文件: {png_path}")
    
    print("所有深度图替换完成")

if __name__ == "__main__":

    scenes = [
        "Balloon1",
        "Balloon2",
        "Jumping",
        "Playground",
        "Skating",
        "Truck",
        "Umbrella"
    ]

    # NPZ文件和目标目录的基础路径
    npz_base_path = "/home/czh/code/mega-sam/outputs_cvd"
    target_base_dir = "/share/czh/nvidia_megasam"
    
    # 依次处理每个场景
    for scene in scenes:
        print(f"\n===== 开始处理场景: {scene} =====")
        try:
            # 构建当前场景的NPZ文件路径和目标目录路径
            npz_file_path = os.path.join(npz_base_path, f"{scene}_sgd_cvd_hr.npz")
            target_directory = os.path.join(target_base_dir, scene, "uni_depth")
            
            # 确保目标目录存在
            os.makedirs(target_directory, exist_ok=True)
            
            # 执行替换操作
            replace_depth_maps(npz_file_path, target_directory)
            print(f"场景 {scene} 处理完成")
        except Exception as e:
            print(f"场景 {scene} 处理失败: {str(e)}")
    
    print("\n所有场景处理完毕")
