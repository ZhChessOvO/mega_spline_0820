import numpy as np
import os

# 定义场景列表
scenes = ["Balloon1", "Balloon2", "Jumping", "Playground", "Skating", "Truck", "Umbrella"]

# 定义数据路径模板
paths = [
    "/share/czh/nvidia_rodynrf/{Scene}/uni_depth/000.npy",
    "/share/czh/nvidia_megasam/{Scene}/uni_depth/000.npy"
]

# 存储结果的列表
results = []

# 遍历所有场景和路径
for scene in scenes:
    for path_template in paths:
        # 构建完整路径
        file_path = path_template.format(Scene=scene)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        # 读取npy文件
        try:
            depth_data = np.load(file_path)
            
            # 计算统计信息
            shape = depth_data.shape
            max_val = np.max(depth_data)
            min_val = np.min(depth_data)
            mean_val = np.mean(depth_data)
            
            # 存储结果
            results.append({
                "scene": scene,
                "source": "nvidia_rodynrf" if "rodynrf" in path_template else "nvidia_megasam",
                "path": file_path,
                "shape": shape,
                "max": max_val,
                "min": min_val,
                "mean": mean_val
            })
            
            print(f"已处理: {file_path}")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

# 打印结果表格
print("\n" + "="*80)
print(f"{'场景':<12} {'来源':<15} {'形状':<15} {'最大值':<10} {'最小值':<10} {'平均值':<10}")
print("-"*80)
for result in results:
    print(f"{result['scene']:<12} {result['source']:<15} {str(result['shape']):<15} "
          f"{result['max']:<10.4f} {result['min']:<10.4f} {result['mean']:<10.4f}")
print("="*80)
    