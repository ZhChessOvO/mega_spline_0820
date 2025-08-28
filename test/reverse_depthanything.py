import numpy as np
import os

# 输入目录，存放原始深度图
input_dir = "/share/czh/nvidia_rodynrf/Balloon1/depth_anything_old"
# 输出目录，存放处理后的深度图
output_dir = "/share/czh/nvidia_rodynrf/Balloon1/depth_anything"

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录中的所有.npy文件
for filename in os.listdir(input_dir):
    if filename.endswith(".npy"):
        # 构建完整文件路径
        input_path = os.path.join(input_dir, filename)
        
        # 加载深度数据
        depth_data = np.load(input_path)
        
        # 保存原始的最大值和最小值
        original_min = np.min(depth_data)
        original_max = np.max(depth_data)
        
        # 计算纠正后的深度值，保持原有的最大值和最小值，但反转中间值
        # 公式：corrected = original_min + original_max - depth_data
        corrected_depth = original_min + original_max - depth_data
        
        # 验证纠正后的数据是否保持了原始的最值
        new_min = np.min(corrected_depth)
        new_max = np.max(corrected_depth)
        if not np.isclose(new_min, original_min) or not np.isclose(new_max, original_max):
            print(f"警告: {filename} 处理后最值发生变化！原始: ({original_min}, {original_max}), 新值: ({new_min}, {new_max})")
        
        # 保存纠正后的深度图
        output_path = os.path.join(output_dir, filename)
        np.save(output_path, corrected_depth)
        print(f"已处理: {filename}")

print("所有深度图处理完成，已保存至:", output_dir)
