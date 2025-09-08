import os
import shutil

# 定义路径
base_dir = "/share/czh/stereo4d/ski"
instance_mask_dir = os.path.join(base_dir, "instance_mask")
motion_masks_dir = os.path.join(base_dir, "motion_masks")

# 创建instance_mask目录（如果不存在）
os.makedirs(instance_mask_dir, exist_ok=True)

# 创建000到059共60个文件夹并复制重命名文件
for i in range(32):
    folder_name = f"{i:03d}"  # 格式化为3位数字，如000, 001, ..., 059
    folder_path = os.path.join(instance_mask_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    print(f"创建文件夹: {folder_path}")

    # 源文件路径 (000.png, 001.png, ..., 059.png)
    src_filename = f"{folder_name}.png"
    src_path = os.path.join(motion_masks_dir, src_filename)
    
    # 目标文件路径，重命名为000.png
    dest_filename = "000.png"
    dest_path = os.path.join(folder_path, dest_filename)
    
    # 检查源文件是否存在，如果存在则复制
    if os.path.exists(src_path):
        shutil.copy2(src_path, dest_path)  # 使用copy2保留文件元数据
        print(f"复制并命名文件: {src_path} -> {dest_path}")
    else:
        print(f"警告: 源文件不存在 - {src_path}")

print("操作完成")
    