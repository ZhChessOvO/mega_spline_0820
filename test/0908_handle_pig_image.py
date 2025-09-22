import os
import glob
from PIL import Image
import numpy as np

def select_and_save_images():
    # 源目录和目标目录
    source_dir = "/share/czh/stereo4d_right/dog/"
    target_images_dir = "/share/czh/stereo4d_right/dog_filtered/"
    target_gt_dir = "/share/czh/stereo4d_right/dog_gt/"
    
    # 创建目标目录（如果不存在）
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_gt_dir, exist_ok=True)
    
    # 获取所有jpg文件并排序
    jpg_files = glob.glob(os.path.join(source_dir, "*.jpg"))
    # 按文件名排序
    jpg_files.sort(key=lambda x: os.path.basename(x))
    
    # 确保有199张图片
    if len(jpg_files) != 199:
        print(f"警告：源目录中找到{len(jpg_files)}张图片，不是预期的199张")
    
    # 均匀选择32张图片的索引
    indices = np.linspace(0, len(jpg_files)-1, 32, dtype=int)
    
    # 处理并保存图片
    for i, idx in enumerate(indices):
        img_path = jpg_files[idx]
        
        # 打开图片并确保是RGB格式
        with Image.open(img_path) as img:
            # 转换为RGB格式（如果是RGBA或其他格式）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 保存到images_2目录 (000.png ~ 031.png)
            img_name = f"{i:03d}.png"
            img.save(os.path.join(target_images_dir, img_name))
            
            # 保存到gt目录 (v000_t000.png ~ v000_t031.png)
            gt_name = f"v000_t{i:03d}.png"
            img.save(os.path.join(target_gt_dir, gt_name))
    
    print(f"成功处理并保存了32张图片到指定目录")

if __name__ == "__main__":
    select_and_save_images()
