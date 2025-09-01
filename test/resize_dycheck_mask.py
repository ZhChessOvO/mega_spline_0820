import os
from PIL import Image

def resize_masks_to_match_images():
    """
    将mask图像缩小到与对应image图像相同的尺寸，并确保保存为灰度图像
    mask和image文件名相同，分别位于不同目录
    """
    # 定义mask和image的目录路径
    mask_dir = "/share/czh/dycheck_splinegs/block/motion_masks/"
    image_dir = "/share/czh/dycheck_splinegs/block/images_2/"
    
    # 检查目录是否存在
    if not os.path.exists(mask_dir):
        print(f"错误：mask目录 {mask_dir} 不存在")
        return
    
    if not os.path.exists(image_dir):
        print(f"错误：image目录 {image_dir} 不存在")
        return
    
    # 获取mask目录中所有PNG文件
    mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".png")]
    
    if not mask_files:
        print(f"在mask目录 {mask_dir} 中未找到PNG文件")
        return
    
    # 处理每个mask文件
    for filename in mask_files:
        try:
            # 构建文件路径
            mask_path = os.path.join(mask_dir, filename)
            image_path = os.path.join(image_dir, filename)
            
            # 检查对应的image是否存在
            if not os.path.exists(image_path):
                print(f"警告：对应的image文件 {filename} 不存在，跳过处理")
                continue
            
            # 打开图像获取目标尺寸
            with Image.open(image_path) as img:
                target_size = img.size  # (width, height)
            
            # 打开mask并调整大小
            with Image.open(mask_path) as mask:
                # 确保mask是灰度模式
                if mask.mode != 'L':
                    mask = mask.convert('L')
                    print(f"已将 {filename} 转换为灰度模式")
                
                # 使用LANCZOS滤镜进行缩小，保持图像质量
                resized_mask = mask.resize(target_size, Image.Resampling.LANCZOS)
                
                # 保存调整后的mask（覆盖原文件），明确指定为PNG格式和灰度模式
                resized_mask.save(mask_path, format='PNG', mode='L')
                print(f"已调整尺寸并保存为灰度图像：{filename}，新尺寸：{target_size}")
                
        except Exception as e:
            print(f"处理 {filename} 时出错：{str(e)}")
    
    print("所有mask尺寸调整完成，均已保存为灰度图像")

if __name__ == "__main__":
    resize_masks_to_match_images()
    