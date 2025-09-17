import os
from PIL import Image

def convert_rgba_to_rgb(directory):
    """
    将指定目录下的所有RGBA格式PNG图片转换为RGB格式并覆盖保存
    """
    # 检查目录是否存在
    if not os.path.exists(directory):
        print(f"错误：目录 {directory} 不存在")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        # 只处理PNG文件
        if filename.lower().endswith('.png'):
            file_path = os.path.join(directory, filename)
            
            try:
                # 打开图片
                with Image.open(file_path) as img:
                    # 检查是否为RGBA格式
                    if img.mode == 'RGBA':
                        # 转换为RGB格式
                        rgb_img = img.convert('RGB')
                        # 覆盖保存
                        rgb_img.save(file_path)
                        print(f"已转换: {file_path}")
                    else:
                        print(f"无需转换 (非RGBA格式): {file_path}")
            except Exception as e:
                print(f"处理 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 要处理的两个目录
    directories = [
        "/share/czh/nvidia_moge/Playground/images_2",
        "/share/czh/nvidia_moge/Truck/images_2"
    ]
    
    # 处理每个目录
    for dir_path in directories:
        print(f"\n开始处理目录: {dir_path}")
        convert_rgba_to_rgb(dir_path)
    
    print("\n处理完成")
    