import os
from PIL import Image

def invert_black_white_images():
    """
    将motion_masks目录下的黑白灰度PNG图像反色处理
    黑色(0)变为白色(255)，白色(255)变为黑色(0)
    """
    # 定义目标目录路径
    dir_path = "/share/czh/dycheck_splinegs/block/motion_masks/"
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"错误：目录 {dir_path} 不存在")
        return
    
    # 获取目录中所有PNG文件
    png_files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
    
    if not png_files:
        print(f"在目录 {dir_path} 中未找到PNG文件")
        return
    
    # 处理每个PNG文件
    for filename in png_files:
        try:
            # 打开图像
            img_path = os.path.join(dir_path, filename)
            with Image.open(img_path) as img:
                # 确保图像是灰度模式
                if img.mode != 'L':
                    img = img.convert('L')
                    print(f"已将 {filename} 转换为灰度模式")
                
                # 反色处理：黑色(0)变白色(255)，白色(255)变黑色(0)
                # 对于灰度图像，每个像素值在0-255之间，反色公式为 255 - 像素值
                inverted_img = Image.eval(img, lambda x: 255 - x)
                
                # 保存处理后的图像（覆盖原文件）
                inverted_img.save(img_path)
                print(f"已处理并保存：{filename}")
                
        except Exception as e:
            print(f"处理 {filename} 时出错：{str(e)}")
    
    print("所有图像处理完成")

if __name__ == "__main__":
    invert_black_white_images()
    