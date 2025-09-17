from PIL import Image
import os
import numpy as np

def convert_mask(image_path):
    # 打开图像并转换为 RGB 模式
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)
    
    # 定义青色区域的判断条件（这里根据参考图，青色大致是 R 较低、G 较高、B 较高的情况，可根据实际微调）
    # 这里示例条件为：R < 100，G > 150，B > 150
    cyan_mask = (arr[:, :, 0] < 100) & (arr[:, :, 1] > 150) & (arr[:, :, 2] > 150)
    # 黑色区域判断条件：R、G、B 都接近 0
    black_mask = (arr[:, :, 0] < 20) & (arr[:, :, 1] < 20) & (arr[:, :, 2] < 20)
    
    # 创建灰度图数组，初始为黑色
    gray_arr = np.zeros(arr.shape[:2], dtype=np.uint8)
    # 青色区域设为白色
    gray_arr[cyan_mask] = 255
    # 黑色区域保持黑色（无需额外操作，因为初始就是 0）
    
    # 转换为灰度图并保存
    result_img = Image.fromarray(gray_arr, mode='L')
    result_img.save(image_path)
    print(f"已处理并保存：{image_path}")

# 图像所在目录
mask_directory = "/share/czh/stereo4d/cow/motion_masks/"

# 遍历目录下 000.png 到 031.png 文件
for i in range(32):
    filename = f"{i:03d}.png"
    file_path = os.path.join(mask_directory, filename)
    if os.path.exists(file_path):
        convert_mask(file_path)
    else:
        print(f"文件不存在：{file_path}")

print("所有文件处理完毕")