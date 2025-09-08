import os
import cv2
import numpy as np
from glob import glob

def get_sorted_images(path):
    """获取指定路径下按名称排序的所有PNG图片"""
    # 获取所有PNG文件
    image_paths = glob(os.path.join(path, "*.png"))
    # 按文件名排序
    image_paths.sort()
    # 确保有32张图片
    if len(image_paths) != 32:
        raise ValueError(f"路径 {path} 下应该有32张图片，但实际有 {len(image_paths)} 张")
    return image_paths

def stitch_and_create_video():
    # 定义输入路径
    input_paths = [
        "/share/czh/stereo4d/ski/images_2",
        "/share/czh/splinegs_0906/image3/train",
        "/share/czh/splinegs_0906/image3/test"
    ]
    
    # 输出视频路径
    output_video = "/share/czh/splinegs_0906/stereo_video.mp4"
    
    # 获取每个路径下的图片列表
    image_lists = [get_sorted_images(path) for path in input_paths]
    
    # 读取第一张图片获取尺寸信息
    first_images = [cv2.imread(images[0]) for images in image_lists]
    
    # 检查所有图片尺寸是否相同
    height, width = first_images[0].shape[:2]
    for img in first_images[1:]:
        if img.shape[:2] != (height, width):
            raise ValueError("所有输入图片必须具有相同的尺寸")
    
    # 计算拼接后图片的尺寸
    stitch_width = width * 3
    stitch_height = height
    
    # 定义视频编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4编码器
    out = cv2.VideoWriter(output_video, fourcc, 10.0, (stitch_width, stitch_height))
    
    # 处理每一帧
    for i in range(32):
        # 读取当前帧的三张图片
        images = [cv2.imread(image_list[i]) for image_list in image_lists]
        
        # 拼接图片（从左到右）
        stitched_image = np.hstack(images)
        
        # 写入视频
        out.write(stitched_image)
        
        # 打印进度
        if (i + 1) % 5 == 0:
            print(f"已处理 {i + 1}/32 帧")
    
    # 释放资源
    out.release()
    print(f"视频已成功保存至: {output_video}")

if __name__ == "__main__":
    stitch_and_create_video()
