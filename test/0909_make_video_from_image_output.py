import cv2
import os
import numpy as np

def create_stereo_video():
    # 输入图片路径
    train_img_path = "/share/czh/splinegs_0908/ski/fine_render/train/images/"
    val_img_path = "/share/czh/splinegs_0908/ski/fine_render/val/images/"
    
    # 输出视频路径
    output_video_path = "/share/czh/splinegs_0908/ski/stereo_video2.mp4"
    
    # 视频参数
    fps = 10
    total_frames = 32
    
    # 获取第一张图片来确定尺寸
    first_train_img = cv2.imread(os.path.join(train_img_path, "000.jpg"))
    if first_train_img is None:
        raise FileNotFoundError("无法找到训练集的第一张图片")
    
    height, width, _ = first_train_img.shape
    
    # 定义编码器和创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 计算输出视频宽度：左半部分是1/3，右半部分是1/6，总宽度是width*(1/3 + 1/6) = width*1/2
    output_width = int(width * (1/3 + 1/6))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, height))
    
    for i in range(total_frames):
        # 构建文件名
        train_filename = f"{i:03d}.jpg"  # 000.jpg 到 031.jpg
        val_filename = f"v000_t{i:03d}.jpg"  # v000_t000.jpg 到 v000_t031.jpg
        
        # 读取图片
        train_img = cv2.imread(os.path.join(train_img_path, train_filename))
        val_img = cv2.imread(os.path.join(val_img_path, val_filename))
        
        if train_img is None or val_img is None:
            print(f"警告：无法读取第{i}帧的图片，跳过该帧")
            continue
        
        # 裁剪左侧部分（左1/3）
        left_part = train_img[:, 0:int(width/3), :]
        
        # 裁剪右侧部分（左1/6到左1/3）
        right_part = val_img[:, int(width/6):int(width/3), :]
        
        # 拼接图片
        combined = np.hstack((left_part, right_part))
        
        # 写入视频
        out.write(combined)
        print(f"已处理第{i+1}/{total_frames}帧")
    
    # 释放资源
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已保存至：{output_video_path}")

if __name__ == "__main__":
    create_stereo_video()
