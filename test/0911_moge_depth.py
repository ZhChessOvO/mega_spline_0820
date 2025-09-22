import cv2
import torch
import numpy as np
import os
from tqdm import tqdm
from moge.model.v2 import MoGeModel  # 使用MoGe-2

def process_scene(scene_name, model, base_dir):
    """处理单个场景的所有图像"""
    # 定义输入输出路径
    image_dir = os.path.join(base_dir, scene_name, "images_2")
    output_dir = os.path.join(base_dir, scene_name, "uni_depth")
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每张图像（000.png到011.png）
    for img_idx in range(32):  # 0到11
        img_name = f"{img_idx:03d}.png"  # 格式化文件名，确保3位数字
        img_path = os.path.join(image_dir, img_name)
        
        # 检查图像文件是否存在
        if not os.path.exists(img_path):
            print(f"警告: 图像文件 {img_path} 不存在，跳过处理")
            continue
        
        try:
            # 读取图像并转换为RGB
            input_image = cv2.imread(img_path)
            if input_image is None:
                print(f"警告: 无法读取图像 {img_path}，跳过处理")
                continue
                
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # 转换为模型需要的张量格式
            input_tensor = torch.tensor(
                input_image / 255, 
                dtype=torch.float32, 
                device=device
            ).permute(2, 0, 1)
            
            # 推理生成深度图
            output = model.infer(input_tensor)
            depth_map = output["depth"]  # 形状为(H, W)
            
            # 转换为numpy数组
            depth_np = depth_map.cpu().numpy()
            
            # 处理inf值：将inf设置为非inf值的最大值+1
            if np.isinf(depth_np).any():
                # 获取所有非inf值
                non_inf_values = depth_np[~np.isinf(depth_np)]
                if len(non_inf_values) > 0:
                    max_val = np.max(non_inf_values)
                    # 替换inf值
                    depth_np[np.isinf(depth_np)] = max_val + 1
                    print(f"图像 {img_path} 中存在inf值，已替换为 {max_val + 1}")
                else:
                    # 如果所有值都是inf，则设置为10（合理的默认值）
                    depth_np[np.isinf(depth_np)] = 10
                    print(f"图像 {img_path} 中所有值都是inf，已替换为10")
            
            # 转换为(H, W, 1)并保存为npy文件
            depth_np_reshaped = depth_np.reshape(depth_np.shape[0], depth_np.shape[1], 1)  # 重塑为(H, W, 1)
            
            npy_path = os.path.join(output_dir, f"{img_idx:03d}.npy")
            np.save(npy_path, depth_np_reshaped)
            
            # 可视化深度图并保存为png
            # 为了更好的可视化效果，我们对深度值进行归一化
            depth_normalized = (depth_np - np.min(depth_np)) / (np.max(depth_np) - np.min(depth_np) + 1e-8)
            depth_visual = (depth_normalized * 255).astype(np.uint8)  # 转换为0-255范围
            depth_visual_colored = cv2.applyColorMap(depth_visual, cv2.COLORMAP_VIRIDIS)  # 应用彩色映射
            
            png_path = os.path.join(output_dir, f"{img_idx:03d}.png")
            cv2.imwrite(png_path, depth_visual_colored)
            
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 配置参数
    base_directory = "/share/czh/stereo4d_moge"
    scenes = ["dog"]
    
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    if device.type == "cpu":
        print("警告: 未检测到GPU，将使用CPU进行处理，这可能会非常慢")
    
    # 加载MoGe模型
    print("加载MoGe模型...")
    try:
        model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(device)
        print("模型加载成功")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        exit(1)
    
    # 处理所有场景
    for scene in tqdm(scenes, desc="处理场景"):
        print(f"\n开始处理场景: {scene}")
        process_scene(scene, model, base_directory)
        print(f"场景 {scene} 处理完成")
    
    print("所有场景处理完毕")
    