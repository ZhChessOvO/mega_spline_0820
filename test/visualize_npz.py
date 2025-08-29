import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def visualize_npz(npz_path, output_dir):
    """
    可视化NPZ文件内容并保存结果
    
    参数:
        npz_path: NPZ文件的路径
        output_dir: 可视化结果保存目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载NPZ文件
        data = np.load(npz_path)
        print(f"成功加载NPZ文件，包含以下键: {list(data.keys())}")
        
        # 遍历所有数据项并可视化
        for key in data.keys():
            item = data[key]
            
            # 确保数据是合适的形状用于可视化
            # 处理可能的通道维度
            if len(item.shape) == 3:
                # 如果是通道在最后的RGB格式
                if item.shape[-1] == 3:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(item)
                    plt.title(f"Key: {key} (RGB Image)")
                    plt.axis('off')
                    save_path = os.path.join(output_dir, f"{key}_rgb.png")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                # 如果是单通道图像
                elif item.shape[0] == 1 or item.shape[-1] == 1:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(item.squeeze(), cmap='viridis')
                    plt.colorbar(label='Value')
                    plt.title(f"Key: {key} (Single Channel)")
                    plt.axis('off')
                    save_path = os.path.join(output_dir, f"{key}_single_channel.png")
                    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    print(f"无法识别的3D数据格式，键: {key}, 形状: {item.shape}")
            
            # 处理2D数据（如深度图）
            elif len(item.shape) == 2:
                plt.figure(figsize=(10, 8))
                plt.imshow(item, cmap='viridis')
                plt.colorbar(label='Value')
                plt.title(f"Key: {key} (2D Data)")
                plt.axis('off')
                save_path = os.path.join(output_dir, f"{key}_2d.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            # 处理其他可能的格式
            else:
                print(f"不支持的维度，键: {key}, 维度: {len(item.shape)}, 形状: {item.shape}")
                
        print(f"所有可视化结果已保存到: {output_dir}")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    # NPZ文件路径
    npz_file_path = "/share/czh/already_used/nvidia_megasam_preprocess/UniDepth/Balloon1/000.npz"
    
    # 输出目录
    output_directory = "image"
    
    # 执行可视化
    visualize_npz(npz_file_path, output_directory)
