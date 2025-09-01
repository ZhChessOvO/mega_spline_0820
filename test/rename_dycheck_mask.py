import os

def rename_motion_masks():
    """
    处理motion_masks目录下的文件，将0_00000.png.png重命名为000.png
    """
    # 定义目标目录路径
    dir_path = "/share/czh/dycheck_splinegs/block/motion_masks/"
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"错误：目录 {dir_path} 不存在")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(dir_path):
        # 检查是否是符合格式的图片文件，注意这里有双重.png扩展名
        if filename.endswith(".png.png") and "_" in filename:
            # 分割文件名和双重扩展名
            # 先去除一个.png，得到类似"0_00000.png"的名称
            name_part = filename[:-4]  # 移除最后一个.png
            # 再分割得到真正的文件名部分
            base_name, _ = os.path.splitext(name_part)
            
            # 分割前缀和数字部分
            if "_" in base_name:
                prefix, num_part = base_name.split("_", 1)
                
                # 提取数字并格式化
                try:
                    # 转换为整数后格式化为3位数字
                    num = int(num_part)
                    new_num = f"{num:03d}"
                    new_filename = f"{new_num}.png"
                    
                    # 构建完整路径
                    old_path = os.path.join(dir_path, filename)
                    new_path = os.path.join(dir_path, new_filename)
                    
                    # 重命名文件
                    os.rename(old_path, new_path)
                    print(f"重命名: {filename} -> {new_filename}")
                except ValueError:
                    print(f"跳过不符合格式的文件: {filename}")
            else:
                print(f"跳过不符合格式的文件: {filename}")
        else:
            # 可以选择打印或忽略不符合条件的文件
            # print(f"忽略文件: {filename}")
            pass

if __name__ == "__main__":
    print("开始处理motion_masks目录...")
    rename_motion_masks()
    print("motion_masks目录处理完成")
    