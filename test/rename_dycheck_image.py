import os

def rename_images_2():
    """处理/images_2/目录下的文件，将0_00000.png重命名为000.png"""
    dir_path = "/share/czh/dycheck_splinegs/block/images_2/"
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"错误：目录 {dir_path} 不存在")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(dir_path):
        # 检查是否是符合格式的图片文件
        if filename.endswith(".png") and "_" in filename:
            # 分割文件名和扩展名
            name_part, ext = os.path.splitext(filename)
            # 分割前缀和数字部分
            prefix, num_part = name_part.split("_", 1)
            
            # 提取后三位数字并格式化
            try:
                # 取最后三位数字并确保是3位格式
                num = int(num_part)
                new_num = f"{num:03d}"  # 格式化为3位数字，如1变为001
                new_filename = f"{new_num}{ext}"
                
                # 构建完整路径
                old_path = os.path.join(dir_path, filename)
                new_path = os.path.join(dir_path, new_filename)
                
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
            except ValueError:
                print(f"跳过不符合格式的文件: {filename}")

def rename_gt():
    """处理/gt/目录下的文件，将1_00000.png重命名为v000_t000.png"""
    dir_path = "/share/czh/dycheck_splinegs/block/gt/"
    
    # 检查目录是否存在
    if not os.path.exists(dir_path):
        print(f"错误：目录 {dir_path} 不存在")
        return
    
    # 遍历目录中的所有文件
    for filename in os.listdir(dir_path):
        # 检查是否是符合格式的图片文件
        if filename.endswith(".png") and "_" in filename:
            # 分割文件名和扩展名
            name_part, ext = os.path.splitext(filename)
            # 分割前缀和数字部分
            parts = name_part.split("_", 1)
            if len(parts) != 2:
                print(f"跳过不符合格式的文件: {filename}")
                continue
                
            prefix, num_part = parts
            
            # 格式化新文件名
            try:
                # 格式化前缀为v000格式
                v_num = int(prefix)
                formatted_v = "v000"
                
                # 格式化数字部分为t000格式
                t_num = int(num_part)
                formatted_t = f"t{t_num:03d}"
                
                # 构建新文件名
                new_filename = f"{formatted_v}_{formatted_t}{ext}"
                
                # 构建完整路径
                old_path = os.path.join(dir_path, filename)
                new_path = os.path.join(dir_path, new_filename)
                
                # 重命名文件
                os.rename(old_path, new_path)
                print(f"重命名: {filename} -> {new_filename}")
            except ValueError:
                print(f"跳过不符合格式的文件: {filename}")

if __name__ == "__main__":
    print("开始处理images_2目录...")
    rename_images_2()
    
    print("\n开始处理gt目录...")
    rename_gt()
    
    print("\n重命名操作完成")
