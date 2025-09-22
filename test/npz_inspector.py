import numpy as np

# 定义NPZ文件路径
file_path = "/home/czh/code/mega-sam/outputs_cvd/Balloon1_sgd_cvd_hr.npz"

try:
    # 加载NPZ文件
    npz_data = np.load(file_path)
    
    # 获取所有字段名称
    fields = npz_data.files
    
    print(f"文件 '{file_path}' 中包含的字段：")
    print(f"共 {len(fields)} 个字段\n")
    
    # 遍历每个字段并显示信息
    for field in fields:
        # 获取字段数据
        data = npz_data[field]
        # 计算元素数量
        # element_count = data.size
        # 显示信息
        print(f"字段名: {field}")
        print(f"  形状(shape): {data.shape}")
        # print(f"  元素数量: {element_count}")
        print(f"  数据类型: {data.dtype}")
        print("-" * 50)
        
    # 关闭文件
    npz_data.close()
    
except FileNotFoundError:
    print(f"错误: 文件 '{file_path}' 未找到。")
except Exception as e:
    print(f"读取文件时发生错误: {str(e)}")
    