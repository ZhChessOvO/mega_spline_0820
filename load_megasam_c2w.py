import numpy as np

def load_megasam_c2w(npz_path, train_size=12, test_size=12):
    """
    加载npz文件中的相机位姿，并按指定大小分割训练集和测试集。
    
    参数:
        npz_path (str): npz文件的完整路径（包含相机位姿数据）
        train_size (int): 训练集样本数量（默认值：12）
        test_size (int): 测试集样本数量（默认值：12）
    
    返回:
        tuple: (train_list, test_list)
            - train_list (list): 训练集相机到世界的变换矩阵列表，每个元素是形状为(4, 4)的np.ndarray
            - test_list (list): 测试集相机到世界的变换矩阵列表，每个元素是形状为(4, 4)的np.ndarray
    """
    # 加载npz文件中的相机位姿
    with np.load(npz_path) as data:
        # 读取相机到世界的变换矩阵（默认键为"cam_c2w"，若实际键不同需修改）
        cam_c2w_all = data["cam_c2w"]

    # 分割训练集（前train_size个）和测试集（剩余部分）
    c2w_train = cam_c2w_all[:train_size]
    c2w_test = cam_c2w_all[test_size:]
    
    # 将numpy数组转换为列表格式
    train_c2w_list = [c2w_train[i] for i in range(train_size)]
    test_c2w_list = [c2w_test[i] for i in range(test_size)]
    
    return train_c2w_list, test_c2w_list


#  测试
if __name__ == "__main__":
    train_c2w, test_c2w = load_megasam_c2w("/home/czh/code/mega-sam/outputs/Balloon1_droid.npz")
    print("Train set camera to world matrices:")
    for mat in train_c2w:
        print(mat)
    print("Test set camera to world matrices:")
    for mat in test_c2w:
        print(mat)