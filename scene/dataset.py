import torch
from scene.cameras import Camera
from torch.utils.data import Dataset
from utils.general_utils import PILtoTorch
from utils.graphics_utils import focal2fov
import numpy as np  # 确保导入numpy


class FourDGSdataset(Dataset):
    def __init__(self, dataset, args, dataset_type):
        self.dataset = dataset
        self.args = args
        self.dataset_type = dataset_type
        self.override_R = None  # 用于存储覆盖的旋转矩阵列表
        self.override_T = None  # 用于存储覆盖的平移向量列表

    def set_poses(self, R, T):
        """
        设置覆盖的相机位姿，后续获取相机数据时会优先使用这些位姿
        Args:
            R: 旋转矩阵列表，每个元素为代表w2c的np array
            T: 平移向量列表，每个元素为代表w2c的np array
        """
        # 验证输入长度是否一致
        if len(R) != len(T):
            raise ValueError("R和T的长度必须一致")
        # 验证输入长度是否与数据集匹配
        if len(R) != len(self.dataset):
            raise ValueError(f"输入的位姿数量({len(R)})必须与数据集长度({len(self.dataset)})匹配")
        self.override_R = R
        self.override_T = T

    def __getitem__(self, index):
        if self.dataset_type != "PanopticSports":
            try:
                image, w2c, time = self.dataset[index]
                # 优先使用覆盖的位姿，如果存在
                if self.override_R is not None and self.override_T is not None:
                    R = self.override_R[index]
                    T = self.override_T[index]
                else:
                    R, T = w2c
                FovX = focal2fov(self.dataset.focal[0], image.shape[2])
                FovY = focal2fov(self.dataset.focal[0], image.shape[1])
                mask = None
            except:
                caminfo = self.dataset[index]
                image = PILtoTorch(caminfo.image, (caminfo.image.width, caminfo.image.height))
                # 优先使用覆盖的位姿，如果存在
                if self.override_R is not None and self.override_T is not None:
                    R = self.override_R[index]
                    T = self.override_T[index]
                else:
                    R = caminfo.R
                    T = caminfo.T
                FovX = caminfo.FovX
                FovY = caminfo.FovY
                time = caminfo.time
                mask = caminfo.mask

            return Camera(
                colmap_id=index,
                R=R,
                T=T,
                FoVx=FovX,
                FoVy=FovY,
                image=image,
                gt_alpha_mask=None,
                image_name=f"{caminfo.image_name}",
                uid=index,
                data_device=torch.device("cuda"),
                time=time,
                mask=mask,
                metadata=caminfo.metadata,
                normal=caminfo.normal,
                depth=caminfo.depth,
                max_time=caminfo.max_time,
                sem_mask=caminfo.sem_mask,
                fwd_flow=caminfo.fwd_flow,
                bwd_flow=caminfo.bwd_flow,
                fwd_flow_mask=caminfo.fwd_flow_mask,
                bwd_flow_mask=caminfo.bwd_flow_mask,
                instance_mask=caminfo.instance_mask,
                target_tracks=caminfo.target_tracks,
                target_visibility=caminfo.target_visibility,
                target_tracks_static=caminfo.target_tracks_static,
                target_visibility_static=caminfo.target_visibility_static,
            )
        else:
            return self.dataset[index]

    def __len__(self):
        return len(self.dataset)