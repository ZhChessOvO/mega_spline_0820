import argparse
import json
import math
from PIL import Image
import os
import os.path as osp
from moviepy.video.io.VideoFileClip import VideoFileClip
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Lock  # 添加 Lock 导入

from absl import app
from absl import flags
from absl import logging
import cv2
import mediapy as media
import numpy as np
from functools import partial
import tqdm
from tqdm.contrib.concurrent import process_map
import utils
from tqdm import tqdm  # 添加 tqdm 导入

# 在文件顶部添加导入
from concurrent.futures import ProcessPoolExecutor

# 创建一个全局锁
lock = Lock()

def load_rgbd_cam_from_pkl(vid: str, root_dir: str, npz_folder: str, save_dir: str):
    """load rgb, and camera"""
    input_dict = {'left': {'camera': [], 'video': []}}
    # Load camera
    dp = utils.load_dataset_npz(osp.join(npz_folder, f'{vid}.npz'))
    
    timestamps = dp['timestamps']
    rectified2rig = dp['rectified2rig']
    meta_fov = dp['meta_fov']
    extrs_rectified = dp['extrs_rectified']
    track3d = dp['track3d']
    video_path = osp.join(
        root_dir,
        vid,
        vid + '-right_rectified.mp4',
    )
    rgbs = media.read_video(video_path)

    ## mkdir
    if not osp.exists(osp.join(save_dir, vid)):
        os.makedirs(osp.join(save_dir, vid))
    
    # 在循环中添加进度条
    final_timestamp = timestamps[-1]
    if not osp.exists(osp.join(save_dir, vid, f'{final_timestamp}_rgb.jpg')) and not osp.exists(osp.join(save_dir, vid, f'{final_timestamp}_extr.npz')) and not osp.exists(osp.join(save_dir, vid, f'{final_timestamp}_track3d.npz')):
        # 如果最后一个时间戳的文件不存在，则处理视频
        print(f"Processing video {vid} with {len(timestamps)} frames.")
    else:
        print(f"Video {vid} already processed, skipping.")
        return

    for i in tqdm(range(len(timestamps)), desc=f"Processing frames for {vid}"):
        timestamp = timestamps[i]
        extr = extrs_rectified[i]
        track3d_i = track3d[:, i, ...]
        rgb = rgbs[i]
    
        np.savez(osp.join(save_dir, vid, f'{timestamp}_extr.npz'), extr)
        np.savez(osp.join(save_dir, vid, f'{timestamp}_track3d.npz'), track3d_i)
        Image.fromarray(rgb).save(osp.join(save_dir, vid, f'{timestamp}_rgb.jpg'))
    
    np.save(osp.join(save_dir, vid, 'rectified2rig.npz'), rectified2rig)
    np.save(osp.join(save_dir, vid, 'meta_fov.npz'), meta_fov)


def process_video(vid, root_dir, npz_folder, save_dir):
    """单独处理一个视频的函数"""
    try:
        load_rgbd_cam_from_pkl(vid, root_dir, npz_folder, save_dir)
    except Exception as e:
        print(f"Error processing video {vid}: {e}")

def main():
    parser = argparse.ArgumentParser()
#   parser.add_argument('--vid', help='video id, in the format of <raw-video-id>_<timestamp>', type=str)
    parser.add_argument('--npz_folder', help='npz folder', type=str, default='/share/stereo4d_dataset/npz/stereo4d/train')
    parser.add_argument('--raw_video_folder', help='raw video folder', type=str, default='/share/stereo4d_dataset/raw')
    parser.add_argument('--output_folder', help='output folder', type=str, default='/share/czh/stereo4d_right/')
    parser.add_argument('--output_hfov', help='output horizontal fov', type=float, default=60)

    root_dir = '/share/stereo4d_dataset/processed/'
    save_dir = '/share/czh/stereo4d_right/'

    args = parser.parse_args()
    # vid_list = os.listdir(args.output_folder)
    vid_list = ["JwV2f29jdYY_36033333"]

     # 使用多进程处理，设置 max_workers=8
    worker = partial(process_video, root_dir=root_dir, npz_folder=args.npz_folder, save_dir=save_dir)

    # 使用多进程处理，添加进度条和错误处理
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        for vid in vid_list:
            futures.append(executor.submit(worker, vid))

if __name__ == '__main__':
    main()
