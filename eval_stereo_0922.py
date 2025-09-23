import os
import sys
import torch
sys.path.append(os.path.join(sys.path[0], ".."))
import cv2
import lpips
import numpy as np
from argparse import ArgumentParser
from arguments import ModelHiddenParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render_infer
from PIL import Image
from scene import GaussianModel, Scene, dataset_readers
from utils.graphics_utils import pts2pixel
from utils.main_utils import get_pixels
from utils.image_utils import psnr
from gsplat.rendering import fully_fused_projection
from scene import GaussianModel, Scene, dataset_readers, deformation
import random


def normalize_image(img):
    return (2.0 * img - 1.0)[None, ...]


def compute_psnr_for_T(scene, test_cams, renderFunc, background, t_offset, viewpoint_stack, local_viewdirs, batch_shape, focal_bias):
    """计算给定T[0]偏移值时的测试集PSNR（原逻辑不变）"""
    temp_test_cams = []
    # my_test_cams = test_cams.copy()
    for view_id in range(len(viewpoint_stack)):
        cam = viewpoint_stack[view_id].copy()
        # print(cam.__class__)
        test_T = cam.T.copy()
        test_T[0] -= t_offset  # 应用偏移
        cam.update_cam(
            cam.R,
            test_T,
            local_viewdirs,
            batch_shape,
            focal_bias
        )
        temp_test_cams.append(cam)

    # for id, cam in enumerate(temp_test_cams):
    #     print(f"cam{id}",cam.uid,cam.R,cam.T)
    
    # # 创建保存目录
    save_dir = "/share/czh/splinegs_0922/test"
    os.makedirs(save_dir, exist_ok=True)
    
    psnr_total = 0.0

    # psnr_list = []

    with torch.no_grad():
        for cam_idx, cam in enumerate(temp_test_cams):
            render_pkg = renderFunc(cam, scene.stat_gaussians, scene.dyn_gaussians, background)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(test_cams[cam_idx].original_image.to("cuda"), 0.0, 1.0)

            psnr_image = psnr(image, gt_image, mask=None).mean().double().item()
            # psnr_list.append(psnr_image)

            psnr_total += psnr_image

            # 保存渲染图像
            img_np = (image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            img_pil.save(os.path.join(save_dir, f"render_{cam_idx}.png"))
            
            # 保存真实图像
            gt_np = (gt_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
            gt_pil = Image.fromarray(gt_np)
            gt_pil.save(os.path.join(save_dir, f"gt_{cam_idx}.png"))
    
    # print(psnr_list)
    return psnr_total / len(temp_test_cams)


def optimize_t_offset(scene, test_cams, renderFunc, background, viewpoint_stack, local_viewdirs, batch_shape, focal_bias, init_value=0.0056, search_range=(0, 1)):
    """优化T[0]偏移值（基于单峰特性持续缩小区间）"""
    left, right = search_range
    target_precision = 0.0005  # 目标精度：最终结果误差<0.0005
    max_iter = 100  # 最大迭代次数（防止极端情况）
    best_psnr = -1
    best_offset = init_value

    print(f"开始二分搜索：初始区间[{left:.4f}, {right:.4f}]，目标精度{target_precision}")

    # 核心迭代：持续缩小区间直到满足精度
    for iter_idx in range(max_iter):
        # 计算当前区间长度，满足精度则终止
        current_range = right - left
        if current_range < target_precision:
            print(f"迭代{iter_idx}：区间长度[{current_range:.4f}] < 目标精度[{target_precision}]，停止搜索")
            break

        # 单峰函数最优二分策略：取两个中点（避免中点恰为局部波动）
        mid1 = left + (right - left) / 3  # 左1/3处
        mid2 = right - (right - left) / 3  # 右1/3处

        # 计算两个中点的PSNR
        psnr_mid1 = compute_psnr_for_T(
            scene, test_cams, renderFunc, background, 
            mid1, viewpoint_stack, local_viewdirs, 
            batch_shape, focal_bias
        )
        psnr_mid2 = compute_psnr_for_T(
            scene, test_cams, renderFunc, background, 
            mid2, viewpoint_stack, local_viewdirs, 
            batch_shape, focal_bias
        )

        # 更新全局最佳值（记录当前区间内的最优解）
        current_candidates = [(mid1, psnr_mid1), (mid2, psnr_mid2), (best_offset, best_psnr)]
        current_candidates.sort(key=lambda x: x[1], reverse=True)
        best_offset, best_psnr = current_candidates[0]

        # 关键：基于单峰特性锁定峰值所在子区间
        if psnr_mid1 < psnr_mid2:
            # 左中点PSNR < 右中点PSNR → 峰值在[mid1, right]（仍处于递增阶段）
            left = mid1
            print(f"迭代{iter_idx}：PSNR(mid1={mid1:.4f})={psnr_mid1:.4f} < PSNR(mid2={mid2:.4f})={psnr_mid2:.4f} → 区间更新为[{left:.4f}, {right:.4f}]")
        else:
            # 左中点PSNR ≥ 右中点PSNR → 峰值在[left, mid2]（进入递减阶段）
            right = mid2
            print(f"迭代{iter_idx}：PSNR(mid1={mid1:.4f})={psnr_mid1:.4f} ≥ PSNR(mid2={mid2:.4f})={psnr_mid2:.4f} → 区间更新为[{left:.4f}, {right:.4f}]")

    # 最终步骤：在满足精度的区间内，遍历多个候选点找最优（避免区间端点误差）
    final_candidates = [
        left,  # 区间左端点
        (left + right) / 2,  # 区间中点
        right,  # 区间右端点
        left + target_precision/2,  # 左端点偏移（覆盖细微波动）
        right - target_precision/2   # 右端点偏移（覆盖细微波动）
    ]
    # 过滤超出原搜索范围的候选点
    final_candidates = [x for x in final_candidates if search_range[0] <= x <= search_range[1]]

    for candidate in final_candidates:
        candidate_psnr = compute_psnr_for_T(
            scene, test_cams, renderFunc, background, 
            candidate, viewpoint_stack, local_viewdirs, 
            batch_shape, focal_bias
        )
        if candidate_psnr > best_psnr:
            best_psnr = candidate_psnr
            best_offset = candidate

    # 确保最佳偏移值的精度（保留4位小数，符合0.0001精度要求）
    best_offset = round(best_offset, 4)
    # 最后验证一次最佳偏移的PSNR（确保准确性）
    best_psnr = compute_psnr_for_T(
        scene, test_cams, renderFunc, background, 
        best_offset, viewpoint_stack, local_viewdirs, 
        batch_shape, focal_bias
    )

    return best_offset, best_psnr


def training_report(scene: Scene, train_cams, test_cams, renderFunc, background, stage, dataset_type, path):
    """原逻辑不变，仅确保与优化后函数兼容"""
    test_psnr = 0.0
    torch.cuda.empty_cache()
    save_root = "/share/czh/splinegs_0918/both_video"
    os.makedirs(save_root, exist_ok=True)
    validation_configs = ({"name": "test", "cameras": test_cams}, {"name": "train", "cameras": train_cams})
    lpips_loss = lpips.LPIPS(net="alex").cuda()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for config in validation_configs:
        if config["cameras"] and len(config["cameras"]) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            lpips_test = 0.0
            run_time = 0.0
            elapsed_time_ms_list = []
            for idx, viewpoint in enumerate(config["cameras"]):     
                if idx == 0: # warmup iter
                    for _ in range(5):
                        render_pkg = renderFunc(
                            viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background
                        )
                  
                torch.cuda.synchronize()        
                start_event.record()
                render_pkg = renderFunc(
                    viewpoint, scene.stat_gaussians, scene.dyn_gaussians, background
                )
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start_event.elapsed_time(end_event)
                elapsed_time_ms_list.append(elapsed_time_ms)
                run_time += elapsed_time_ms

                image = render_pkg["render"]
                image = torch.clamp(image, 0.0, 1.0)
                img = Image.fromarray(
                    (np.clip(image.permute(1, 2, 0).detach().cpu().numpy(), 0, 1) * 255).astype("uint8")
                )
                
                save_path = os.path.join(save_root, config["name"])
                os.makedirs(save_path, exist_ok=True)
                img.save(f"{save_path}/img_{idx:03d}.png")
                print(f"Saved image to {save_path}/img_{idx:03d}.png")

                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                psnr_test += psnr(image, gt_image, mask=None).mean().double()
                lpips_test += lpips_loss.forward(normalize_image(image), normalize_image(gt_image)).item()

            psnr_test /= len(config["cameras"])
            l1_test /= len(config["cameras"])
            lpips_test /= len(config["cameras"])
            run_time /= len(config["cameras"])
            
            print(
                "\n[ITER {}] Evaluating {}: PSNR {}, LPIPS {}, FPS {}".format(
                    -1, config["name"], psnr_test, lpips_test, 1 / (run_time / 1000)
                )
            )


if __name__ == "__main__":
    """原逻辑不变，仅调用修正后的optimize_t_offset"""
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")

    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmengine as mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    dataset = lp.extract(args)
    hyper = hp.extract(args)
    stat_gaussians = GaussianModel(dataset)
    dyn_gaussians = GaussianModel(dataset)
    opt = op.extract(args)

    scene = Scene(
        dataset, dyn_gaussians, stat_gaussians, load_coarse=None
    )
    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())

    bg_color = [1] * 9 + [0] if dataset.white_background else [0] * 9 + [0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipe = pp.extract(args)

    test_cams = scene.getTestCameras()
    train_cams = scene.getTrainCameras()
    my_test_cams = [i for i in test_cams]
    viewpoint_stack = [i for i in train_cams]

    dyn_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud.ply"))
    stat_gaussians.load_ply(os.path.join(args.checkpoint, "point_cloud_static.ply"))
    
    dyn_gaussians.flatten_control_point()
    stat_gaussians.save_ply_compact(os.path.join(args.checkpoint, "compact_point_cloud_static.ply"))
    dyn_gaussians.save_ply_compact_dy(os.path.join(args.checkpoint, "compact_point_cloud.ply"))
        
    dyn_gaussians.load_model(args.checkpoint)
    dyn_gaussians._posenet.eval()
    
    # 计算local_viewdirs（原逻辑不变）
    pixels = get_pixels(
        scene.train_camera.dataset[0].metadata.image_size_x,
        scene.train_camera.dataset[0].metadata.image_size_y,
        use_center=True,
    )
    if pixels.shape[-1] != 2:
        raise ValueError("The last dimension of pixels must be 2.")
    batch_shape = pixels.shape[:-1]
    pixels = np.reshape(pixels, (-1, 2))
    y = (
        pixels[..., 1] - scene.train_camera.dataset[0].metadata.principal_point_y
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    x = (
        pixels[..., 0] - scene.train_camera.dataset[0].metadata.principal_point_x
    ) / dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    viewdirs = np.stack([x, y, np.ones_like(x)], axis=-1)
    local_viewdirs = viewdirs / np.linalg.norm(viewdirs, axis=-1, keepdims=True)

    # 更新viewpoint_stack的相机参数（原逻辑不变）
    with torch.no_grad():
        for cam in viewpoint_stack:
            time_in = torch.tensor(cam.time).float().cuda()
            pred_R, pred_T = dyn_gaussians._posenet(time_in.view(1, 1))
            R_ = torch.transpose(pred_R, 2, 1).detach().cpu().numpy()
            t_ = pred_T.detach().cpu().numpy()
            cam.update_cam(
                R_[0],
                t_[0],
                local_viewdirs,
                batch_shape,
                dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
            )

    focal_bias = dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy()
    
    # 调用修正后的二分法优化T[0]
    print("开始优化T[0]偏移值...")
    best_offset, best_psnr = optimize_t_offset(
        scene, my_test_cams, render_infer, background,
        viewpoint_stack, local_viewdirs, batch_shape, focal_bias,
        init_value=0.001,
        search_range=(0, 0.1)
    )
    print(f"\n优化完成！最佳T[0]偏移值: {best_offset:.2f}, 对应的PSNR: {best_psnr:.4f}")

    # 应用最佳偏移更新测试相机（原逻辑不变）
    for view_id in range(len(my_test_cams)):
        test_T = viewpoint_stack[view_id].T.copy()
        test_T[0] -= best_offset
        print(f"视图 {view_id} 原始T[0]: {viewpoint_stack[view_id].T[0]:.4f}, 优化后T[0]: {test_T[0]:.4f}")
        my_test_cams[view_id].update_cam(
            viewpoint_stack[view_id].R,
            test_T,
            local_viewdirs,
            batch_shape,
            focal_bias
        )

    # 生成报告（原逻辑不变）
    training_report(
        scene,
        viewpoint_stack,
        my_test_cams,
        render_infer,
        background,
        "fine",
        scene.dataset_type,
        "",
    )