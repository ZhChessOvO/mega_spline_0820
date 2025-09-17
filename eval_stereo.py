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
import random
from tqdm import tqdm


def normalize_image(img):
    return (2.0 * img - 1.0)[None, ...]


def load_stereo_gt_images(gt_dir):
    """加载立体图像作为Ground Truth"""
    gt_list = []
    # 读取PNG和JPG格式的图像
    for ext in ['*.png', '*.jpg']:
        gt_list.extend(sorted(glob.glob(os.path.join(gt_dir, ext))))
    
    img_data = []
    for img_path in gt_list:
        image = cv2.imread(img_path)[..., ::-1]  # 转换为RGB
        h, w, _ = image.shape
        # 保持与渲染图像相同尺寸
        image = cv2.resize(image, (w, h))
        img_data.append(image)
    
    # 转换为Tensor并归一化
    img_data = torch.Tensor(np.array(img_data)).float().cuda() / 255.0  # [B, H, W, C]
    return img_data


def optimize_camera_poses(scene, test_cams, renderFunc, background, gt_images, num_steps=1000):
    """优化测试相机的平移向量以最大化PSNR"""
    # 确保测试相机数量与GT图像数量匹配
    assert len(test_cams) == len(gt_images), "测试相机数量与GT图像数量不匹配"
    
    # 初始化可优化的平移向量
    cam_translations = []
    for cam in test_cams:
        T = torch.tensor(cam.T, dtype=torch.float32, device="cuda", requires_grad=True)
        cam_translations.append(T)
    
    # 设置优化器
    optimizer = torch.optim.Adam(cam_translations, lr=1e-4)
    
    # 优化循环
    for step in tqdm(range(num_steps), desc="优化相机位姿"):
        optimizer.zero_grad()
        total_loss = 0.0
        psnr_values = []
        
        for i, (cam, T, gt_img) in enumerate(zip(test_cams, cam_translations, gt_images)):
            # 更新相机平移向量
            cam.T = T.detach().cpu().numpy()
            
            # 渲染当前相机视角的图像
            render_pkg = renderFunc(cam, scene.stat_gaussians, scene.dyn_gaussians, background)
            rendered_img = torch.clamp(render_pkg["render"], 0.0, 1.0)  # [C, H, W]
            
            # 转换为与GT相同的形状 [H, W, C]
            rendered_img = rendered_img.permute(1, 2, 0)
            
            # 计算L1损失（用于优化）
            loss = torch.abs(rendered_img - gt_img).mean()
            total_loss += loss
            
            # 计算PSNR（用于监控）
            mse = torch.nn.functional.mse_loss(rendered_img, gt_img)
            current_psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_values.append(current_psnr.item())
        
        # 反向传播和参数更新
        total_loss.backward()
        optimizer.step()
        
        # 打印优化进度
        if (step + 1) % 10 == 0:
            avg_psnr = sum(psnr_values) / len(psnr_values)
            print(f"优化步骤 {step+1}/{num_steps} - 平均PSNR: {avg_psnr:.2f} dB - 总损失: {total_loss.item():.6f}")
    
    # 更新相机最终位姿
    for cam, T in zip(test_cams, cam_translations):
        cam.T = T.detach().cpu().numpy()
    
    return test_cams


def training_report(scene: Scene, train_cams, test_cams, renderFunc, background, stage, dataset_type, path):
    test_psnr = 0.0
    torch.cuda.empty_cache()

    # 定义目标保存目录
    save_root = "/share/czh/splinegs_0909/ski_align_test"
    # 创建保存目录（如果不存在）
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
                
                # 构建保存路径（区分train和test，并保留原始索引）
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
    # 添加GT图像目录的命令行参数
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)

    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file",
    )
    parser.add_argument("--expname", type=str, default="")
    parser.add_argument("--configs", type=str, default="")
    parser.add_argument(
        "--stereo_gt_dir", type=str, required=True, 
        help="Directory containing stereo ground truth images"
    )
    parser.add_argument(
        "--optim_steps", type=int, default=1000, 
        help="Number of optimization steps for camera poses"
    )

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
    )  # for other datasets rather than iPhone dataset

    dyn_gaussians.create_pose_network(hyper, scene.getTrainCameras())  # pose network with instance scaling

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

    # 加载立体图像GT
    import glob
    gt_images = load_stereo_gt_images(args.stereo_gt_dir)
    
    # 初始化测试相机位姿（替换原来的test_T[0] -= 0.02）
    for view_id in range(len(my_test_cams)):
        # 使用训练相机的位姿作为初始值
        my_test_cams[view_id].update_cam(
            viewpoint_stack[view_id].R,
            viewpoint_stack[view_id].T,  # 初始平移向量
            local_viewdirs,
            batch_shape,
            dyn_gaussians._posenet.focal_bias.exp().detach().cpu().numpy(),
        )
    
    # 优化测试相机的平移向量
    optimized_test_cams = optimize_camera_poses(
        scene, 
        my_test_cams, 
        render_infer, 
        background, 
        gt_images,
        num_steps=args.optim_steps
    )

    # 保存优化后的结果
    training_report(
        scene,
        viewpoint_stack,
        optimized_test_cams,
        render_infer,
        background,
        "fine",
        scene.dataset_type,
        "",
    )