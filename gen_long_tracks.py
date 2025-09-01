import argparse
import glob
import os

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm

from cotracker.utils.visualizer import Visualizer
import torch
import torch.nn.functional as F

from cotracker.models.core.model_utils import smart_cat, get_points_on_a_grid
from cotracker.models.build_cotracker import build_cotracker
from tqdm import tqdm

DEFAULT_DEVICE = (
    # "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

class CoTrackerOnlinePredictor(torch.nn.Module):
    def init(
        self,
        checkpoint="./checkpoints/scaled_online.pth",
        offline=False,
        v2=False,
        window_len=16,
    ):
        super().init()
        self.v2 = v2
        self.support_grid_size = 6
        model = build_cotracker(checkpoint, v2=v2, offline=False, window_len=window_len)
        self.interp_shape = model.model_resolution
        self.step = model.window_len // 2
        self.model = model
        self.model.eval()
    
    @torch.no_grad()
    def forward(
        self,
        video_chunk,
        is_first_step: bool = False,
        queries: torch.Tensor = None,
        segm_mask: torch.Tensor = None,
        grid_size: int = 5,
        grid_query_frame: int = 0,
        add_support_grid=False,
    ):
        B, T, C, H, W = video_chunk.shape
        # Initialize online video processing and save queried points
        # This needs to be done before processing *each new video*
        if is_first_step:
            self.model.init_video_online_processing()
            if queries is not None:
                B, N, D = queries.shape
                self.N = N
                assert D == 3
                queries = queries.clone()
                queries[:, :, 1:] *= queries.new_tensor(
                    [
                        (self.interp_shape[1] - 1) / (W - 1),
                        (self.interp_shape[0] - 1) / (H - 1),
                    ]
                )
                if add_support_grid:
                    grid_pts = get_points_on_a_grid(
                        self.support_grid_size, self.interp_shape, device=video_chunk.device
                    )
                    grid_pts = torch.cat(
                        [torch.zeros_like(grid_pts[:, :, :1]), grid_pts], dim=2
                    )
                    queries = torch.cat([queries, grid_pts], dim=1)
            elif grid_size > 0:
                grid_pts = get_points_on_a_grid(
                    grid_size, self.interp_shape, device=video_chunk.device
                )
                
                if segm_mask is not None:
                    segm_mask = F.interpolate(segm_mask, tuple(self.interp_shape), mode="nearest")
                    point_mask = segm_mask[0, 0][
                        (grid_pts[0, :, 1]).round().long().cpu(),
                        (grid_pts[0, :, 0]).round().long().cpu(),
                    ].bool()
                    grid_pts = grid_pts[:, point_mask]
                    
                # self.N = grid_size**2
                self.N = grid_pts.shape[1]
                queries = torch.cat(
                    [torch.ones_like(grid_pts[:, :, :1]) * grid_query_frame, grid_pts],
                    dim=2,
                )
            self.queries = queries
            return (None, None)

        video_chunk = video_chunk.reshape(B * T, C, H, W)
        video_chunk = F.interpolate(
            video_chunk, tuple(self.interp_shape), mode="bilinear", align_corners=True
        )
        video_chunk = video_chunk.reshape(
            B, T, 3, self.interp_shape[0], self.interp_shape[1]
        )
        if self.v2:
            tracks, visibilities, __ = self.model(
                video=video_chunk, queries=self.queries, iters=6, is_online=True
            )
        else:
            tracks, visibilities, confidence, __ = self.model(
                video=video_chunk, queries=self.queries, iters=6, is_online=True
            )
        if add_support_grid:
            tracks = tracks[:,:,:self.N]
            visibilities = visibilities[:,:,:self.N]
            if not self.v2:
                confidence = confidence[:,:,:self.N]
            
        if not self.v2:
            visibilities = visibilities * confidence
        thr = 0.6
        return (
            tracks
            * tracks.new_tensor(
                [
                    (W - 1) / (self.interp_shape[1] - 1),
                    (H - 1) / (self.interp_shape[0] - 1),
                ]
            ),
            visibilities > thr,
        )
    
def read_video(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path)).transpose(2, 0, 1)[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def read_mask(folder_path):
    frame_paths = sorted(glob.glob(os.path.join(folder_path, "*")))
    video = np.concatenate([np.array(Image.open(frame_path))[None, None] for frame_path in frame_paths], axis=1)
    video = torch.from_numpy(video).float()
    return video

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="image dir")
    parser.add_argument("--mask_dir", type=str, required=True, help="mask dir")
    parser.add_argument("--out_dir", type=str, required=True, help="out dir")
    parser.add_argument("--is_static", action="store_true")
    parser.add_argument("--grid_size", type=int, default=100, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--backward_tracking",
        action="store_true",
        help="Compute tracks in both directions, not only forward",
    )
    args = parser.parse_args()
    folder_path = args.image_dir
    mask_dir = args.mask_dir
    frame_names = [
        os.path.basename(f) for f in sorted(glob.glob(os.path.join(folder_path, "*")))
    ]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "vis"), exist_ok=True)

    done = True
    for t in range(len(frame_names)):
        for j in range(len(frame_names)):
            name_t = os.path.splitext(frame_names[t])[0]
            name_j = os.path.splitext(frame_names[j])[0]
            out_path = f"{out_dir}/{name_t}_{name_j}.npy"
            if not os.path.exists(out_path):
                done = False
                break
    # print(f"{done}")
    if done:
        print("Already done")
        return

    ## Load model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEFAULT_DEVICE)
    # model = CoTrackerOnlinePredictor(checkpoint="cotracker_checkpoints/scaled_online.pth").to(DEFAULT_DEVICE)
    video = read_video(folder_path).to(DEFAULT_DEVICE)

    masks = read_mask(mask_dir).to(DEFAULT_DEVICE)
    
    masks[masks>0] = 1.
    if args.is_static:
        masks = 1.0 - masks
        
    model(video_chunk=video, is_first_step=True, grid_size=args.grid_size, segm_mask=masks[:,0].unsqueeze(1))  

    _, num_frames,_, height, width = video.shape
    vis = Visualizer(save_dir=os.path.join(out_dir, "vis"), pad_value=120, linewidth=3)

    for ind in tqdm(range(0, video.shape[1] - model.step, model.step)):
        pred_tracks, pred_visibility = model(
            video_chunk=video[:, ind : ind + model.step * 2],
            grid_size=args.grid_size,
            grid_query_frame=0,
        )  # B T N 2,  B T N 1

    track_name = "dynamic_track" if not args.is_static else "static_track"
    pred = torch.cat([pred_tracks, pred_visibility.unsqueeze(-1)], dim=-1)

    np.save(f"{out_dir}/{track_name}.npy", pred.cpu().numpy())
    vis.visualize(video, pred_tracks, pred_visibility, filename=f"{track_name}")

    # for t in tqdm(range(num_frames), desc="query frames"):
    #     name_t = os.path.splitext(frame_names[t])[0]
    #     file_matches = glob.glob(f"{out_dir}/{name_t}_*.npy")
    #     if len(file_matches) == num_frames:
    #         print(f"Already computed tracks with query {t} {name_t}")
    #         continue

    #     current_mask = masks[:,t].unsqueeze(1)
    #     start_pred = None
        
    #     for j in range(num_frames):
    #         if j > t:
    #             current_video = video[:,t:j+1]
    #         elif j < t:
    #             current_video = torch.flip(video[:,j:t+1], dims=(1,)) # reverse
    #         else:
    #             continue
    #             # current_video = video[:,t:t+1]
            
        
    #         pred_tracks, pred_visibility = model(
    #             current_video,
    #             grid_size=args.grid_size,
    #             grid_query_frame=0,
    #             backward_tracking=False,
    #             segm_mask=current_mask
    #         )
            

    #         pred = torch.cat([pred_tracks, pred_visibility.unsqueeze(-1)], dim=-1)
    #         current_pred = pred[0,-1]
    #         start_pred = pred[0,0]

    #         # save
    #         name_j = os.path.splitext(frame_names[j])[0]
    #         np.save(f"{out_dir}/{name_t}_{name_j}.npy", current_pred.cpu().numpy())
            
    #         # visualize
    #         # vis.visualize(current_video, pred_tracks, pred_visibility, filename=f"{name_t}_{name_j}")
            
    #     np.save(f"{out_dir}/{name_t}_{name_t}.npy", start_pred.cpu().numpy())

if __name__ == "__main__":
    main()