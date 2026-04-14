"""
Point tracking module for extracting and tracking CoTracker query points in videos.

This module provides functionality to:
- Initialize CoTracker queries from grids, motion-guided random samples, or custom points
- Track points across video frames using CoTracker
- Save tracking results and generate visualizations
"""
import sys
import os
import time
import random
import argparse
import pickle
import json
import torch
import numpy as np
import cv2
from einops import rearrange
import pandas as pd
from PIL import Image
from utils import save_video
from new_video_loader import load_video_pyvideo_reader
from omni_vis import vis_trail, vis_trail_middle_frame

# set seeds
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

BASE_PATH = '/fs/cfar-projects/actionloc/camera_ready/tats_v2/dumps'

os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
# pylint: disable=redefined-outer-name


def _extract_state_dict(ckpt_obj):
    """Extract a model state dict from common checkpoint formats."""
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj
    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
            return ckpt_obj[key]
    return ckpt_obj


def check_columns_in_df(df):
    """Check if the dataframe has the required columns.

    Args:
        df (pd.DataFrame): Dataframe to check

    Raises:
        ValueError: If the dataframe does not have the required columns
    """
    required_columns = ['video_path', 'dataset']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in the dataframe")


def resolve_base_feat_path(base_feat_path):
    """Resolve a writable feature dump root.

    Falls back to local project directory when the configured path is not writable.
    """
    try:
        os.makedirs(base_feat_path, exist_ok=True)
        test_file = os.path.join(base_feat_path, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("ok")
        os.remove(test_file)
        return base_feat_path
    except OSError:
        local_fallback = os.path.join(os.getcwd(), "feat_dumps", "camera_ready", "tats_v2")
        os.makedirs(local_fallback, exist_ok=True)
        print(f"base_feat_path not writable: {base_feat_path}")
        print(f"Falling back to local writable path: {local_fallback}")
        return local_fallback


def load_custom_queries_from_json(json_path, num_frames, height, width, device):
    """Load custom query points from JSON and return CoTracker query tensor."""
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        points = raw.get("points", [])
    else:
        points = raw

    if not isinstance(points, list) or len(points) == 0:
        raise ValueError(
            f"No valid points found in {json_path}. Expected non-empty list or {{\"points\": [...]}}."
        )

    query_rows = []
    query_ids = []
    for idx, p in enumerate(points):
        if isinstance(p, dict):
            frame = p.get("frame", 0)
            x = p.get("x")
            y = p.get("y")
            pid = p.get("id", idx)
        elif isinstance(p, (list, tuple)) and len(p) >= 3:
            frame, x, y = p[0], p[1], p[2]
            pid = p[3] if len(p) > 3 else idx
        else:
            raise ValueError(
                "Each point must be dict like {frame,x,y[,id]} or list like [frame,x,y[,id]]."
            )

        if x is None or y is None:
            raise ValueError("Each custom point must include x and y.")

        frame = int(np.clip(int(round(float(frame))), 0, max(0, num_frames - 1)))
        x = float(np.clip(float(x), 0.0, max(0.0, width - 1.0)))
        y = float(np.clip(float(y), 0.0, max(0.0, height - 1.0)))
        query_rows.append([float(frame), x, y])
        query_ids.append(int(pid))

    query_np = np.asarray(query_rows, dtype=np.float32)
    query_ids_np = np.asarray(query_ids, dtype=np.int64)
    query_tensor = torch.from_numpy(query_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    return query_tensor, query_ids_np


def sample_uniform_spatiotemporal_queries(num_frames, height, width, num_points, device, seed=1234):
    """Uniformly sample query points from video spatiotemporal volume."""
    if num_points <= 0:
        raise ValueError("num_points must be > 0 for random spatiotemporal queries.")
    rng = np.random.default_rng(seed)
    t = rng.integers(low=0, high=max(1, num_frames), size=(num_points, 1), dtype=np.int64)
    x = rng.uniform(low=0.0, high=max(1.0, float(width)), size=(num_points, 1)).astype(np.float32)
    y = rng.uniform(low=0.0, high=max(1.0, float(height)), size=(num_points, 1)).astype(np.float32)
    x = np.clip(x, 0.0, max(0.0, width - 1.0))
    y = np.clip(y, 0.0, max(0.0, height - 1.0))
    query_np = np.concatenate([t.astype(np.float32), x, y], axis=1)
    query_ids = np.arange(num_points, dtype=np.int64)
    query_tensor = torch.from_numpy(query_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    return query_tensor, query_ids


def sample_motion_guided_spatiotemporal_queries(
    video_tensor,
    num_points,
    device,
    seed=1234,
    motion_percentile=80.0,
    motion_power=2.0,
    uniform_ratio=0.15,
    compensate_camera_motion=True,
    max_corners=400,
    motion_map_type="residual_diff",
):
    """Sample query points biased toward high-motion regions in (t,x,y)."""
    if num_points <= 0:
        raise ValueError("num_points must be > 0 for random spatiotemporal queries.")

    rng = np.random.default_rng(seed)
    vt = video_tensor.detach().float().cpu().numpy()  # (B, T, C, H, W)
    frames = vt[0]
    num_frames, _, height, width = frames.shape
    gray = frames.mean(axis=1)  # (T, H, W)

    if num_frames <= 1:
        return sample_uniform_spatiotemporal_queries(
            num_frames=num_frames,
            height=height,
            width=width,
            num_points=num_points,
            device=device,
            seed=seed,
        )

    # Residual motion map M_t.
    motion_list = []
    for t in range(1, num_frames):
        prev = gray[t - 1].astype(np.uint8)
        curr = gray[t].astype(np.uint8)
        flow = None
        if motion_map_type == "residual_flow":
            flow = cv2.calcOpticalFlowFarneback(
                prev,
                curr,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )  # (H, W, 2)

        gdx, gdy = 0.0, 0.0
        if compensate_camera_motion:
            corners = cv2.goodFeaturesToTrack(
                prev,
                maxCorners=int(max(20, max_corners)),
                qualityLevel=0.01,
                minDistance=5,
                blockSize=5,
            )
            if corners is not None and len(corners) >= 8:
                next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev, curr, corners, None)
                if next_pts is not None and status is not None:
                    valid = status.reshape(-1) > 0
                    if valid.sum() >= 8:
                        sparse_flow = next_pts[valid] - corners[valid]
                        gdx = float(np.median(sparse_flow[:, 0, 0]))
                        gdy = float(np.median(sparse_flow[:, 0, 1]))

        if motion_map_type == "residual_flow":
            if compensate_camera_motion:
                flow[..., 0] -= gdx
                flow[..., 1] -= gdy
            motion_t = np.sqrt(flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1])
        else:
            if compensate_camera_motion:
                M = np.array([[1.0, 0.0, gdx], [0.0, 1.0, gdy]], dtype=np.float32)
                prev_aligned = cv2.warpAffine(
                    prev,
                    M,
                    (prev.shape[1], prev.shape[0]),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                motion_t = np.abs(curr.astype(np.float32) - prev_aligned.astype(np.float32))
            else:
                motion_t = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
        motion_list.append(motion_t.astype(np.float32))

    diff = np.stack(motion_list, axis=0)  # (T-1, H, W), residual motion map
    motion = np.zeros_like(gray, dtype=np.float32)
    motion[0] = diff[0]
    motion[1:] = diff

    motion_percentile = float(np.clip(motion_percentile, 0.0, 100.0))
    uniform_ratio = float(np.clip(uniform_ratio, 0.0, 1.0))
    motion_power = max(1e-6, float(motion_power))
    threshold = np.percentile(motion, motion_percentile)
    motion_mask = motion >= threshold
    motion_weights = np.where(motion_mask, motion, 0.0).astype(np.float64)
    motion_weights = np.power(motion_weights + 1e-8, motion_power)
    weight_sum = float(motion_weights.sum())

    n_motion = int(round(num_points * (1.0 - uniform_ratio)))
    n_motion = max(0, min(num_points, n_motion))
    n_uniform = num_points - n_motion

    query_rows = []
    if n_motion > 0:
        if weight_sum <= 0.0:
            t_idx = rng.integers(0, max(1, num_frames), size=n_motion)
            y_idx = rng.integers(0, max(1, height), size=n_motion)
            x_idx = rng.integers(0, max(1, width), size=n_motion)
        else:
            probs = (motion_weights.reshape(-1) / weight_sum).astype(np.float64)
            flat_idx = rng.choice(probs.shape[0], size=n_motion, replace=True, p=probs)
            t_idx = flat_idx // (height * width)
            rem = flat_idx % (height * width)
            y_idx = rem // width
            x_idx = rem % width
        query_rows.append(np.stack([t_idx, x_idx, y_idx], axis=1).astype(np.float32))

    if n_uniform > 0:
        t_idx = rng.integers(0, max(1, num_frames), size=n_uniform)
        x_idx = rng.uniform(0.0, max(1.0, float(width)), size=n_uniform)
        y_idx = rng.uniform(0.0, max(1.0, float(height)), size=n_uniform)
        x_idx = np.clip(x_idx, 0.0, max(0.0, width - 1.0))
        y_idx = np.clip(y_idx, 0.0, max(0.0, height - 1.0))
        query_rows.append(np.stack([t_idx, x_idx, y_idx], axis=1).astype(np.float32))

    query_np = np.concatenate(query_rows, axis=0)
    rng.shuffle(query_np)
    query_ids = np.arange(query_np.shape[0], dtype=np.int64)
    query_tensor = torch.from_numpy(query_np).unsqueeze(0).to(device=device, dtype=torch.float32)
    return query_tensor, query_ids


def extract_points(args, cotracker, video_path, ds_dump_path, custom_fps=None):
    """Extract points from a video and save them to a pickle file.

    Args:
        args (argparse.Namespace): Arguments
        cotracker (torch.nn.Module): Cotracker model
        video_path (str): Path to the video
        ds_dump_path (str): Path to the directory where the pickle file will be saved
        custom_fps (int): Custom fps to use for the video if video duration > 90s

    Returns:
        bool: True if the points were extracted, False otherwise
    """
    vid_name = video_path.split('/')[-1].split('.')[0]
    feat_dump_path = os.path.join(ds_dump_path, 'feat_dump', f'{vid_name}.pkl')
    vis_dump_path = os.path.join(ds_dump_path, 'vis_dump', f'{vid_name}.png')
    gif_dump_path = os.path.join(ds_dump_path, 'gif_dump', f'{vid_name}.gif')
    vis_frames_dump_dir = os.path.join(ds_dump_path, 'vis_frames_dump', vid_name)
    if os.path.exists(feat_dump_path) and not args.rerun:
        return True

    custom_only = bool(args.custom_queries_json) and bool(args.custom_queries_only)

    _, video, _ = load_video_pyvideo_reader(video_path, return_tensor=True, use_float=True,
                             device=args.device, sample_all_frames=True,
                             fps=custom_fps if custom_fps is not None else args.fps)  # B T C H W
    _, num_frames_full, _, h_full, w_full = video.shape
    custom_queries = None
    custom_cluster_ids = None
    random_queries = None
    random_cluster_ids = None
    if args.custom_queries_json:
        custom_queries, custom_cluster_ids = load_custom_queries_from_json(
            json_path=args.custom_queries_json,
            num_frames=num_frames_full,
            height=h_full,
            width=w_full,
            device=args.device,
        )
    if args.random_st_queries:
        if args.random_query_mode == "uniform":
            random_queries, random_cluster_ids = sample_uniform_spatiotemporal_queries(
                num_frames=num_frames_full,
                height=h_full,
                width=w_full,
                num_points=args.num_random_st_queries,
                device=args.device,
                seed=args.random_query_seed,
            )
        else:
            random_queries, random_cluster_ids = sample_motion_guided_spatiotemporal_queries(
                video_tensor=video,
                num_points=args.num_random_st_queries,
                device=args.device,
                seed=args.random_query_seed,
                motion_percentile=args.motion_percentile,
                motion_power=args.motion_power,
                uniform_ratio=args.motion_uniform_ratio,
                compensate_camera_motion=args.compensate_camera_motion,
                max_corners=args.cam_motion_max_corners,
                motion_map_type=args.motion_map_type,
            )
    if args.debug_mode:
        time_start = time.time()
    grid_queries = None
    if args.use_grid and args.grid_query_stride_frames > 0:
        grid_queries = build_grid_queries_multi_frame(
            video_tensor=video,
            grid_size=args.cotracker_grid_size,
            stride_frames=args.grid_query_stride_frames,
            device=args.device,
        )

    pure_grid_mode = (
        args.use_grid
        and args.grid_query_stride_frames <= 0
        and custom_queries is None
        and random_queries is None
        and (not custom_only)
    )

    if pure_grid_mode:
        pred_tracks, pred_visibility = cotracker(
            video, grid_size=args.cotracker_grid_size,
            queries=None, backward_tracking=False)
    else:
        all_queries = []
        all_cluster_ids = []
        cluster_id_offset = 0
        if args.use_grid and (not custom_only):
            if grid_queries is None:
                grid_queries = build_grid_queries_multi_frame(
                    video_tensor=video,
                    grid_size=args.cotracker_grid_size,
                    stride_frames=0,
                    device=args.device,
                )
            all_queries.append(grid_queries)
            num_grid = grid_queries.shape[1]
            grid_ids = np.arange(cluster_id_offset, cluster_id_offset + num_grid, dtype=np.int64)
            all_cluster_ids.append(grid_ids)
            cluster_id_offset += num_grid
        if custom_queries is not None:
            all_queries.append(custom_queries)
            custom_ids = np.array(custom_cluster_ids, dtype=np.int64) + cluster_id_offset
            all_cluster_ids.append(custom_ids)
            if len(custom_ids) > 0:
                cluster_id_offset = int(np.max(custom_ids)) + 1
        if random_queries is not None:
            all_queries.append(random_queries)
            random_ids = np.array(random_cluster_ids, dtype=np.int64) + cluster_id_offset
            all_cluster_ids.append(random_ids)
        if not all_queries:
            raise ValueError(
                "No query points found. Enable --use_grid, --random_st_queries, "
                "or provide --custom_queries_json."
            )
        merged_queries = torch.cat(all_queries, dim=1)
        merged_cluster_ids = np.concatenate(all_cluster_ids, axis=0)
        if args.query_dedup_tol > 0:
            merged_queries, merged_cluster_ids = deduplicate_queries_by_frame_xy(
                merged_queries, merged_cluster_ids, xy_tol=args.query_dedup_tol
            )
        pred_tracks, pred_visibility = cotracker(
            video, queries=merged_queries, backward_tracking=True
        )
    if args.debug_mode:
        time_end = time.time()
        print(f"Time taken to run cotracker: {time_end - time_start} seconds")
    pred_tracks = pred_tracks.cpu().squeeze(0).numpy()
    pred_visibility = pred_visibility.cpu().squeeze(0).numpy()
    video = video.cpu().squeeze(0).numpy()
    video = rearrange(video, 't c h w -> t h w c')
    if pure_grid_mode:
        num_points = pred_tracks.shape[1]
        point_queries = np.zeros(num_points, dtype=np.int64)
        cluster_ids_all_frames = np.arange(num_points, dtype=np.int64)
    else:
        point_queries = merged_queries.cpu().squeeze(0).numpy()[:, 0]
        cluster_ids_all_frames = merged_cluster_ids
    if args.traj_dedup:
        pred_tracks, pred_visibility, cluster_ids_all_frames, point_queries = deduplicate_similar_trajectories(
            pred_tracks=pred_tracks,
            pred_visibility=pred_visibility,
            cluster_ids=cluster_ids_all_frames,
            point_queries=point_queries,
            dist_thresh=args.traj_dedup_dist_thresh,
            min_overlap=args.traj_dedup_min_overlap,
        )
    pt_obj_cluster_dict = {}

    dump_dict = {
        'pred_tracks': torch.tensor(pred_tracks).half(),
        'pred_visibility': torch.tensor(pred_visibility).bool(),
        'obj_ids': torch.tensor(cluster_ids_all_frames).long(),
        'point_queries': torch.tensor(point_queries).long(),
        'point_ids': torch.arange(len(point_queries)).long(),
        **pt_obj_cluster_dict
    }

    os.makedirs(os.path.dirname(feat_dump_path), exist_ok=True)
    pickle.dump(dump_dict, open(feat_dump_path, "wb"))
    torch.cuda.empty_cache()

    if args.debug_mode or args.make_vis:
        vis_img = vis_trail_middle_frame(
            video,
            pred_tracks,
            pred_visibility,
            kpts_queries=point_queries,
            cluster_ids=cluster_ids_all_frames,
            line_thickness=args.vis_line_thickness,
            require_visibility=not args.vis_ignore_visibility,
            anchor_frame=args.vis_anchor_frame,
        )
        os.makedirs(os.path.dirname(vis_dump_path), exist_ok=True)
        if isinstance(vis_img, Image.Image):
            vis_img.save(vis_dump_path)
        else:
            Image.fromarray(vis_img.astype(np.uint8)).save(vis_dump_path)

    if args.debug_mode or args.make_vis_all_frames:
        vis_frames = vis_trail(
            video,
            pred_tracks,
            pred_visibility,
            kpts_queries=point_queries,
            cluster_ids=cluster_ids_all_frames,
            line_thickness=args.vis_line_thickness,
        )
        step = max(1, int(args.vis_every_n_frames))
        selected_indices = list(range(0, len(vis_frames), step))
        vis_frames = [vis_frames[i] for i in selected_indices]
        if args.save_vis_frames:
            os.makedirs(vis_frames_dump_dir, exist_ok=True)
            for out_idx, frame_idx in enumerate(selected_indices):
                frame = vis_frames[out_idx]
                frame_path = os.path.join(vis_frames_dump_dir, f'frame_{frame_idx:05d}.png')
                if isinstance(frame, Image.Image):
                    frame.save(frame_path)
                else:
                    Image.fromarray(frame.astype(np.uint8)).save(frame_path)
        os.makedirs(os.path.dirname(gif_dump_path), exist_ok=True)
        save_video(vis_frames, gif_dump_path, fps=args.gif_fps)
    return True


def build_grid_queries_multi_frame(video_tensor, grid_size, stride_frames, device):
    """Build CoTracker queries on a regular grid for one or multiple query frames."""
    _, num_frames, _, height, width = video_tensor.shape
    if stride_frames is None or int(stride_frames) <= 0:
        query_frames = [0]
    else:
        stride_frames = max(1, int(stride_frames))
        query_frames = list(range(0, num_frames, stride_frames))

    ys = np.linspace(0, height - 1, grid_size, dtype=np.float32)
    xs = np.linspace(0, width - 1, grid_size, dtype=np.float32)
    mesh_y, mesh_x = np.meshgrid(ys, xs, indexing="ij")
    base_xy = np.stack([mesh_x.reshape(-1), mesh_y.reshape(-1)], axis=1)  # (N, 2) in x,y

    all_queries = []
    for qf in query_frames:
        qf_col = np.full((base_xy.shape[0], 1), qf, dtype=np.float32)
        all_queries.append(np.concatenate([qf_col, base_xy], axis=1))

    all_queries = np.concatenate(all_queries, axis=0)  # (M, 3)
    queries = torch.from_numpy(all_queries).unsqueeze(0).to(device=device, dtype=torch.float32)
    return queries


def deduplicate_queries_by_frame_xy(queries, cluster_ids, xy_tol=2.0):
    """Deduplicate query points by (frame, x, y) with a spatial tolerance."""
    if queries is None or queries.shape[1] <= 1:
        return queries, cluster_ids

    q_np = queries.squeeze(0).detach().cpu().numpy()  # (N, 3): frame, x, y
    cluster_ids = np.asarray(cluster_ids, dtype=np.int64)
    if len(cluster_ids) != q_np.shape[0]:
        raise ValueError("cluster_ids length does not match number of queries")

    tol = max(float(xy_tol), 1e-6)
    seen = {}
    keep = []
    for idx, (f, x, y) in enumerate(q_np):
        key = (int(round(float(f))), int(round(float(x) / tol)), int(round(float(y) / tol)))
        if key in seen:
            continue
        seen[key] = idx
        keep.append(idx)

    keep = np.asarray(keep, dtype=np.int64)
    q_np = q_np[keep]
    cluster_ids = cluster_ids[keep]
    q_t = torch.from_numpy(q_np).unsqueeze(0).to(device=queries.device, dtype=queries.dtype)
    return q_t, cluster_ids


def deduplicate_similar_trajectories(
    pred_tracks,
    pred_visibility,
    cluster_ids,
    point_queries,
    dist_thresh=2.0,
    min_overlap=5,
):
    """Remove near-duplicate trajectories using mean distance on overlapping visible frames."""
    if pred_tracks.ndim != 3:
        raise ValueError("pred_tracks must have shape (T, N, 2)")
    _, num_points, _ = pred_tracks.shape
    if num_points <= 1:
        return pred_tracks, pred_visibility, cluster_ids, point_queries

    vis = pred_visibility.astype(bool)
    keep = []
    dist_thresh = max(float(dist_thresh), 1e-6)
    min_overlap = max(int(min_overlap), 1)

    for i in range(num_points):
        duplicated = False
        for j in keep:
            overlap = vis[:, i] & vis[:, j]
            if int(overlap.sum()) < min_overlap:
                continue
            mean_dist = np.linalg.norm(pred_tracks[overlap, i] - pred_tracks[overlap, j], axis=1).mean()
            if mean_dist <= dist_thresh:
                duplicated = True
                break
        if not duplicated:
            keep.append(i)

    keep = np.asarray(keep, dtype=np.int64)
    return (
        pred_tracks[:, keep],
        pred_visibility[:, keep],
        np.asarray(cluster_ids, dtype=np.int64)[keep],
        np.asarray(point_queries)[keep],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_mode", action="store_true",
                        help="Enable debug mode")

    parser.add_argument("--use_grid", action="store_true",
                        help="Use grid")

    parser.add_argument("--cotracker_grid_size", type=int, default=16,
                        help="Cotracker grid size")
    parser.add_argument("--grid_query_stride_frames", type=int, default=0,
                        help="If >0 with --use_grid, add grid queries every N frames.")
    parser.add_argument("--query_dedup_tol", type=float, default=2.0,
                        help="Dedup tolerance (pixels) when merging query sources.")
    parser.add_argument("--traj_dedup", action="store_true",
                        help="Deduplicate near-identical trajectories after tracking.")
    parser.add_argument("--traj_dedup_dist_thresh", type=float, default=2.0,
                        help="Mean pixel distance threshold for trajectory dedup.")
    parser.add_argument("--traj_dedup_min_overlap", type=int, default=5,
                        help="Minimum jointly visible frames to compare trajectories.")

    parser.add_argument("--csv_path", type=str, default='sample.csv',
                        help='Path to csv file')

    parser.add_argument("--fps", type=int, default=None,
                        help="FPS for point tracking")

    parser.add_argument("--base_feat_path", type=str, default=BASE_PATH,
                        help="Base path for feature dumps")

    parser.add_argument("--make_vis", action="store_true",
                        help="Make static trajectory visualization images")
    parser.add_argument("--make_vis_all_frames", action="store_true",
                        help="Make per-frame trajectory GIF visualization.")
    parser.add_argument("--gif_fps", type=float, default=4.0,
                        help="Output GIF frame rate.")
    parser.add_argument("--cotracker_vis_only", action="store_true",
                        help="Shortcut mode: CoTracker-only grid tracking + GIF + last-frame trail image.")
    parser.add_argument("--custom_queries_json", type=str, default=None,
                        help="Path to JSON custom query points: list of {frame,x,y[,id]} "
                             "or [frame,x,y[,id]].")
    parser.add_argument("--custom_queries_only", action="store_true",
                        help="Use only custom query points (disable grid/random query generation).")
    parser.add_argument("--random_st_queries", action="store_true",
                        help="Initialize CoTracker queries by uniformly random sampling in (t,x,y).")
    parser.add_argument("--num_random_st_queries", type=int, default=384,
                        help="Number of uniformly random spatiotemporal query points.")
    parser.add_argument("--random_query_seed", type=int, default=1234,
                        help="Random seed for spatiotemporal query sampling.")
    parser.add_argument("--random_query_mode", type=str, default="motion",
                        choices=["motion", "uniform"],
                        help="Random query initializer: motion-guided or uniform.")
    parser.add_argument("--motion_percentile", type=float, default=80.0,
                        help="Keep top motion percentile when random_query_mode=motion.")
    parser.add_argument("--motion_power", type=float, default=2.0,
                        help="Sharpening power for motion weights when random_query_mode=motion.")
    parser.add_argument("--motion_uniform_ratio", type=float, default=0.15,
                        help="Fraction of uniformly random points mixed into motion-guided sampling.")
    parser.add_argument("--compensate_camera_motion", action="store_true",
                        help="Estimate and compensate global camera motion before motion-guided sampling.")
    parser.add_argument("--cam_motion_max_corners", type=int, default=400,
                        help="Maximum corners for global camera motion estimation.")
    parser.add_argument("--motion_map_type", type=str, default="residual_diff",
                        choices=["residual_diff", "residual_flow"],
                        help="Residual motion map type for motion-guided sampling.")
    parser.add_argument("--vis_line_thickness", type=int, default=1,
                        help="Line thickness for trajectory rendering.")
    parser.add_argument("--vis_every_n_frames", type=int, default=1,
                        help="Use every N-th frame for GIF visualization (1 = all frames).")
    parser.add_argument("--save_vis_frames", action="store_true",
                        help="Save sampled trajectory visualization frames as multiple PNG files.")
    parser.add_argument("--vis_ignore_visibility", action="store_true",
                        help="Connect trajectories even when points are marked invisible.")
    parser.add_argument("--vis_anchor_frame", type=str, default="first",
                        help="Anchor frame for single-image trajectory overlay: "
                             "first/middle/last/or frame index.")
    parser.add_argument("--rerun", action="store_true",
                        help="Rerun the point tracking")
    parser.add_argument("--cotracker_repo_path", type=str, default=None,
                        help="Local path to cloned facebookresearch/co-tracker repo. "
                             "If provided, load CoTracker from local source and skip "
                             "torch.hub download.")
    parser.add_argument("--cotracker_ckpt_path", type=str, default=None,
                        help="Local CoTracker checkpoint path (.pth/.pt). "
                             "If provided, load model with pretrained=False and "
                             "then load this checkpoint to avoid any download.")

    args = parser.parse_args()

    if args.cotracker_vis_only:
        args.use_grid = True
        args.make_vis = True
        args.make_vis_all_frames = True
        args.vis_anchor_frame = "last"

    if args.random_st_queries:
        args.use_grid = False
        args.make_vis = True
        args.make_vis_all_frames = True
        args.save_vis_frames = True
        args.traj_dedup = True
        if args.random_query_mode == "motion":
            args.compensate_camera_motion = True

    if args.custom_queries_only and not args.custom_queries_json:
        raise ValueError("--custom_queries_only requires --custom_queries_json")
    df = pd.read_csv(args.csv_path)
    check_columns_in_df(df)
    if args.debug_mode:
        df = df[df['video_name'] == '27k-12-1-2|P1|6116|6514.mp4']
        args.rerun = True
        args.make_vis = True
        # df = df.iloc[:1]  # just running on the first sample for debugging

    dump_parts = ["cotracker3"]
    if args.use_grid:
        dump_parts.append(f"grid{args.cotracker_grid_size}")
        if args.grid_query_stride_frames > 0:
            dump_parts.append(f"stride{args.grid_query_stride_frames}")
    if args.custom_queries_json:
        dump_parts.append("custom")
    if args.fps is not None:
        dump_parts.append(f"fps{args.fps}")
    if args.random_st_queries:
        dump_parts.append(f"rand{args.random_query_mode}_{args.num_random_st_queries}")
    dump_name = "_".join(dump_parts)

    args.base_feat_path = resolve_base_feat_path(args.base_feat_path)

    # base_featpath = '/fs/cfar-projects/actionloc/shirley/sam_based_debug/somethingv2'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(args, 'device', device)
    load_kwargs = {}
    if args.cotracker_ckpt_path is not None:
        load_kwargs["pretrained"] = False

    if args.cotracker_repo_path is not None:
        cotracker = torch.hub.load(
            args.cotracker_repo_path,
            "cotracker3_offline",
            source="local",
            **load_kwargs,
        ).to(device)
    else:
        cotracker = torch.hub.load(
            "facebookresearch/co-tracker",
            "cotracker3_offline",
            **load_kwargs,
        ).to(device)

    if args.cotracker_ckpt_path is not None:
        if not os.path.isfile(args.cotracker_ckpt_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {args.cotracker_ckpt_path}"
            )
        ckpt = torch.load(args.cotracker_ckpt_path, map_location="cpu")
        state_dict = _extract_state_dict(ckpt)
        cotracker.model.load_state_dict(state_dict)

    if not args.use_grid and not args.random_st_queries and not args.custom_queries_json:
        raise ValueError(
            "At least one query source is required: --use_grid, --random_st_queries, "
            "or --custom_queries_json."
        )

    for video_index, vid_info_row in df.iterrows():
        dataset = vid_info_row['dataset']
        video_path = vid_info_row['video_path']
        if 'duration' in vid_info_row:
            duration = vid_info_row['duration']
            if duration>90:
                custom_fps = 1
            else:
                custom_fps = None
        else:
            custom_fps = None
        ds_dump_path = os.path.join(args.base_feat_path, dump_name, dataset)
        extract_points(args, cotracker, video_path, ds_dump_path, custom_fps=custom_fps)
