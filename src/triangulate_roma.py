#!/usr/bin/env python3
"""
Dense per-frame triangulation using RoMa feature matching.

Given multi-view images + known camera poses for each frame,
produce dense per-frame point clouds for FreeTimeGS initialization.

This replaces the need for pre-existing per-frame PLY files and
generates MUCH denser point clouds (100K-800K+ points/frame vs 32K).

Pipeline:
    For each frame t:
        1. Select camera pairs (all adjacent pairs + some skip pairs)
        2. Run RoMa dense matching on each pair
        3. Triangulate matched 2D points using known camera poses
        4. Filter by reprojection error and mask
        5. Merge all triangulated points for this frame
        6. Save: points3d_frame{t:06d}.npy, colors_frame{t:06d}.npy

Usage:
    python src/triangulate_roma.py \
        --data-dir /path/to/freetime/data \
        --output-dir /path/to/triangulation/output \
        --frame-start 0 --frame-end 60 \
        --masks-dir /path/to/masks  (optional)
"""

import argparse
import itertools
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm


def load_cameras_from_colmap(sparse_dir: str):
    """Load camera intrinsics and extrinsics from COLMAP binary files."""
    import sys
    # Use the repo's pycolmap
    from pycolmap import SceneManager

    mgr = SceneManager(sparse_dir + '/')
    mgr.load_cameras()
    mgr.load_images()

    cameras = {}
    for k in mgr.images:
        im = mgr.images[k]
        cam = mgr.cameras[im.camera_id]
        R = im.R()
        t = im.tvec

        # Intrinsics
        K = np.array([
            [cam.fx, 0, cam.cx],
            [0, cam.fy, cam.cy],
            [0, 0, 1]
        ])

        # Camera-to-world
        c2w = np.eye(4)
        c2w[:3, :3] = R.T
        c2w[:3, 3] = -R.T @ t

        # World-to-camera
        w2c = np.eye(4)
        w2c[:3, :3] = R
        w2c[:3, 3] = t

        cam_name = Path(im.name).stem
        cameras[cam_name] = {
            'K': K,
            'c2w': c2w,
            'w2c': w2c,
            'width': cam.width,
            'height': cam.height,
        }

    return cameras


def triangulate_pair(pts1, pts2, K1, K2, w2c1, w2c2):
    """Triangulate 2D correspondences from two views.

    Args:
        pts1: [N, 2] 2D points in image 1
        pts2: [N, 2] 2D points in image 2
        K1, K2: [3, 3] intrinsics
        w2c1, w2c2: [4, 4] world-to-camera transforms

    Returns:
        points3d: [N, 3] triangulated 3D points
        reproj_err: [N] reprojection error
    """
    # Projection matrices: P = K @ [R|t]
    P1 = K1 @ w2c1[:3]  # [3, 4]
    P2 = K2 @ w2c2[:3]  # [3, 4]

    # Triangulate using OpenCV
    pts1_h = pts1.T.astype(np.float64)  # [2, N]
    pts2_h = pts2.T.astype(np.float64)  # [2, N]

    points4d = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)  # [4, N]
    points3d = (points4d[:3] / points4d[3:]).T  # [N, 3]

    # Compute reprojection error
    pts3d_h = np.hstack([points3d, np.ones((len(points3d), 1))])  # [N, 4]

    proj1 = (P1 @ pts3d_h.T).T  # [N, 3]
    proj1 = proj1[:, :2] / proj1[:, 2:]
    err1 = np.linalg.norm(proj1 - pts1, axis=1)

    proj2 = (P2 @ pts3d_h.T).T  # [N, 3]
    proj2 = proj2[:, :2] / proj2[:, 2:]
    err2 = np.linalg.norm(proj2 - pts2, axis=1)

    reproj_err = (err1 + err2) / 2

    return points3d, reproj_err


def match_pair_roma(model, img1, img2, device='cuda', resolution=560):
    """Run RoMa dense matching between two images.

    Args:
        model: RoMa model
        img1, img2: [H, W, 3] uint8 images
        device: torch device
        resolution: matching resolution

    Returns:
        pts1, pts2: [N, 2] matched points in original resolution
        confidence: [N] match confidence
    """
    H1, W1 = img1.shape[:2]
    H2, W2 = img2.shape[:2]

    # RoMa expects PIL images or torch tensors
    from romatch.utils import get_tuple_transform_ops
    from PIL import Image

    pil1 = Image.fromarray(img1)
    pil2 = Image.fromarray(img2)

    # Run matching
    warp, certainty = model.match(pil1, pil2, device=device)

    # Extract good matches
    matches, batch_certainty = model.sample(warp, certainty, num=10000)

    # Convert from normalized [-1, 1] to pixel coordinates
    kp1 = matches[:, :2].cpu().numpy()
    kp2 = matches[:, 2:].cpu().numpy()

    kp1[:, 0] = (kp1[:, 0] + 1) / 2 * W1
    kp1[:, 1] = (kp1[:, 1] + 1) / 2 * H1
    kp2[:, 0] = (kp2[:, 0] + 1) / 2 * W2
    kp2[:, 1] = (kp2[:, 1] + 1) / 2 * H2

    confidence = batch_certainty.cpu().numpy()

    return kp1, kp2, confidence


def select_camera_pairs(cam_names, max_pairs=50):
    """Select camera pairs for triangulation.
    Use all pairs for small camera counts, sample for large counts.
    """
    n = len(cam_names)
    all_pairs = list(itertools.combinations(range(n), 2))

    if len(all_pairs) <= max_pairs:
        return all_pairs

    # Sample diverse pairs: adjacent + skip
    pairs = []
    for stride in [1, 2, 4, 8]:
        for i in range(n):
            j = (i + stride) % n
            pair = (min(i, j), max(i, j))
            if pair not in pairs:
                pairs.append(pair)

    # Fill remaining with random pairs
    remaining = [p for p in all_pairs if p not in pairs]
    np.random.shuffle(remaining)
    pairs.extend(remaining[:max(0, max_pairs - len(pairs))])

    return pairs[:max_pairs]


def triangulate_frame(
    frame_idx: int,
    cameras: dict,
    images_dir: str,
    frame_digits: int,
    frame_start_offset: int,
    roma_model,
    device: str = 'cuda',
    masks_dir: str = None,
    max_reproj_err: float = 2.0,
    min_confidence: float = 0.5,
):
    """Triangulate dense point cloud for a single frame.

    Returns:
        positions: [N, 3] 3D points
        colors: [N, 3] RGB (0-255)
    """
    cam_names = sorted(cameras.keys())
    frame_num = frame_idx + frame_start_offset
    frame_str = f"{frame_num:0{frame_digits}d}"

    # Load all images for this frame
    images = {}
    masks = {}
    for cam_name in cam_names:
        img_path = os.path.join(images_dir, cam_name, f"{frame_str}.png")
        if os.path.exists(img_path):
            images[cam_name] = cv2.imread(img_path)
            images[cam_name] = cv2.cvtColor(images[cam_name], cv2.COLOR_BGR2RGB)

            # Load mask if available
            if masks_dir:
                mask_path = os.path.join(masks_dir, cam_name, f"{frame_str}.png")
                if os.path.exists(mask_path):
                    masks[cam_name] = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if len(images) < 2:
        return None, None

    available_cams = sorted(images.keys())
    pairs = select_camera_pairs(available_cams, max_pairs=50)

    all_points = []
    all_colors = []

    for i, j in pairs:
        cam1, cam2 = available_cams[i], available_cams[j]
        img1, img2 = images[cam1], images[cam2]
        K1 = cameras[cam1]['K']
        K2 = cameras[cam2]['K']
        w2c1 = cameras[cam1]['w2c']
        w2c2 = cameras[cam2]['w2c']

        # Run RoMa matching
        try:
            pts1, pts2, conf = match_pair_roma(roma_model, img1, img2, device=device)
        except Exception as e:
            continue

        if len(pts1) < 10:
            continue

        # Filter by confidence
        good = conf > min_confidence
        pts1, pts2, conf = pts1[good], pts2[good], conf[good]

        if len(pts1) < 10:
            continue

        # Filter by mask (only keep matches where both points are in foreground)
        if cam1 in masks and cam2 in masks:
            m1 = masks[cam1]
            m2 = masks[cam2]
            h1, w1 = m1.shape
            h2, w2 = m2.shape

            px1 = np.clip(pts1[:, 0].astype(int), 0, w1 - 1)
            py1 = np.clip(pts1[:, 1].astype(int), 0, h1 - 1)
            px2 = np.clip(pts2[:, 0].astype(int), 0, w2 - 1)
            py2 = np.clip(pts2[:, 1].astype(int), 0, h2 - 1)

            fg = (m1[py1, px1] > 128) & (m2[py2, px2] > 128)
            pts1, pts2, conf = pts1[fg], pts2[fg], conf[fg]

        if len(pts1) < 10:
            continue

        # Triangulate
        points3d, reproj_err = triangulate_pair(pts1, pts2, K1, K2, w2c1, w2c2)

        # Filter by reprojection error
        good = reproj_err < max_reproj_err
        points3d = points3d[good]
        pts1_good = pts1[good]

        if len(points3d) == 0:
            continue

        # Get colors from image 1
        px = np.clip(pts1_good[:, 0].astype(int), 0, img1.shape[1] - 1)
        py = np.clip(pts1_good[:, 1].astype(int), 0, img1.shape[0] - 1)
        colors = img1[py, px].astype(np.float32)

        all_points.append(points3d.astype(np.float32))
        all_colors.append(colors)

    if not all_points:
        return None, None

    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    # Deduplicate via voxel grid
    voxel_size = 0.005  # 5mm voxels
    voxel_indices = np.floor(all_points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)

    all_points = all_points[unique_idx]
    all_colors = all_colors[unique_idx]

    return all_points, all_colors


def main():
    parser = argparse.ArgumentParser(description="Dense per-frame triangulation using RoMa")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="FreeTimeGS data directory (with images/ and sparse/0/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for per-frame NPY files")
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=60)
    parser.add_argument("--masks-dir", type=str, default=None,
                        help="Masks directory (masks/{cam}/{frame}.png)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-reproj-err", type=float, default=2.0)
    parser.add_argument("--roma-model", type=str, default="roma_outdoor",
                        choices=["roma_outdoor", "roma_indoor"])
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    sparse_dir = str(data_dir / "sparse" / "0")
    images_dir = str(data_dir / "images")
    masks_dir = args.masks_dir or str(data_dir / "masks")
    if not os.path.isdir(masks_dir):
        masks_dir = None

    # Load cameras
    print("Loading cameras from COLMAP...")
    cameras = load_cameras_from_colmap(sparse_dir)
    print(f"  {len(cameras)} cameras loaded")

    # Auto-detect frame format
    first_cam = sorted(cameras.keys())[0]
    first_cam_dir = os.path.join(images_dir, first_cam)
    first_file = sorted(os.listdir(first_cam_dir))[0]
    frame_digits = len(Path(first_file).stem)
    frame_start_offset = int(Path(first_file).stem)
    print(f"  Frame format: {frame_digits} digits, start_offset={frame_start_offset}")

    # Load RoMa model
    print(f"Loading RoMa model ({args.roma_model})...")
    import romatch
    roma_model = romatch.roma_outdoor(device=args.device)
    print("  RoMa loaded!")

    # Process each frame
    total_points = 0
    for frame_idx in tqdm(range(args.frame_start, args.frame_end), desc="Triangulating"):
        positions, colors = triangulate_frame(
            frame_idx=frame_idx,
            cameras=cameras,
            images_dir=images_dir,
            frame_digits=frame_digits,
            frame_start_offset=frame_start_offset,
            roma_model=roma_model,
            device=args.device,
            masks_dir=masks_dir,
            max_reproj_err=args.max_reproj_err,
        )

        if positions is None:
            print(f"  Frame {frame_idx}: FAILED (no triangulated points)")
            continue

        np.save(output_dir / f"points3d_frame{frame_idx:06d}.npy", positions)
        np.save(output_dir / f"colors_frame{frame_idx:06d}.npy", colors)

        total_points += len(positions)
        if frame_idx == args.frame_start or frame_idx % 10 == 0:
            print(f"  Frame {frame_idx}: {len(positions):,} points")

    avg = total_points // max(args.frame_end - args.frame_start, 1)
    print(f"\nDone! Total: {total_points:,} points, avg: {avg:,}/frame")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
