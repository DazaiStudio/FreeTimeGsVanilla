#!/usr/bin/env python3
"""
Filter per-frame point clouds using masks to keep only foreground points.

For each frame, reprojects 3D points to all available cameras and checks
if the point falls within the foreground mask. Points visible in foreground
from at least `min_views` cameras are kept.

Usage:
    python src/filter_npz_with_mask.py \
        --input-dir /path/to/per_frame_npy \
        --masks-dir /path/to/masks \
        --cameras-dir /path/to/sparse/0 \
        --output-dir /path/to/filtered_npy \
        --frame-start 0 --frame-end 60
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def load_cameras_simple(sparse_dir):
    """Load cameras from COLMAP (simplified, uses text files if available)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]])
        P = K @ np.hstack([R, t.reshape(3, 1)])  # [3, 4] projection matrix

        cam_name = Path(im.name).stem
        cameras[cam_name] = {
            'K': K, 'R': R, 't': t, 'P': P,
            'width': cam.width, 'height': cam.height,
        }
    return cameras


def filter_points_with_masks(
    positions: np.ndarray,
    colors: np.ndarray,
    cameras: dict,
    masks_dir: str,
    frame_str: str,
    min_views: int = 2,
) -> tuple:
    """Filter 3D points by reprojecting to cameras and checking masks.

    Args:
        positions: [N, 3] 3D points
        colors: [N, 3] colors
        cameras: dict of camera info
        masks_dir: path to masks/{cam}/{frame}.png
        frame_str: frame filename (e.g., "0000001")
        min_views: minimum cameras where point must be in foreground

    Returns:
        filtered_positions, filtered_colors
    """
    N = len(positions)
    fg_votes = np.zeros(N, dtype=np.int32)

    pts_h = np.hstack([positions, np.ones((N, 1))])  # [N, 4]

    for cam_name, cam in cameras.items():
        mask_path = os.path.join(masks_dir, cam_name, f"{frame_str}.png")
        if not os.path.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        h, w = mask.shape

        # Project points to this camera
        proj = (cam['P'] @ pts_h.T).T  # [N, 3]
        z = proj[:, 2]
        valid_z = z > 0.01

        px = (proj[:, 0] / (z + 1e-8)).astype(np.int32)
        py = (proj[:, 1] / (z + 1e-8)).astype(np.int32)

        in_bounds = valid_z & (px >= 0) & (px < w) & (py >= 0) & (py < h)

        # Check mask for in-bounds points
        for i in np.where(in_bounds)[0]:
            if mask[py[i], px[i]] > 128:
                fg_votes[i] += 1

    # Keep points with enough foreground votes
    keep = fg_votes >= min_views
    return positions[keep], colors[keep]


def main():
    parser = argparse.ArgumentParser(description="Filter per-frame point clouds with masks")
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--masks-dir", type=str, required=True)
    parser.add_argument("--cameras-dir", type=str, required=True,
                        help="COLMAP sparse/0/ directory")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--frame-start", type=int, default=0)
    parser.add_argument("--frame-end", type=int, default=60)
    parser.add_argument("--min-views", type=int, default=2,
                        help="Min cameras where point must be foreground")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading cameras...")
    cameras = load_cameras_simple(args.cameras_dir)
    print(f"  {len(cameras)} cameras")

    # Detect frame naming
    npy_files = sorted(input_dir.glob("points3d_frame*.npy"))
    if not npy_files:
        print("No input NPY files found!")
        return

    total_before = 0
    total_after = 0

    for frame_idx in tqdm(range(args.frame_start, args.frame_end), desc="Filtering"):
        pts_path = input_dir / f"points3d_frame{frame_idx:06d}.npy"
        col_path = input_dir / f"colors_frame{frame_idx:06d}.npy"

        if not pts_path.exists():
            continue

        positions = np.load(pts_path)
        colors = np.load(col_path)
        total_before += len(positions)

        # Detect frame string for mask lookup
        # Try different frame naming conventions
        for frame_str in [f"{frame_idx:07d}", f"{frame_idx:06d}", f"{frame_idx+1:07d}"]:
            test_cam = sorted(cameras.keys())[0]
            test_path = os.path.join(args.masks_dir, test_cam, f"{frame_str}.png")
            if os.path.exists(test_path):
                break

        filtered_pos, filtered_col = filter_points_with_masks(
            positions, colors, cameras, args.masks_dir, frame_str,
            min_views=args.min_views,
        )

        total_after += len(filtered_pos)

        np.save(output_dir / f"points3d_frame{frame_idx:06d}.npy", filtered_pos)
        np.save(output_dir / f"colors_frame{frame_idx:06d}.npy", filtered_col)

        if frame_idx == args.frame_start:
            print(f"  Frame {frame_idx}: {len(positions)} -> {len(filtered_pos)} "
                  f"({100*len(filtered_pos)/len(positions):.1f}% kept)")

    pct = 100 * total_after / max(total_before, 1)
    print(f"\nTotal: {total_before:,} -> {total_after:,} ({pct:.1f}% kept)")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
