import argparse
import numpy as np
import open3d as o3d

def load_kitti_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # (x,y,z,intensity)
    return pts[:, :3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="KITTI .bin file path")
    ap.add_argument("--voxel", type=float, default=0.2, help="voxel size (m)")
    ap.add_argument("--viz", action="store_true", help="show Open3D viewer")
    args = ap.parse_args()

    # 1) load
    xyz = load_kitti_bin(args.bin)
    if xyz.size == 0:
        print("[RESULT] empty point cloud")
        return

    # 2) point cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))

    # 3) downsample
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)

    # 4) simple forward crop (前方80m, 左右±40m, 高さ±3m)
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        print("[RESULT] no points after voxel")
        return

    mask = (
        (pts[:, 0] >= 0.0) & (pts[:, 0] <= 80.0) &
        (np.abs(pts[:, 1]) <= 40.0) &
        (pts[:, 2] >= -3.0) & (pts[:, 2] <= 3.0)
    )
    pts_roi = pts[mask]
    if len(pts_roi) == 0:
        print("[RESULT] no points in ROI")
        return

    # 5) nearest distance from LiDAR origin
    dists = np.linalg.norm(pts_roi, axis=1)
    i = int(np.argmin(dists))
    nearest_xyz = pts_roi[i]
    nearest_dist = float(dists[i])

    print("[RESULT] nearest point (approx):")
    print({
        "centroid": nearest_xyz.tolist(),
        "distance_m": nearest_dist
    })

    # === 追記 A: 原点→最近傍点の線分とマーカー ===
    origin_pts = np.array([[0.0, 0.0, 0.0], nearest_xyz])
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(origin_pts)
    line.lines = o3d.utility.Vector2iVector([[0, 1]])
    line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])  # 赤

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])
    T = np.eye(4); T[:3, 3] = nearest_xyz
    sphere.transform(T)
    # === 追記ここまで ===

    # 6) optional viz
    if args.viz:
        pcd_roi = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_roi))
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        T = np.eye(4); T[:3, 3] = nearest_xyz
        frame.transform(T)
        # 追記要素（line, sphere）を可視化に追加
        o3d.visualization.draw_geometries([pcd_roi, frame, line, sphere])


if __name__ == "__main__":
    main()
