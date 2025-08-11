import argparse
import numpy as np
import open3d as o3d

# ---------- utils ----------
def load_kitti_bin(path: str) -> np.ndarray:
    pts = np.fromfile(path, dtype=np.float32).reshape(-1, 4)  # (x,y,z,intensity)
    return pts[:, :3]

def make_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def crop_forward(xyz: np.ndarray,
                 x_min=0.0, x_max=80.0, y_abs=40.0, z_min=-3.0, z_max=3.0) -> np.ndarray:
    m = (
        (xyz[:, 0] >= x_min) & (xyz[:, 0] <= x_max) &
        (np.abs(xyz[:, 1]) <= y_abs) &
        (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    )
    return xyz[m]

def remove_ground_ransac(pcd: o3d.geometry.PointCloud,
                         dist_thresh=0.2, ransac_n=3, num_iter=1000):
    plane_model, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iter)
    ground = pcd.select_by_index(inliers)
    non_ground = pcd.select_by_index(inliers, invert=True)
    return plane_model, ground, non_ground

def dbscan_labels(pcd: o3d.geometry.PointCloud, eps=0.6, min_points=20):
    return np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))

def cluster_centroids(xyz: np.ndarray, labels: np.ndarray):
    cents = []
    for k in np.unique(labels):
        if k == -1:
            continue
        pts = xyz[labels == k]
        if len(pts) == 0:
            continue
        cen = pts.mean(axis=0)
        ext = (pts.max(axis=0) - pts.min(axis=0))  # extent (x,y,z)
        cents.append((int(k), cen, int(len(pts)), ext))
    return cents

def l2(p): return float(np.linalg.norm(p, ord=2))

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="KITTI .bin file path")
    ap.add_argument("--voxel", type=float, default=0.2, help="voxel size (m)")
    ap.add_argument("--crop", action="store_true", help="apply forward ROI crop")
    ap.add_argument("--remove_ground", action="store_true", help="RANSAC ground removal")
    ap.add_argument("--cluster", action="store_true", help="DBSCAN clustering & nearest cluster")
    ap.add_argument("--viz", action="store_true", help="show Open3D viewer")
    ap.add_argument("--save_bev", action="store_true", help="save BEV PNG (bev_nearest.png)")
    args = ap.parse_args()

    # 1) load
    xyz = load_kitti_bin(args.bin)
    if xyz.size == 0:
        print("[RESULT] empty point cloud")
        return

    # 2) voxel
    pcd = make_pcd(xyz)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    xyz = np.asarray(pcd.points)
    if xyz.size == 0:
        print("[RESULT] no points after voxel")
        return

    # 3) ROI
    if args.crop:
        xyz = crop_forward(xyz)
        if len(xyz) == 0:
            print("[RESULT] no points in ROI")
            return
        pcd = make_pcd(xyz)

    # 4) ground removal (optional)
    plane_model = None
    if args.remove_ground:
        plane_model, ground, non_ground = remove_ground_ransac(pcd)
        pcd = non_ground
        xyz = np.asarray(pcd.points)
        if len(xyz) == 0:
            print("[RESULT] no points after ground removal")
            return

    # 5) target selection
    nearest_info = None
    if args.cluster:
        labels = dbscan_labels(pcd, eps=0.6, min_points=20)
        cents = cluster_centroids(xyz, labels)
        if cents:
            cents_sorted = sorted(cents, key=lambda t: l2(t[1]))
            cid, cxyz, cnt, ext = cents_sorted[0]
            # “局所信頼度”として近傍半径0.5m内の距離分散
            d = np.linalg.norm(xyz - cxyz, axis=1)
            local = xyz[d <= 0.5]
            local_std = float(np.std(np.linalg.norm(local, axis=1))) if len(local) > 3 else None
            nearest_info = {
                "mode": "cluster",
                "cluster_id": cid,
                "centroid": cxyz.tolist(),
                "distance_m": l2(cxyz),
                "points": cnt,
                "extent_m": ext.tolist(),
                "local_points": int(len(local)),
                "local_distance_std_m": local_std
            }
        else:
            print("[RESULT] clustering found no clusters; falling back to nearest point")
    if nearest_info is None:
        # nearest point fallback
        dists = np.linalg.norm(xyz, axis=1)
        i = int(np.argmin(dists))
        nearest_info = {
            "mode": "point",
            "cluster_id": None,
            "centroid": xyz[i].tolist(),
            "distance_m": float(dists[i]),
            "points": 1,
            "extent_m": None,
            "local_points": None,
            "local_distance_std_m": None
        }

    print("[RESULT] nearest target:")
    print(nearest_info)

    # 6) visualization
    if args.viz or args.save_bev:
        # line & sphere
        nearest_xyz = np.array(nearest_info["centroid"], dtype=float)
        pcd_disp = make_pcd(xyz)
        origin_pts = np.array([[0.0, 0.0, 0.0], nearest_xyz])
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector(origin_pts)
        line.lines = o3d.utility.Vector2iVector([[0, 1]])
        line.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 0.0]])
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])
        T = np.eye(4); T[:3, 3] = nearest_xyz
        sphere.transform(T)

        if args.viz:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            o3d.visualization.draw_geometries([pcd_disp, frame, line, sphere])

        if args.save_bev:
            # simple BEV save (requires matplotlib)
            try:
                import matplotlib.pyplot as plt
                xy = xyz[:, :2]
                plt.figure(figsize=(6, 6))
                plt.scatter(xy[:, 0], xy[:, 1], s=1)
                for r in [5, 10, 20, 30, 40, 60, 80]:
                    c = plt.Circle((0, 0), r, fill=False, linestyle='--', linewidth=0.5)
                    plt.gca().add_artist(c)
                plt.scatter([nearest_xyz[0]], [nearest_xyz[1]], s=40, marker='x')
                plt.axis('equal'); plt.grid(True)
                plt.xlabel("X [m]"); plt.ylabel("Y [m]")
                plt.title(f"Nearest {nearest_info['mode']}: {nearest_info['distance_m']:.2f} m")
                plt.tight_layout()
                plt.savefig("bev_nearest.png", dpi=200)
                plt.close()
                print("[RESULT] saved bev_nearest.png")
            except Exception as e:
                print(f"[WARN] BEV save failed: {e}. Install matplotlib if needed.")

if __name__ == "__main__":
    main()
