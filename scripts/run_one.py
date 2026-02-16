import os
import argparse
import numpy as np
import open3d as o3d

from src.prepare_scene import prepare_scene

from src.infer import classify


def colorize(points, labels):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(float))

    colors = np.zeros((len(labels), 3), dtype=np.float32)
    colors[labels == 1] = [0, 1, 0]  # walk
    colors[labels == 2] = [0, 0, 1]  # support
    colors[labels == 3] = [1, 0, 0]  # obstruction

    pcd.colors = o3d.utility.Vector3dVector(colors.astype(float))
    return pcd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--data_root", default="data/replica")
    ap.add_argument("--cache_dir", default="cache/replica")
    args = ap.parse_args()

    mesh_path = os.path.join(args.data_root, args.scene, "mesh.ply")
    cache_path = os.path.join(args.cache_dir, f"{args.scene}.npz")

    if not os.path.exists(cache_path):
        prepare_scene(
            mesh_path=mesh_path,
            cache_path=cache_path,
            sample_points=200000,
            voxel_size=0.03,
            knn_normals=30
)


    data = np.load(cache_path)
    points = data["points"]
    height = data["height"]
    cos_theta = data["cos_theta"]

    labels = classify(height, cos_theta)

    pcd = colorize(points, labels)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
