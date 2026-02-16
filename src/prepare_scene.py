import os
import numpy as np
import open3d as o3d
import trimesh


def sample_points_trimesh(mesh_path: str, n_points: int) -> np.ndarray:
    tm = trimesh.load(mesh_path, force="mesh")

    if tm is None or tm.vertices is None or len(tm.vertices) == 0:
        raise ValueError(f"Falha ao ler mesh com trimesh: {mesh_path}")

    # Se vier como cena, tenta consolidar
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) == 0:
            raise ValueError(f"Scene trimesh sem geometria: {mesh_path}")
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))

    # Garante triangulação
    if hasattr(tm, "triangulate"):
        tm = tm.triangulate()

    if tm.faces is None or len(tm.faces) == 0:
        raise ValueError(f"Mesh sem faces: {mesh_path}")

    # Amostragem na superfície (retorna exatamente n_points, em geral)
    pts, _ = trimesh.sample.sample_surface(tm, n_points)
    pts = pts.astype(np.float32)
    return pts


def voxel_downsample(points: np.ndarray, voxel_size: float) -> np.ndarray:
    if voxel_size is None or voxel_size <= 0:
        return points

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(float)))
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(pcd.points).astype(np.float32)


def estimate_normals(points: np.ndarray, knn_normals: int) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(float)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn_normals))
    pcd.normalize_normals()
    return np.asarray(pcd.normals).astype(np.float32)


def compute_features(points: np.ndarray, normals: np.ndarray) -> dict:
    z_min = points[:, 2].min()
    height = (points[:, 2] - z_min).astype(np.float32)

    g = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    cos_theta = np.abs(normals @ g).astype(np.float32)

    return {
        "points": points.astype(np.float32),
        "normals": normals.astype(np.float32),
        "height": height,
        "cos_theta": cos_theta,
        "z_min": np.array([z_min], dtype=np.float32),
    }


def save_cache(cache_path: str, features: dict):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez_compressed(cache_path, **features)


def prepare_scene(mesh_path: str,
                  cache_path: str,
                  sample_points: int = 200000,
                  voxel_size: float = 0.03,
                  knn_normals: int = 30):

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh não encontrada: {mesh_path}")

    print(f"[INFO] Processando mesh (trimesh-first): {mesh_path}")

    points = sample_points_trimesh(mesh_path, sample_points)
    points = voxel_downsample(points, voxel_size)

    if len(points) < 5000:
        raise ValueError(f"Poucos pontos após downsample ({len(points)}). Ajuste voxel_size ou verifique mesh.")

    normals = estimate_normals(points, knn_normals)
    feats = compute_features(points, normals)

    save_cache(cache_path, feats)
    print(f"[INFO] Cache salvo em: {cache_path} | n_points={len(points)}")
