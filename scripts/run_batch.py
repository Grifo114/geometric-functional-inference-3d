import os
import time
import numpy as np
import pandas as pd

from src.prepare_scene import prepare_scene
from src.infer import classify


DATA_ROOT = "data/replica"
CACHE_ROOT = "cache/replica"
RESULTS_DIR = "results"

os.makedirs(CACHE_ROOT, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def process_scene(scene_name):
    mesh_path = os.path.join(DATA_ROOT, scene_name, "mesh.ply")
    cache_path = os.path.join(CACHE_ROOT, f"{scene_name}.npz")

    start_time = time.time()

    # Gera cache se não existir
    if not os.path.exists(cache_path):
        prepare_scene(
            mesh_path=mesh_path,
            cache_path=cache_path,
            sample_points=200000,
            voxel_size=0.03,
            knn_normals=30
        )

    data = np.load(cache_path)

    height = data["height"]
    cos_theta = data["cos_theta"]

    # PROTEÇÃO: descarta caches suspeitos
    if len(height) < 5000:
        raise ValueError(f"Cache suspeito: poucos pontos ({len(height)})")

    labels = classify(height, cos_theta)

    total_points = len(labels)

    pct_walk = np.sum(labels == 1) / total_points
    pct_support = np.sum(labels == 2) / total_points
    pct_obstruction = np.sum(labels == 3) / total_points

    elapsed = time.time() - start_time

    return {
        "scene": scene_name,
        "n_points": total_points,
        "pct_walk": pct_walk,
        "pct_support": pct_support,
        "pct_obstruction": pct_obstruction,
        "time_sec": elapsed,
        "status": "ok"
    }


def main():
    scenes = sorted(os.listdir(DATA_ROOT))

    results = []

    for scene in scenes:
        mesh_path = os.path.join(DATA_ROOT, scene, "mesh.ply")

        if not os.path.isdir(os.path.join(DATA_ROOT, scene)):
            continue

        if not os.path.exists(mesh_path):
            print(f"[SKIP] {scene} sem mesh.ply")
            continue

        print(f"Processando: {scene}")

        try:
            stats = process_scene(scene)
        except Exception as e:
            print(f"[ERRO] {scene}: {e}")
            stats = {
                "scene": scene,
                "status": f"error",
                "error_msg": str(e)
            }

        results.append(stats)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(RESULTS_DIR, "batch_results.csv"), index=False)

    print("\nResumo:")
    print(df.describe(include="all"))


if __name__ == "__main__":
    main()
