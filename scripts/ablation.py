import os
import numpy as np
import pandas as pd

CACHE_ROOT = "cache/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def classify_height_only(height,
                         tau_walk_max=0.10,
                         tau_support_min=0.35,
                         tau_support_max=0.90):

    labels = np.zeros_like(height, dtype=np.uint8)

    walk = (height <= tau_walk_max)
    support = (height >= tau_support_min) & (height <= tau_support_max)
    obstruction = (height > tau_support_max)

    labels[walk] = 1
    labels[support] = 2
    labels[obstruction] = 3

    return labels


def classify_normal_only(cos_theta,
                         tau_theta=0.90):

    labels = np.zeros_like(cos_theta, dtype=np.uint8)

    horizontal = cos_theta >= tau_theta
    vertical = cos_theta < tau_theta

    # tudo horizontal vira walk (simplificação extrema)
    labels[horizontal] = 1
    labels[vertical] = 3

    return labels


def classify_full(height, cos_theta,
                  tau_theta=0.90,
                  tau_walk_max=0.10,
                  tau_support_min=0.35,
                  tau_support_max=0.90):

    labels = np.zeros_like(height, dtype=np.uint8)

    walk = (height <= tau_walk_max) & (cos_theta >= tau_theta)
    support = (height >= tau_support_min) & (height <= tau_support_max) & (cos_theta >= tau_theta)
    obstruction = (cos_theta < tau_theta) | (height > tau_support_max)

    labels[walk] = 1
    labels[support] = 2
    labels[obstruction] = 3

    return labels


def process_scene(scene_name, path):
    data = np.load(path)
    height = data["height"]
    cos_theta = data["cos_theta"]

    results = {}

    for name, fn in {
        "height_only": lambda: classify_height_only(height),
        "normal_only": lambda: classify_normal_only(cos_theta),
        "full": lambda: classify_full(height, cos_theta)
    }.items():

        labels = fn()
        n = len(labels)

        results[f"{name}_walk"] = float(np.sum(labels == 1) / n)
        results[f"{name}_support"] = float(np.sum(labels == 2) / n)
        results[f"{name}_obstruction"] = float(np.sum(labels == 3) / n)

    return results


def main():
    rows = []

    for fn in sorted(os.listdir(CACHE_ROOT)):
        if not fn.endswith(".npz"):
            continue

        scene = fn.replace(".npz", "")
        path = os.path.join(CACHE_ROOT, fn)

        print(f"Processando: {scene}")
        stats = process_scene(scene, path)
        stats["scene"] = scene
        rows.append(stats)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation.csv"), index=False)

    print("\nResumo:")
    print(df.describe())


if __name__ == "__main__":
    main()
