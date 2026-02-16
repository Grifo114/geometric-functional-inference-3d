import os
import itertools
import numpy as np
import pandas as pd

from src.infer import classify

CACHE_ROOT = "cache/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

TAU_THETA_LIST = [0.85, 0.88, 0.90, 0.92, 0.94, 0.96]
TAU_WALK_LIST = [0.05, 0.10, 0.15, 0.20]

# Mantém suporte fixo por enquanto
TAU_SUP_MIN = 0.35
TAU_SUP_MAX = 0.90


def iter_scenes():
    for fn in sorted(os.listdir(CACHE_ROOT)):
        if fn.endswith(".npz"):
            yield fn.replace(".npz", ""), os.path.join(CACHE_ROOT, fn)


def main():
    rows = []
    for scene, path in iter_scenes():
        data = np.load(path)
        height = data["height"]
        cos_theta = data["cos_theta"]

        for tau_theta, tau_walk in itertools.product(TAU_THETA_LIST, TAU_WALK_LIST):
            labels = classify(
                height, cos_theta,
                tau_theta=tau_theta,
                tau_walk_max=tau_walk,
                tau_support_min=TAU_SUP_MIN,
                tau_support_max=TAU_SUP_MAX
            )

            n = len(labels)
            rows.append({
                "scene": scene,
                "tau_theta": tau_theta,
                "tau_walk": tau_walk,
                "pct_walk": float(np.sum(labels == 1) / n),
                "pct_support": float(np.sum(labels == 2) / n),
                "pct_obstruction": float(np.sum(labels == 3) / n),
            })

        df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "sweep_params.csv"), index=False)

    # Agregado (apenas colunas numéricas)
    numeric_cols = ["pct_walk", "pct_support", "pct_obstruction"]

    agg = (
        df.groupby(["tau_theta", "tau_walk"])[numeric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten das colunas MultiIndex
    agg.columns = [
        f"{a}_{b}" if b else a
        for a, b in agg.columns.to_flat_index()
    ]

    agg.to_csv(os.path.join(RESULTS_DIR, "sweep_params_summary.csv"), index=False)

    print("[OK] results/sweep_params.csv")
    print("[OK] results/sweep_params_summary.csv")



if __name__ == "__main__":
    main()
