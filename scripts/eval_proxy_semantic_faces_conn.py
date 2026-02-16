# scripts/eval_proxy_semantic_faces_conn.py
#
# Avalia:
#   (1) FULL raw
#   (2) FULL + walk por conectividade (maior cluster em XY)
#
# Execução:
#   python -m scripts.eval_proxy_semantic_faces_conn
#
# Saída:
#   results/proxy_semantic_metrics_conn.csv
#
# Observação:
# - Aqui só melhoramos WALK (conectividade).
# - SUPPORT fica igual ao raw (próximo passo é suporte por cluster).

import os
import json

import numpy as np
import pandas as pd
import trimesh
import open3d as o3d
from sklearn.cluster import DBSCAN

from src.infer import classify


DATA_ROOT = "data/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_POINTS = 250_000

# Parâmetros do seu classificador full
TAU_THETA = 0.90
TAU_WALK_MAX = 0.10
TAU_SUPPORT_MIN = 0.35
TAU_SUPPORT_MAX = 0.90

# DBSCAN em XY (para conectividade do chão)
DBSCAN_EPS = 0.12        # ajuste típico: 0.08–0.20
DBSCAN_MIN_SAMPLES = 80  # ajuste típico: 30–150

# Proxy semântico
WALK_CLASSES = {"floor"}
SUPPORT_CLASSES = {"table", "countertop", "nightstand", "desk"}
OBSTRUCTION_CLASSES = {
    "wall", "door", "window", "blinds",
    "major-appliance", "base-cabinet", "wall-cabinet",
    "shower-stall", "ceiling", "stair",
}
NON_INFORMATIVE = {"other-leaf", "undefined", "", None}


def load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    tm = trimesh.load(mesh_path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) == 0:
            raise ValueError(f"Scene sem geometria: {mesh_path}")
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    return tm


def list_scenes(data_root: str):
    scenes = []
    for d in sorted(os.listdir(data_root)):
        scene_dir = os.path.join(data_root, d)
        if not os.path.isdir(scene_dir):
            continue
        if os.path.exists(os.path.join(scene_dir, "mesh.ply")) and \
           os.path.exists(os.path.join(scene_dir, "semantic.json")) and \
           os.path.exists(os.path.join(scene_dir, "semantic.bin")):
            scenes.append(d)
    return scenes


def load_segmentation_graph(semantic_json_path: str):
    j = json.load(open(semantic_json_path, "r"))
    seg = j.get("segmentation")
    if not isinstance(seg, list):
        raise ValueError(f"'segmentation' não é lista em {semantic_json_path}")

    id_to_class = {}
    id_to_parent = {}

    for item in seg:
        if not isinstance(item, dict):
            continue
        sid = int(item.get("id", 0))
        cls = str(item.get("class", "")).lower().strip()

        parent = item.get("parent", "")
        parent = str(parent).strip()
        pid = int(parent) if parent.isdigit() else None

        id_to_class[sid] = cls
        id_to_parent[sid] = pid

    return id_to_class, id_to_parent


def resolve_class(seg_id: int, id_to_class: dict, id_to_parent: dict, max_hops: int = 50) -> str:
    cur = int(seg_id)
    for _ in range(max_hops):
        cls = id_to_class.get(cur, "")
        if cls not in NON_INFORMATIVE:
            return cls
        cur = id_to_parent.get(cur, None)
        if cur is None:
            break
    return ""


def gravity_vector_from_semantic(semantic_json_path: str) -> np.ndarray:
    j = json.load(open(semantic_json_path, "r"))
    gd = j.get("gravityDirection", None)
    if not isinstance(gd, dict):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    g = np.array([gd.get("x", 0.0), gd.get("y", 0.0), gd.get("z", 1.0)], dtype=np.float32)
    g = g / (np.linalg.norm(g) + 1e-9)
    return g.astype(np.float32)


def class_to_proxy_label(cls: str) -> int:
    cls = str(cls).lower().strip()
    if cls in WALK_CLASSES:
        return 1
    if cls in SUPPORT_CLASSES:
        return 2
    if cls in OBSTRUCTION_CLASSES:
        return 3
    return 0


def estimate_normals(points: np.ndarray, knn: int = 30) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(float)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.normalize_normals()
    return np.asarray(pcd.normals).astype(np.float32)


def compute_height_cos(points: np.ndarray, normals: np.ndarray, g: np.ndarray):
    zmin = points[:, 2].min()
    height = (points[:, 2] - zmin).astype(np.float32)
    cos_theta = np.abs(normals @ g).astype(np.float32)
    return height, cos_theta


def metrics_binary(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & (~gt_mask))
    fn = np.sum((~pred_mask) & gt_mask)

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return float(prec), float(rec), float(f1), float(iou)


def load_seg_ids_per_face(semantic_bin_path: str) -> np.ndarray:
    # 2 x uint16 por primitive/face; coluna 1 é o segment id (no seu caso)
    raw16 = np.fromfile(semantic_bin_path, dtype=np.uint16)
    if raw16.size % 2 != 0:
        raise ValueError(f"semantic.bin com tamanho ímpar (uint16): {raw16.size}")
    pairs = raw16.reshape(-1, 2)
    return pairs[:, 1].astype(np.int32)


def walk_by_connectivity_xy(points: np.ndarray, walk_candidate_mask: np.ndarray) -> np.ndarray:
    """
    Recebe pontos e máscara de candidatos a walk.
    Faz DBSCAN em XY e retorna uma máscara final de walk (maior cluster).
    """
    idx = np.where(walk_candidate_mask)[0]
    if idx.size == 0:
        return np.zeros(len(points), dtype=bool)

    xy = points[idx, :2]  # só XY
    # DBSCAN
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES).fit_predict(xy)

    # remove ruído
    valid = labels >= 0
    if not np.any(valid):
        return np.zeros(len(points), dtype=bool)

    # maior cluster
    lab = labels[valid]
    ids = idx[valid]
    unique, counts = np.unique(lab, return_counts=True)
    best = unique[np.argmax(counts)]

    walk_final = np.zeros(len(points), dtype=bool)
    walk_final[ids[lab == best]] = True
    return walk_final


def evaluate_variant(pred: np.ndarray, gt: np.ndarray):
    valid = gt != 0
    if int(valid.sum()) == 0:
        return {
            "n_eval_points": 0,
            "walk_prec": np.nan, "walk_rec": np.nan, "walk_f1": np.nan, "walk_iou": np.nan,
            "support_prec": np.nan, "support_rec": np.nan, "support_f1": np.nan, "support_iou": np.nan,
            "obstruction_prec": np.nan, "obstruction_rec": np.nan, "obstruction_f1": np.nan, "obstruction_iou": np.nan,
            "gt_walk_frac": np.nan, "gt_support_frac": np.nan, "gt_obstruction_frac": np.nan,
        }

    pv = pred[valid]
    gv = gt[valid]

    out = {"n_eval_points": int(valid.sum())}

    for lab, name in [(1, "walk"), (2, "support"), (3, "obstruction")]:
        p = (pv == lab)
        g = (gv == lab)
        prec, rec, f1, iou = metrics_binary(p, g)
        out[f"{name}_prec"] = prec
        out[f"{name}_rec"] = rec
        out[f"{name}_f1"] = f1
        out[f"{name}_iou"] = iou

    out["gt_walk_frac"] = float(np.mean(gv == 1))
    out["gt_support_frac"] = float(np.mean(gv == 2))
    out["gt_obstruction_frac"] = float(np.mean(gv == 3))

    return out


def main():
    scenes = list_scenes(DATA_ROOT)
    if not scenes:
        raise RuntimeError(f"Nenhuma cena encontrada em {DATA_ROOT}")

    rows = []

    for scene in scenes:
        mesh_path = os.path.join(DATA_ROOT, scene, "mesh.ply")
        sem_json = os.path.join(DATA_ROOT, scene, "semantic.json")
        sem_bin = os.path.join(DATA_ROOT, scene, "semantic.bin")

        print(f"Cena: {scene}")

        tm = load_trimesh(mesh_path)
        n_faces = len(tm.faces)

        seg_ids_all = load_seg_ids_per_face(sem_bin)
        n_lab = int(seg_ids_all.size)
        n_use = min(n_faces, n_lab)

        if n_faces != n_lab:
            print(f"[WARN] {scene}: n_faces={n_faces} != n_labels={n_lab}. Usando n_use={n_use} (min).")

        id_to_class, id_to_parent = load_segmentation_graph(sem_json)
        gvec = gravity_vector_from_semantic(sem_json)

        # amostra pontos e face_idx
        points, face_idx = trimesh.sample.sample_surface(tm, N_POINTS)
        points = points.astype(np.float32)
        face_idx = face_idx.astype(np.int64)

        # restringe aos faces com label
        mask = face_idx < n_use
        points = points[mask]
        face_idx = face_idx[mask]

        seg_ids = seg_ids_all[face_idx]

        # GT proxy
        gt = np.zeros(len(seg_ids), dtype=np.uint8)
        for i, sid in enumerate(seg_ids):
            cls = resolve_class(int(sid), id_to_class, id_to_parent)
            gt[i] = class_to_proxy_label(cls)

        # features geométricas
        normals = estimate_normals(points, knn=30)
        height, cos_theta = compute_height_cos(points, normals, gvec)

        # --- variante 1: raw ---
        pred_raw = classify(
            height, cos_theta,
            tau_theta=TAU_THETA,
            tau_walk_max=TAU_WALK_MAX,
            tau_support_min=TAU_SUPPORT_MIN,
            tau_support_max=TAU_SUPPORT_MAX
        )

        # --- variante 2: conectividade no walk (XY) ---
        walk_candidate = (height <= TAU_WALK_MAX) & (cos_theta >= TAU_THETA)
        walk_final = walk_by_connectivity_xy(points, walk_candidate)

        pred_conn = pred_raw.copy()
        # zera walk antigo e aplica walk_final
        pred_conn[pred_conn == 1] = 0
        pred_conn[walk_final] = 1

        # avalia
        out_raw = evaluate_variant(pred_raw, gt)
        out_conn = evaluate_variant(pred_conn, gt)

        row = {"scene": scene}

        for k, v in out_raw.items():
            row[f"raw_{k}"] = v
        for k, v in out_conn.items():
            row[f"conn_{k}"] = v

        # diagnóstico: fração de candidatos a walk e fração final
        row["walk_candidate_frac"] = float(np.mean(walk_candidate))
        row["walk_final_frac"] = float(np.mean(walk_final))

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "proxy_semantic_metrics_conn.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[OK] Salvo: {out_csv}")

    # médias globais (ignorando NaN)
    num = df.drop(columns=["scene"]).mean(numeric_only=True)
    print("\nMédias globais (raw):")
    print(num[[c for c in num.index if c.startswith("raw_")]])

    print("\nMédias globais (conn):")
    print(num[[c for c in num.index if c.startswith("conn_")]])


if __name__ == "__main__":
    main()
