# scripts/eval_proxy_semantic_faces_plane.py
#
# Avalia o classificador geométrico contra proxy semântico,
# mas usando height baseado no plano do piso (RANSAC) em vez de z-zmin.
#
# Execução:
#   python -m scripts.eval_proxy_semantic_faces_plane
#
# Saída:
#   results/proxy_semantic_metrics_plane.csv

import os
import json

import numpy as np
import pandas as pd
import trimesh
import open3d as o3d

from src.infer import classify


DATA_ROOT = "data/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_POINTS = 250_000

TAU_THETA = 0.80
TAU_WALK_MAX = 0.25
TAU_SUPPORT_MIN = 0.35
TAU_SUPPORT_MAX = 0.90

# Proxy semântico
WALK_CLASSES = {"floor"}
SUPPORT_CLASSES = {"table", "countertop", "nightstand", "desk"}
OBSTRUCTION_CLASSES = {
    "wall", "door", "window", "blinds",
    "major-appliance", "base-cabinet", "wall-cabinet",
    "shower-stall", "ceiling", "stair",
}
NON_INFORMATIVE = {"other-leaf", "undefined", "", None}

# RANSAC plano do piso
PLANE_DIST_THRESH = 0.02  # 2cm
PLANE_RANSAC_N = 3
PLANE_ITERS = 1500


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


def class_to_proxy_label(cls: str) -> int:
    cls = str(cls).lower().strip()
    if cls in WALK_CLASSES:
        return 1
    if cls in SUPPORT_CLASSES:
        return 2
    if cls in OBSTRUCTION_CLASSES:
        return 3
    return 0


def gravity_vector_from_semantic(semantic_json_path: str) -> np.ndarray:
    j = json.load(open(semantic_json_path, "r"))
    gd = j.get("gravityDirection", None)
    if not isinstance(gd, dict):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    g = np.array([gd.get("x", 0.0), gd.get("y", 0.0), gd.get("z", 1.0)], dtype=np.float32)
    g = g / (np.linalg.norm(g) + 1e-9)
    return g.astype(np.float32)


def load_seg_ids_per_face(semantic_bin_path: str) -> np.ndarray:
    raw16 = np.fromfile(semantic_bin_path, dtype=np.uint16)
    if raw16.size % 2 != 0:
        raise ValueError(f"semantic.bin com tamanho ímpar (uint16): {raw16.size}")
    pairs = raw16.reshape(-1, 2)
    return pairs[:, 1].astype(np.int32)  # coluna correta


def estimate_normals(points: np.ndarray, knn: int = 30) -> np.ndarray:
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points.astype(float)))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.normalize_normals()
    return np.asarray(pcd.normals).astype(np.float32)


def compute_cos_theta(normals: np.ndarray, g: np.ndarray) -> np.ndarray:
    return np.abs(normals @ g).astype(np.float32)


def fit_floor_plane(points: np.ndarray, normals: np.ndarray, g: np.ndarray):
    """
    Encontra plano dominante do piso:
    - usa candidatos horizontais (cos>=tau_theta)
    - usa candidatos baixos por quantil de projeção em g (robusto)
    Retorna coeficientes (a,b,c,d) com ||(a,b,c)||=1 e inliers idx.
    """
    cos_theta = compute_cos_theta(normals, g)
    horiz = cos_theta >= TAU_THETA

    # projeção ao longo da gravidade (não é altura absoluta, é só para filtrar baixo)
    proj = points @ g
    q = np.quantile(proj, 0.10)  # pega 10% mais baixo como candidatos
    low = proj <= q

    cand = horiz & low
    idx = np.where(cand)[0]
    if idx.size < 5000:
        # fallback: só horizontais
        idx = np.where(horiz)[0]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points[idx].astype(float)))
    plane, inliers = pcd.segment_plane(
        distance_threshold=PLANE_DIST_THRESH,
        ransac_n=PLANE_RANSAC_N,
        num_iterations=PLANE_ITERS
    )

    # plane = [a,b,c,d]
    a, b, c, d = plane
    n = np.array([a, b, c], dtype=np.float32)
    n = n / (np.linalg.norm(n) + 1e-9)
    d = float(d) / (np.linalg.norm([a, b, c]) + 1e-9)

    # garante normal apontando “para cima” (alinhada com g)
    if float(n @ g) < 0:
        n = -n
        d = -d

    return n, d


def height_from_plane(points: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    # distância assinada ao plano n·x + d = 0
    return (points @ n + d).astype(np.float32)


def metrics_binary(pred_mask: np.ndarray, gt_mask: np.ndarray):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & (~gt_mask))
    fn = np.sum((~pred_mask) & gt_mask)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    iou = tp / (tp + fp + fn + 1e-9)
    return float(prec), float(rec), float(f1), float(iou)


def evaluate(pred: np.ndarray, gt: np.ndarray):
    valid = gt != 0
    if int(valid.sum()) == 0:
        return None

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
        g = gravity_vector_from_semantic(sem_json)

        points, face_idx = trimesh.sample.sample_surface(tm, N_POINTS)
        points = points.astype(np.float32)
        face_idx = face_idx.astype(np.int64)

        mask = face_idx < n_use
        points = points[mask]
        face_idx = face_idx[mask]

        seg_ids = seg_ids_all[face_idx]

        gt = np.zeros(len(seg_ids), dtype=np.uint8)
        for i, sid in enumerate(seg_ids):
            cls = resolve_class(int(sid), id_to_class, id_to_parent)
            gt[i] = class_to_proxy_label(cls)

        # Normais + cos_theta
        normals = estimate_normals(points, knn=30)
        cos_theta = compute_cos_theta(normals, g)

        # height antigo (z-zmin) — para comparar
        zmin = points[:, 2].min()
        height_z = (points[:, 2] - zmin).astype(np.float32)

        pred_z = classify(
            height_z, cos_theta,
            tau_theta=TAU_THETA,
            tau_walk_max=TAU_WALK_MAX,
            tau_support_min=TAU_SUPPORT_MIN,
            tau_support_max=TAU_SUPPORT_MAX
        )

        # height por plano do piso
        n_plane, d_plane = fit_floor_plane(points, normals, g)
        height_p = height_from_plane(points, n_plane, d_plane)

        # altura tem que ser >=0 no chão; clampa negativos pequenos
        height_p = np.maximum(height_p, 0.0)

        pred_p = classify(
            height_p, cos_theta,
            tau_theta=TAU_THETA,
            tau_walk_max=TAU_WALK_MAX,
            tau_support_min=TAU_SUPPORT_MIN,
            tau_support_max=TAU_SUPPORT_MAX
        )

        out_z = evaluate(pred_z, gt)
        out_p = evaluate(pred_p, gt)

        row = {"scene": scene}

        if out_z is None:
            for k in ["n_eval_points","walk_prec","walk_rec","walk_f1","walk_iou",
                      "support_prec","support_rec","support_f1","support_iou",
                      "obstruction_prec","obstruction_rec","obstruction_f1","obstruction_iou",
                      "gt_walk_frac","gt_support_frac","gt_obstruction_frac"]:
                row[f"z_{k}"] = np.nan
                row[f"plane_{k}"] = np.nan
        else:
            for k,v in out_z.items():
                row[f"z_{k}"] = v
            for k,v in out_p.items():
                row[f"plane_{k}"] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "proxy_semantic_metrics_plane.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[OK] Salvo: {out_csv}")

    num = df.drop(columns=["scene"]).mean(numeric_only=True)
    print("\nMédias globais (z-zmin):")
    print(num[[c for c in num.index if c.startswith("z_")]])

    print("\nMédias globais (plano):")
    print(num[[c for c in num.index if c.startswith("plane_")]])


if __name__ == "__main__":
    main()
