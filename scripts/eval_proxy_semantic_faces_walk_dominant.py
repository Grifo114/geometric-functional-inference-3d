# scripts/eval_proxy_semantic_faces_walk_dominant.py
#
# Variante B: Walk como "maior região horizontal dominante" via grid XY + componente conectada
#
# Execução:
#   python -m scripts.eval_proxy_semantic_faces_walk_dominant
#
# Saída:
#   results/proxy_semantic_metrics_walk_dominant.csv

import os
import json
from collections import deque

import numpy as np
import pandas as pd
import trimesh
import open3d as o3d

from src.infer import classify


DATA_ROOT = "data/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

N_POINTS = 250_000

# Parâmetros do classificador "raw"
TAU_THETA = 0.80
TAU_WALK_MAX = 0.25
TAU_SUPPORT_MIN = 0.35
TAU_SUPPORT_MAX = 0.90

# Walk dominante (global)
WALK_LOW_Q = 0.25        # "faixa inferior" ao longo da gravidade (quantil)
GRID_RES = 0.10          # 10 cm por célula (tente 0.08–0.15)
MIN_CELL_POINTS = 6      # mínimo de pontos na célula para contar como ocupada
CONNECTIVITY = 8         # 4 ou 8

# Proxy semântico
WALK_CLASSES = {"floor"}
SUPPORT_CLASSES = {"table", "countertop", "nightstand", "desk"}
OBSTRUCTION_CLASSES = {
    "wall", "door", "window", "blinds",
    "major-appliance", "base-cabinet", "wall-cabinet",
    "shower-stall", "ceiling", "stair",
}
NON_INFORMATIVE = {"other-leaf", "undefined", "", None}


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


def load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    tm = trimesh.load(mesh_path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        if len(tm.geometry) == 0:
            raise ValueError(f"Scene sem geometria: {mesh_path}")
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    return tm


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
    # 2 x uint16 por face; coluna 1 é o segment_id (confirmado no seu caso)
    raw16 = np.fromfile(semantic_bin_path, dtype=np.uint16)
    if raw16.size % 2 != 0:
        raise ValueError(f"semantic.bin com tamanho ímpar (uint16): {raw16.size}")
    pairs = raw16.reshape(-1, 2)
    return pairs[:, 1].astype(np.int32)


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


def grid_indices_xy(points: np.ndarray, res: float):
    xmin, ymin = points[:, 0].min(), points[:, 1].min()
    ix = np.floor((points[:, 0] - xmin) / res).astype(np.int32)
    iy = np.floor((points[:, 1] - ymin) / res).astype(np.int32)
    return ix, iy, xmin, ymin


def largest_component_mask(occ: np.ndarray, connectivity: int = 8) -> np.ndarray:
    """
    occ: bool grid [H,W]
    retorna mask bool da maior componente conectada
    """
    H, W = occ.shape
    visited = np.zeros_like(occ, dtype=bool)

    if connectivity == 8:
        neigh = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    else:
        neigh = [(-1,0),(1,0),(0,-1),(0,1)]

    best = []
    for r in range(H):
        for c in range(W):
            if not occ[r, c] or visited[r, c]:
                continue
            q = deque()
            q.append((r, c))
            visited[r, c] = True
            comp = [(r, c)]
            while q:
                rr, cc = q.popleft()
                for dr, dc in neigh:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and occ[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            if len(comp) > len(best):
                best = comp

    mask = np.zeros_like(occ, dtype=bool)
    for r, c in best:
        mask[r, c] = True
    return mask


def walk_dominant_global(points: np.ndarray, height: np.ndarray, cos_theta: np.ndarray, g: np.ndarray):
    """
    Retorna máscara booleana de walk usando:
    - horizontais (cos_theta >= TAU_THETA)
    - baixa altitude via quantil de projeção em g (WALK_LOW_Q)
    - maior componente conectada em grid XY
    """
    horiz = cos_theta >= TAU_THETA

    # filtro "baixo" robusto: projeção ao longo da gravidade
    proj = points @ g
    thr = np.quantile(proj, WALK_LOW_Q)
    low = proj <= thr

    cand = horiz & low
    if np.sum(cand) < 2000:
        # fallback: usa o critério antigo de altura (menos robusto)
        cand = (height <= TAU_WALK_MAX) & horiz

    idx = np.where(cand)[0]
    if idx.size == 0:
        return np.zeros(len(points), dtype=bool), cand

    # grid XY só nos candidatos
    ix, iy, _, _ = grid_indices_xy(points[idx], GRID_RES)
    W = int(ix.max()) + 1
    H = int(iy.max()) + 1

    # conta pontos por célula
    counts = np.zeros((H, W), dtype=np.int32)
    for x, y in zip(ix, iy):
        counts[y, x] += 1

    occ = counts >= MIN_CELL_POINTS
    if not np.any(occ):
        return np.zeros(len(points), dtype=bool), cand

    dom_cells = largest_component_mask(occ, connectivity=CONNECTIVITY)

    # marca como walk os pontos candidatos que caem nas células dominantes
    walk = np.zeros(len(points), dtype=bool)
    # para cada candidato, verifica se célula (iy,ix) está no componente dominante
    good = dom_cells[iy, ix]
    walk[idx[good]] = True

    return walk, cand


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

        #normals = estimate_normals(points, knn=30)
        face_normals = tm.face_normals.astype(np.float32)
        normals = face_normals[face_idx]
        height, cos_theta = compute_height_cos(points, normals, g)

        # RAW
        pred_raw = classify(
            height, cos_theta,
            tau_theta=TAU_THETA,
            tau_walk_max=TAU_WALK_MAX,
            tau_support_min=TAU_SUPPORT_MIN,
            tau_support_max=TAU_SUPPORT_MAX
        )

        # Walk dominante global
        walk_dom, walk_cand = walk_dominant_global(points, height, cos_theta, g)

        pred_dom = pred_raw.copy()
        pred_dom[pred_dom == 1] = 0
        pred_dom[walk_dom] = 1

        out_raw = evaluate(pred_raw, gt)
        out_dom = evaluate(pred_dom, gt)

        row = {"scene": scene}
        if out_raw is None:
            rows.append(row)
            continue

        for k, v in out_raw.items():
            row[f"raw_{k}"] = v
        for k, v in out_dom.items():
            row[f"dom_{k}"] = v

        row["walk_cand_frac"] = float(np.mean(walk_cand))
        row["walk_dom_frac"] = float(np.mean(walk_dom))
        rows.append(row)

    df = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "proxy_semantic_metrics_walk_dominant.csv")
    df.to_csv(out_csv, index=False)

    print(f"\n[OK] Salvo: {out_csv}")

    num = df.drop(columns=["scene"]).mean(numeric_only=True)

    print("\nMédias globais (RAW):")
    print(num[[c for c in num.index if c.startswith("raw_")]])

    print("\nMédias globais (WALK_DOMINANT):")
    print(num[[c for c in num.index if c.startswith("dom_")]])

    print("\nDiagnóstico (frações):")
    diag_cols = [c for c in num.index if "walk_" in c]
    for c in diag_cols:
        print(c, "=", float(num[c]))

# depois de calcular walk_mask e gt (proxy label)
    walk_idx = np.where(pred_raw == 1)[0]
    print("Fraction of predicted walk that is GT floor:",
        np.mean(gt[walk_idx] == 1))


if __name__ == "__main__":
    main()
