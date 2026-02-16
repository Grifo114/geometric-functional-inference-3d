import os, json
import numpy as np
import pandas as pd
import trimesh
from sklearn.neighbors import KDTree

from src.infer import classify

DATA_ROOT = "data/replica"
CACHE_ROOT = "cache/replica"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Ajuste essas listas quando você confirmar os nomes exatos das classes no JSON
WALK_CLASSES = {"floor", "carpet", "rug"}
SUPPORT_CLASSES = {"table", "desk", "counter", "shelf", "bench"}
OBSTRUCTION_CLASSES = {"wall", "door", "window", "column", "pillar"}

def load_trimesh(mesh_path: str):
    tm = trimesh.load(mesh_path, force="mesh")
    if isinstance(tm, trimesh.Scene):
        tm = trimesh.util.concatenate(tuple(tm.geometry.values()))
    return tm

def parse_semantic_mapping(semantic_json_path: str):
    j = json.load(open(semantic_json_path, "r"))
    # tente achar uma lista de objetos/instâncias
    candidates = None
    for key in ["objects", "instances"]:
        if key in j and isinstance(j[key], list):
            candidates = j[key]
            break
    if candidates is None:
        raise ValueError(f"Não achei 'objects' nem 'instances' em {semantic_json_path}. Keys={list(j.keys())}")

    # mapa: instance_id -> class_name (string)
    id_to_class = {}
    for obj in candidates:
        # tenta campos comuns
        iid = obj.get("id", obj.get("instanceId", obj.get("instance_id")))
        # nome/categoria
        cname = obj.get("class", obj.get("category", obj.get("name", "")))
        if iid is None:
            continue
        if isinstance(cname, dict):
            # às vezes vem { "name": "floor" }
            cname = cname.get("name", "")
        cname = str(cname).lower().strip()
        id_to_class[int(iid)] = cname

    return id_to_class

def load_vertex_labels_sparse(bin_path: str, n_verts: int) -> np.ndarray:
    raw = np.fromfile(bin_path, dtype=np.int32)
    if raw.size % 2 != 0:
        raise ValueError(f"semantic.bin tem tamanho ímpar ({raw.size}).")

    pairs = raw.reshape(-1, 2)
    vid = pairs[:, 0]
    lab = pairs[:, 1]

    if vid.min() < 0 or vid.max() >= n_verts:
        raise ValueError("coluna 0 não parece índice de vértice (fora do range).")

    vlab = np.zeros(n_verts, dtype=np.int32)
    vlab[vid] = lab
    return vlab


def class_to_proxy_label(class_name: str) -> int:
    # 0=other, 1=walk, 2=support, 3=obstruction
    if class_name in WALK_CLASSES:
        return 1
    if class_name in SUPPORT_CLASSES:
        return 2
    if class_name in OBSTRUCTION_CLASSES:
        return 3
    return 0

def metrics_binary(pred_mask, gt_mask):
    tp = np.sum(pred_mask & gt_mask)
    fp = np.sum(pred_mask & (~gt_mask))
    fn = np.sum((~pred_mask) & gt_mask)

    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    iou  = tp / (tp + fp + fn + 1e-9)
    return prec, rec, f1, iou

def main():
    rows = []
    for fn in sorted(os.listdir(CACHE_ROOT)):
        if not fn.endswith(".npz"):
            continue
        scene = fn.replace(".npz","")
        cache_path = os.path.join(CACHE_ROOT, fn)

        mesh_path = os.path.join(DATA_ROOT, scene, "mesh.ply")
        sem_json  = os.path.join(DATA_ROOT, scene, "semantic.json")
        sem_bin   = os.path.join(DATA_ROOT, scene, "semantic.bin")

        if not (os.path.exists(mesh_path) and os.path.exists(sem_json) and os.path.exists(sem_bin)):
            print(f"[SKIP] {scene} faltando mesh/semantic")
            continue

        # carrega cache (pontos amostrados)
        data = np.load(cache_path)
        points = data["points"]
        height = data["height"]
        cos_theta = data["cos_theta"]

        # predição geométrica (full)
        pred = classify(height, cos_theta, tau_theta=0.90, tau_walk_max=0.10, tau_support_min=0.35, tau_support_max=0.90)

        # carrega mesh + labels por vértice
        tm = load_trimesh(mesh_path)
        verts = tm.vertices.astype(np.float32)
        vertex_labels = load_vertex_labels_sparse(sem_bin, len(verts))
        id_to_class = parse_semantic_mapping(sem_json)

        # mapeia instância -> proxy label
        v_proxy = np.zeros(len(verts), dtype=np.uint8)
        for i, iid in enumerate(inst_ids):
            cname = id_to_class.get(int(iid), "")
            v_proxy[i] = class_to_proxy_label(cname)

        # atribui proxy aos pontos do cache via nearest vertex
        tree = KDTree(verts)
        nn = tree.query(points, k=1, return_distance=False).reshape(-1)
        gt = v_proxy[nn]  # 0/1/2/3

        # calcula métricas por classe funcional (um-vs-rest)
        out = {"scene": scene}
        for label, name in [(1,"walk"), (2,"support"), (3,"obstruction")]:
            pred_m = (pred == label)
            gt_m   = (gt == label)
            prec, rec, f1, iou = metrics_binary(pred_m, gt_m)
            out[f"{name}_prec"] = prec
            out[f"{name}_rec"]  = rec
            out[f"{name}_f1"]   = f1
            out[f"{name}_iou"]  = iou

        rows.append(out)
        print(f"[OK] {scene}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "proxy_semantic_metrics.csv"), index=False)

    print("\nResumo (média):")
    print(df.drop(columns=["scene"]).mean(numeric_only=True))

if __name__ == "__main__":
    main()
