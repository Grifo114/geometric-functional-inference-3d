# Geometric Functional Inference in 3D Scenes

This repository contains the experimental framework developed for the Master's qualification.

## Research Goal

Investigate the extent to which explicit 3D geometry alone is sufficient to infer structural-functional properties of indoor environments.

## Key Idea

We distinguish between:
- Structural geometric properties
- Semantic object categories

This work evaluates what can be inferred from geometry alone.

## Experiments

1. Batch inference on Replica scenes
2. Parameter sweep
3. Ablation studies
4. Proxy semantic evaluation

## Dataset

Experiments use the Replica dataset.
Dataset is not included due to size.

## Reproducibility

Python 3.11

Install dependencies:
pip install -r requirements.txt

Run batch:
python -m scripts.run_batch

Run evaluation:
python -m scripts.eval_proxy_semantic_faces_conn

