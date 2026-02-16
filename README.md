# Geometric Functional Inference in 3D Scenes

This repository contains the experimental framework developed for the Master's qualification.

## Conceptual Positioning

This project does not attempt to recover semantic object categories
(e.g., "floor", "table", "chair") from geometry.

Instead, it investigates whether explicit 3D geometry is sufficient
to infer structural-functional properties of space, such as:

- Horizontally low surfaces compatible with locomotion
- Horizontally elevated surfaces structurally compatible with support
- Vertical or volumetric regions acting as obstruction

The goal is to experimentally characterize the limits of geometry-only
functional inference.

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

## Summary of Preliminary Results

Experiments on 18 Replica scenes show:

- Obstruction inference is structurally consistent
- Walkable-surface inference captures horizontal-low regions,
  but does not directly align with semantic floor labels
- Support inference remains structurally ambiguous

These findings support the hypothesis that geometry provides a
basal structural-functional layer, but is insufficient for
context-dependent affordances.

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

