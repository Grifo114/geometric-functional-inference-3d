import numpy as np


def classify(height,
             cos_theta,
             tau_theta=0.92,
             tau_walk_max=0.10,
             tau_support_min=0.35,
             tau_support_max=0.90):
    """
    Classificação funcional baseada apenas em geometria.

    Labels:
      0 = desconhecido
      1 = walk (caminhável)
      2 = support (suporte)
      3 = obstruction (obstrução)
    """

    labels = np.zeros_like(height, dtype=np.uint8)

    walk = (height <= tau_walk_max) & (cos_theta >= tau_theta)

    support = (
        (height >= tau_support_min) &
        (height <= tau_support_max) &
        (cos_theta >= tau_theta)
    )

    obstruction = (cos_theta < tau_theta) | (height > tau_support_max)

    labels[walk] = 1
    labels[support] = 2
    labels[obstruction] = 3

    return labels
