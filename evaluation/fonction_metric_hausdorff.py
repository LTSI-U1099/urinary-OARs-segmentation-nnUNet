import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion

def _get_points(mask: np.ndarray, use_surface: bool) -> np.ndarray:
#    """
#    Renvoie les coordonnées (z,y,x) des points d'un masque binaire.
#    - use_surface=True : points de surface via érosion binaire
#    - use_surface=False : tous les voxels non-nuls
#    """
    if mask is None or mask.size == 0 or mask.sum() == 0:
        return np.empty((0, 3), dtype=int)
    if not use_surface:
        return np.argwhere(mask)
    eroded = binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)
    surf = mask & (~eroded)
    return np.argwhere(surf)


def _nn_dists(A: np.ndarray, B: np.ndarray) -> np.ndarray:
#    """
#    Distances des plus proches voisins de A vers B (euclidiennes).
#    Gère proprement les cas vides.
#    """
    if A.shape[0] == 0 and B.shape[0] == 0:
        return np.array([0.0])
    if A.shape[0] == 0 or B.shape[0] == 0:
        return np.array([np.inf])
    treeB = cKDTree(B)
    dists, _ = treeB.query(A, k=1, workers=-1)
    return dists


def hd_percentile_binary(mask_pred: np.ndarray,
                         mask_gt: np.ndarray,
                         spacing_xyz=None,           # (sx, sy, sz) en mm. None => voxels.
                         percentile: float = 95.0,   # 90, 95, 99, ...
                         use_surface: bool = True,
                         max_points: int | None = 2000,
                         seed: int = 42,
                         empty_policy: str = "inf") -> float:
#    """
#    HD^(percentile) symétrique entre 2 masques binaires (z,y,x).
#    - Si spacing_xyz est fourni (SimpleITK: (sx,sy,sz)), distances en mm (anisotropie).
#      Sinon, distances en voxels.
#    - empty_policy: 'inf' (par défaut), 'zero' ou 'nan' quand un seul des deux masques est vide.
#    """
    rng = np.random.default_rng(seed)

    A = _get_points(mask_pred.astype(bool), use_surface=use_surface)
    B = _get_points(mask_gt.astype(bool),   use_surface=use_surface)

    # Traitement des cas vides avant échantillonnage (plus clair)
    if A.shape[0] == 0 and B.shape[0] == 0:
        return 0.0
    if A.shape[0] == 0 or B.shape[0] == 0:
        return {"inf": float("inf"), "zero": 0.0, "nan": float("nan")}[empty_policy]

    # Sous-échantillonnage optionnel (comme ta version)
    if max_points is not None and A.shape[0] > max_points:
        A = A[rng.choice(A.shape[0], max_points, replace=False)]
    if max_points is not None and B.shape[0] > max_points:
        B = B[rng.choice(B.shape[0], max_points, replace=False)]

    # Passage aux mm si spacing fourni (attention à l’ordre)
    if spacing_xyz is not None:
        sx, sy, sz = spacing_xyz            # SimpleITK -> (x, y, z)
        scale_zyx = np.array([sz, sy, sx])  # tableau -> (z, y, x)
        A = A.astype(np.float64) * scale_zyx
        B = B.astype(np.float64) * scale_zyx

    # Distances dirigées et percentile par direction
    dA = _nn_dists(A, B)
    dB = _nn_dists(B, A)
    pA = np.percentile(dA, percentile)
    pB = np.percentile(dB, percentile)
    return float(max(pA, pB))  # symétrique


def hd_percentile_per_label(seg_pred: np.ndarray,
                            seg_gt: np.ndarray,
                            max_label: int,
                            spacing_xyz=None,          # None => voxels; (sx,sy,sz) => mm
                            percentile: float = 95.0,
                            use_surface: bool = True,
                            max_points: int | None = 2000,
                            seed: int = 42,
                            empty_policy: str = "inf") -> np.ndarray:
#    """
#    Renvoie un array de taille max_label :
#    distances[k-1] = HD^(percentile) pour le label k (fond 0 exclu).
#    Règles de fusion conservées : 6 ? {6,10} et 7 ? {7,10}.
#    """
    dists = np.zeros(max_label-1, dtype=float)

    for lab in range(1, max_label):
        if lab == 6:
            pred_mask = (seg_pred == 6) | (seg_pred == 10)
            gt_mask   = (seg_gt   == 6) | (seg_gt   == 10)
        elif lab == 7:
            pred_mask = (seg_pred == 7) | (seg_pred == 10)
            gt_mask   = (seg_gt   == 7) | (seg_gt   == 10)
        else:
            pred_mask = (seg_pred == lab)
            gt_mask   = (seg_gt   == lab)

        dists[lab - 1] = hd_percentile_binary(
            pred_mask, gt_mask,
            spacing_xyz=spacing_xyz,
            percentile=percentile,
            use_surface=use_surface,
            max_points=max_points,
            seed=seed,
            empty_policy=empty_policy
        )
    return dists