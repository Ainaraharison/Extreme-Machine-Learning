from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from utils import _grassmann_distance, _prng_vector

logger = logging.getLogger(__name__)


def _orth(M: np.ndarray) -> np.ndarray:
    """Base orthonormale des lignes de M via QR. Retourne (rank, d)."""
    Q, _ = np.linalg.qr(M.astype(np.float64).T)
    return Q[:, :M.shape[0]].T


def _grassmann_distance_fast(Ao: np.ndarray, Q_cur: np.ndarray,
                              rv: np.ndarray, s: int) -> float:
    """Distance de Grassmann optimisée pour la boucle de sélection LFSR.

    Ao    : base orthonormale de B_T_full, pré-calculée une fois par couche
    Q_cur : base orthonormale de BNRC_current, pré-calculée une fois par vecteur
    rv    : vecteur candidat LFSR (non normalisé)
    s     : dimension cible (nb de vecteurs dans B_T_full)

    Évite de refaire QR(B_T_full) et QR(BNRC_current) à chaque candidat.
    Seule la projection de rv + SVD(s × cur_size+1) sont recalculés.
    """
    rv_d = rv.astype(np.float64)
    # Projeter rv hors du sous-espace courant
    rv_perp = rv_d - Q_cur.T @ (Q_cur @ rv_d)
    norm = np.linalg.norm(rv_perp)
    if norm < 1e-10:
        # rv quasi-colinéaire à la base courante → pénalité maximale
        n_missing = s - (Q_cur.shape[0] + 1)
        return float(np.sqrt((n_missing + 1) * (np.pi / 2) ** 2))

    Bo = np.vstack([Q_cur, (rv_perp / norm)[np.newaxis, :]])  # (cur_size+1, d)

    sigma = np.linalg.svd(Ao @ Bo.T, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    angles = np.arccos(sigma)

    n_missing = abs(s - Bo.shape[0])
    return float(np.sqrt(np.sum(angles ** 2) + n_missing * (np.pi / 2) ** 2))


def _build_bnrc(analysis: dict, randomization_ratio: float,
                K: int, seed: int, name: str) -> dict:
    """Construit la base hybride BNRC pour une couche (étape 3 NRC).

    Scinde la base PCA B_T (s vecteurs) selon le ratio p :
        n_keep = max(1, round(s × (1 − p)))  vecteurs PCA conservés
        n_rand = s − n_keep                   vecteurs remplacés par LFSR
    """
    np.random.seed(seed)

    B_T_full = analysis["B_T"]
    s, d     = B_T_full.shape

    n_keep = max(1, int(round(s * (1 - randomization_ratio))))
    n_rand = s - n_keep

    B_T_kept   = B_T_full[:n_keep]
    B_T_target = B_T_full[n_keep:]

    BNRC_current   = B_T_kept.copy()
    random_vectors = []
    selected_seeds = []
    grassmann_dists = []

    logger.info("[%s] s=%d | n_keep=%d (Vs) | n_rand=%d (Vr) | K=%d",
                name, s, n_keep, n_rand, K)

    # QR de B_T_full calculé une seule fois pour toute la couche
    Ao = _orth(B_T_full)  # (s, d)

    for i in range(n_rand):
        candidate_seeds = np.random.randint(0, 2**31, size=K)
        best_seed, best_dist = None, np.inf

        # QR de BNRC_current calculé une fois pour les K candidats
        Q_cur = _orth(BNRC_current)  # (n_keep+i, d)

        for sd in candidate_seeds:
            rv   = _prng_vector(int(sd), d)
            dist = _grassmann_distance_fast(Ao, Q_cur, rv, s)
            if dist < best_dist:
                best_dist, best_seed = dist, int(sd)

        rv_best      = _prng_vector(best_seed, d)
        BNRC_current = np.vstack([BNRC_current, rv_best[np.newaxis, :]])
        random_vectors.append(rv_best)
        selected_seeds.append(best_seed)
        grassmann_dists.append(best_dist)
        logger.debug("  vecteur %d/%d | seed=%d | Grassmann=%.4f",
                     i + 1, n_rand, best_seed, best_dist)

    rand_arr   = (np.stack(random_vectors, axis=0)
                  if random_vectors else np.empty((0, d), np.float32))
    BNRC_final = (np.vstack([B_T_kept, rand_arr])
                  if rand_arr.shape[0] > 0 else B_T_kept.copy())

    logger.info("→ BNRC [%s] shape=%s (%d PCA + %d LFSR)",
                name, BNRC_final.shape, n_keep, n_rand)
    if n_rand > 0:
        final_dist = _grassmann_distance(B_T_full, BNRC_final)
        logger.info("→ Distance Grassmann finale : %.4f", final_dist)

    return dict(
        BNRC            = BNRC_final.astype(np.float32),
        BNRC_keep       = B_T_kept.astype(np.float32),
        BNRC_rand       = rand_arr.astype(np.float32),
        B_T_target      = B_T_target.astype(np.float32),
        selected_seeds  = selected_seeds,
        grassmann_dists = grassmann_dists,
        n_keep          = n_keep,
        n_rand          = n_rand,
        t               = s,
    )


class PseudoRandomFilterSelector:
    """Étape 3 NRC — construction de BNRC pour toutes les couches Conv2d.

    Usage :
        selector = PseudoRandomFilterSelector(p=0.5, K=100, seed=42)
        selector.select(analyzer.results)
        # selector.results["features.0"] → dict BNRC

    Reprise possible via cache_dir : chaque couche terminée est sauvegardée
    dans un fichier .pkl. Au redémarrage, les couches déjà traitées sont
    chargées depuis le cache sans recalcul.
    """

    def __init__(self, randomization_ratio: float = 0.5,
                 K: int = 100, seed: int = 0):
        self.randomization_ratio = randomization_ratio
        self.K    = K
        self.seed = seed
        self.results: dict[str, dict] = {}

    @staticmethod
    def _cache_path(cache_dir: Path, layer_idx: int, layer_name: str) -> Path:
        safe = layer_name.replace(".", "__")
        return cache_dir / f"{layer_idx:03d}_{safe}.pkl"

    def select(self, analyses: dict[str, dict],
               cache_dir: str | None = None) -> "PseudoRandomFilterSelector":
        self.results = {}
        cache = Path(cache_dir) if cache_dir else None
        if cache:
            cache.mkdir(parents=True, exist_ok=True)

        for i, (name, analysis) in enumerate(analyses.items()):
            ckpt = self._cache_path(cache, i, name) if cache else None

            if ckpt and ckpt.exists():
                with open(ckpt, "rb") as f:
                    self.results[name] = pickle.load(f)
                logger.info("[%s] chargé depuis cache (%s)", name, ckpt.name)
                continue

            self.results[name] = _build_bnrc(
                analysis, self.randomization_ratio,
                self.K, self.seed + i, name,
            )

            if ckpt:
                with open(ckpt, "wb") as f:
                    pickle.dump(self.results[name], f)
                logger.info("[%s] sauvegardé → %s", name, ckpt.name)

        return self
