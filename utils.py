# ─── Utilitaires partagés : LFSR & Grassmann ─────────────────────────────────

import logging
import os
from datetime import datetime

import numpy as np

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", level: int = logging.INFO) -> str:
    """Configure le logging global : console + fichier horodaté.

    Returns:
        Chemin du fichier de log créé.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"run_{timestamp}.log")

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root.addHandler(ch)

    logger.info("Logging initialisé → %s", log_path)
    return log_path


# ─────────────────────────────────────────────────────────────────────────────
# Distances sur Grassmann
# ─────────────────────────────────────────────────────────────────────────────

def _grassmann_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Distance de Grassmann entre les sous-espaces engendrés par A et B."""
    def _orth(M):
        Q, _ = np.linalg.qr(M.T)
        return Q[:, :M.shape[0]].T

    Ao = _orth(A.astype(np.float64))
    Bo = _orth(B.astype(np.float64))

    sigma = np.linalg.svd(Ao @ Bo.T, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)

    angles = np.arccos(sigma)

    # Gestion dimensions différentes
    n_missing = abs(Ao.shape[0] - Bo.shape[0])
    missing_term = n_missing * (np.pi / 2) ** 2

    return float(np.sqrt(np.sum(angles**2) + missing_term))


def _projection_residual(target: np.ndarray, candidate: np.ndarray) -> float:
    """Part du sous-espace cible NON capturée par le sous-espace candidat."""
    Q_t, _ = np.linalg.qr(target.astype(np.float64).T)
    Q_c, _ = np.linalg.qr(candidate.astype(np.float64).T)

    proj = Q_c @ (Q_c.T @ Q_t)
    return float(np.sum((Q_t - proj) ** 2))


# ─────────────────────────────────────────────────────────────────────────────
# LFSR Fibonacci (conforme à ton schéma)
# ─────────────────────────────────────────────────────────────────────────────

def _lfsr_fibonacci_uint32(state: int, taps: list[int], n_bits: int = 32) -> int:
    """
    LFSR Fibonacci (shift → injection du feedback au MSB).

    Args:
        state  : état courant (32 bits)
        taps   : positions des taps (ex: [32, 22, 2, 1])
        n_bits : taille du registre

    Returns:
        nouvel état
    """
    state &= (1 << n_bits) - 1

    if state == 0:
        state = 0xACE1FACE  # éviter état nul

    # 1) Calcul du feedback (XOR des taps)
    feedback = 0
    for t in taps:
        feedback ^= (state >> (t - 1)) & 1

    # 2) Shift + injection
    state = (state >> 1) | (feedback << (n_bits - 1))

    return state


# ─────────────────────────────────────────────────────────────────────────────
# PRNG basé sur LFSR Fibonacci
# ─────────────────────────────────────────────────────────────────────────────

def _prng_vector(seed: int, dim: int) -> np.ndarray:
    """
    Génère un vecteur pseudo-aléatoire normalisé via LFSR Fibonacci 32-bit.

    Utilise le polynôme primitif équivalent à 0x80200003 :
        taps = [32, 22, 2, 1]
    """

    TAPS = [32, 22, 2, 1]

    state = int(seed) & 0xFFFFFFFF
    if state == 0:
        state = 0xACE1FACE

    def _next_uint32() -> int:
        nonlocal state
        state = _lfsr_fibonacci_uint32(state, TAPS)
        return state

    def _uniform() -> float:
        # Normalisation vers (-0.5, 0.5)
        return (_next_uint32() + 0.5) / 4_294_967_296.0 - 0.5

    v = np.array([_uniform() for _ in range(dim)], dtype=np.float32)

    return v / (np.linalg.norm(v) + 1e-8)