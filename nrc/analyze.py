from __future__ import annotations

import logging

import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


def _analyze_conv(conv: nn.Conv2d, energy_threshold: float, name: str) -> dict:
    """Analyse PCA d'une couche Conv2d (étape 2 NRC).

    Args:
        conv             : couche Conv2d à analyser
        energy_threshold : seuil eTh — on conserve les s premiers vecteurs
                           propres tels que trace(D_s)/trace(D) >= eTh
        name             : label pour les logs

    Returns:
        dict avec les matrices PCA nécessaires aux étapes suivantes.
    """
    W_tensor = conv.weight.data.detach().cpu()
    n_filters, c_in, kH, kW = W_tensor.shape
    d = c_in * kH * kW

    W      = W_tensor.view(n_filters, d).numpy().astype(np.float64).T  # (d, n_filters)
    w_mean = W.mean(axis=1, keepdims=True)
    W_C    = W - w_mean                                                 # (d, n_filters)

    n_max = min(n_filters, d)
    pca = PCA(n_components=n_max)
    pca.fit(W_C.T)
    B_ED        = pca.components_
    eigenvalues = pca.explained_variance_

    trace_D  = eigenvalues.sum()
    cumtrace = np.cumsum(eigenvalues) / trace_D
    s = max(2, int(np.searchsorted(cumtrace, energy_threshold) + 1))
    s = min(s, n_max)

    B_T = B_ED[:s]
    Y_T = W_C.T @ B_T.T

    logger.info("[%s] s=%d/%d vecteurs PCA | énergie=%.1f%%",
                name, s, n_max, cumtrace[s - 1] * 100)

    return dict(
        s               = s,
        n_filters       = n_filters,
        filter_shape    = (c_in, kH, kW),
        w_mean          = w_mean.astype(np.float32),
        W_original      = W.astype(np.float32),
        W_C             = W_C.astype(np.float32),
        B_T             = B_T.astype(np.float32),
        B_ED            = B_ED.astype(np.float32),
        Y_T             = Y_T.astype(np.float32),
        important_indices   = list(range(s)),
        unimportant_indices = list(range(s, n_max)),
        explained_variance  = pca.explained_variance_ratio_,
        energy_captured     = float(cumtrace[s - 1]),
    )


class FilterAnalyzer:
    """Étape 2 NRC — analyse PCA de toutes les couches Conv2d d'un modèle.

    Usage :
        layers = {name: mod for name, mod in model.named_modules()
                  if isinstance(mod, nn.Conv2d)}
        analyzer = FilterAnalyzer(energy_threshold=0.9).analyze(layers)
        # analyzer.results["features.0"] → dict PCA
    """

    def __init__(self, energy_threshold: float = 0.90):
        self.energy_threshold = energy_threshold
        self.results: dict[str, dict] = {}

    def analyze(self, layers: dict[str, nn.Conv2d]) -> "FilterAnalyzer":
        self.results = {
            name: _analyze_conv(conv, self.energy_threshold, name)
            for name, conv in layers.items()
        }
        return self
