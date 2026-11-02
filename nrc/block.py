from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NRCConvBlock(nn.Module):
    """Bloc NRC — remplacement drop-in d'une couche Conv2d.

    Structure :
        conv_keep   : Conv2d(c_in, n_keep, kH, kW, bias=False)  BNRC_keep — GELÉ, stocké
        rand_weight : buffer (n_rand, c_in, kH, kW)             BNRC_rand — régénéré via
                      seed + PRNG, NON stocké (non-persistent buffer, exclu de .parameters())
        conv_coord  : Conv2d(t, c_out, 1, 1,  bias=True)        Y_NRC — ENTRAÎNABLE
        mean_conv   : Conv2d(c_in, 1, kH, kW, bias=False)       w_mean — GELÉ

    Conformément à la thèse (Crozet 2024), les filtres pseudo-aléatoires ne
    doivent pas être stockés : seule leur graine l'est. Ils sont donc gardés
    en buffer non persistant plutôt qu'en nn.Parameter, pour que le comptage
    des paramètres (et donc le gain de compression) reflète correctement
    l'effet du paramètre p.

    Reconstruction : y ≈ conv_coord(concat(conv_keep(x), rand_conv(x))) + mean_conv(x)
    """

    def __init__(
        self,
        c_in:      int,
        c_out:     int,
        kH:        int,
        kW:        int,
        BNRC_keep: np.ndarray,
        BNRC_rand: np.ndarray,
        Y_NRC:     np.ndarray,
        bias_vec:  np.ndarray | None = None,
        w_mean:    np.ndarray | None = None,
        stride:    int = 1,
        padding:   int = 0,
        dilation:  int = 1,
    ):
        super().__init__()
        n_keep = BNRC_keep.shape[0]
        n_rand = BNRC_rand.shape[0]
        t = n_keep + n_rand

        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.n_keep, self.n_rand = n_keep, n_rand

        # ── Filtres PCA conservés (gelés, stockés) ─────────────────────
        self.conv_keep = None
        if n_keep > 0:
            self.conv_keep = nn.Conv2d(
                c_in, n_keep, kernel_size=(kH, kW), bias=False,
                stride=stride, padding=padding, dilation=dilation)
            self.conv_keep.weight.data.copy_(
                torch.from_numpy(BNRC_keep).view(n_keep, c_in, kH, kW))
            for p in self.conv_keep.parameters():
                p.requires_grad = False

        # ── Filtres pseudo-aléatoires (régénérés depuis la graine) ─────
        # Buffer non persistant : présent en mémoire pour l'inférence
        # courante mais jamais compté comme paramètre ni sauvegardé —
        # seule la graine (voir manifest["selected_seeds"]) doit l'être.
        if n_rand > 0:
            self.register_buffer(
                "rand_weight",
                torch.from_numpy(BNRC_rand).view(n_rand, c_in, kH, kW),
                persistent=False,
            )
        else:
            self.rand_weight = None

        # ── Coordonnées Y_NRC (entraînables) ──────────────────────────
        self.conv_coord = nn.Conv2d(t, c_out, kernel_size=1, bias=True)
        self.conv_coord.weight.data.copy_(
            torch.from_numpy(Y_NRC).view(c_out, t, 1, 1))
        if bias_vec is not None:
            self.conv_coord.bias.data.copy_(torch.from_numpy(bias_vec))

        # ── Filtre moyen (gelé) ────────────────────────────────────────
        self.mean_conv = None
        if w_mean is not None:
            self.mean_conv = nn.Conv2d(
                c_in, 1, kernel_size=(kH, kW), bias=False,
                stride=stride, padding=padding, dilation=dilation)
            self.mean_conv.weight.data.copy_(
                torch.from_numpy(w_mean.flatten()).view(1, c_in, kH, kW))
            for p in self.mean_conv.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        if self.conv_keep is not None:
            feats.append(self.conv_keep(x))
        if self.rand_weight is not None:
            feats.append(F.conv2d(
                x, self.rand_weight, bias=None,
                stride=self.stride, padding=self.padding, dilation=self.dilation))
        base_out = feats[0] if len(feats) == 1 else torch.cat(feats, dim=1)

        out = self.conv_coord(base_out)
        if self.mean_conv is not None:
            out = out + self.mean_conv(x)
        return out
