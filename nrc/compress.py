"""Compression NRC générique — fonctionne avec n'importe quel nn.Module."""

from __future__ import annotations

import logging
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn

from nrc.block   import NRCConvBlock
from nrc.analyze import FilterAnalyzer
from nrc.select  import PseudoRandomFilterSelector

logger = logging.getLogger(__name__)


# ─── Utilitaires ─────────────────────────────────────────────────────────────

def get_conv_layers(model: nn.Module) -> dict[str, nn.Conv2d]:
    """Retourne {nom: Conv2d} pour toutes les couches Conv2d du modèle."""
    return {name: mod
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Conv2d)}


def count_conv_params(model: nn.Module) -> int:
    """Nombre total de paramètres dans les couches Conv2d / NRCConvBlock.

    Les blocs NRCConvBlock contiennent eux-mêmes des Conv2d enfants ; on évite
    de les compter deux fois en ne descendant pas dans un NRCConvBlock.
    """
    counted_prefixes: set[str] = set()
    total = 0
    for name, mod in model.named_modules():
        if any(name.startswith(p + ".") for p in counted_prefixes):
            continue
        if isinstance(mod, (nn.Conv2d, NRCConvBlock)):
            total += sum(p.numel() for p in mod.parameters())
            counted_prefixes.add(name)
    return total


def count_conv_macs(model: nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 32, 32)) -> int:
    """Nombre de MAC (multiplications-accumulations) dans les couches conv.

    Calculé via une passe forward factice + hooks, pour tenir compte des
    tailles de sortie réelles (stride/padding/dilation). Un NRCConvBlock est
    traité comme un tout (branches conv_keep + pseudo-aléatoire + conv_coord
    + mean_conv), sans descendre dans ses Conv2d enfants, afin de ne pas les
    compter deux fois — même logique que count_conv_params.
    """
    macs = 0
    hooks = []
    counted_prefixes: set[str] = set()

    def _conv2d_hook(mod, _inp, out):
        nonlocal macs
        h, w = out.shape[-2:]
        macs += mod.weight.numel() * h * w

    def _nrc_block_hook(mod, _inp, out):
        nonlocal macs
        h, w = out.shape[-2:]
        if mod.conv_keep is not None:
            macs += mod.conv_keep.weight.numel() * h * w
        if mod.rand_weight is not None:
            macs += mod.rand_weight.numel() * h * w
        macs += mod.conv_coord.weight.numel() * h * w
        if mod.mean_conv is not None:
            macs += mod.mean_conv.weight.numel() * h * w

    for name, mod in model.named_modules():
        if any(name.startswith(p + ".") for p in counted_prefixes):
            continue
        if isinstance(mod, NRCConvBlock):
            hooks.append(mod.register_forward_hook(_nrc_block_hook))
            counted_prefixes.add(name)
        elif isinstance(mod, nn.Conv2d):
            hooks.append(mod.register_forward_hook(_conv2d_hook))

    was_training = model.training
    device = next(model.parameters()).device
    try:
        model.eval()
        with torch.no_grad():
            model(torch.zeros(*input_size, device=device))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)

    return macs


def _set_module(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Remplace un sous-module par chemin (e.g. 'features.0')."""
    parts  = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


# ─── Compression ─────────────────────────────────────────────────────────────

def compress(
    model:          nn.Module,
    eth:            float,
    p:              float,
    K:              int  = 100,
    seed:           int  = 42,
    freeze_linear:  bool = False,
    cache_dir:      str | None = None,
) -> tuple[nn.Module, dict]:
    """Remplace toutes les Conv2d par des NRCConvBlock.

    Applique la méthode NRC (PCA + pseudo-randomisation LFSR) à chaque couche
    Conv2d trouvée dans *model*, indépendamment de l'architecture.

    Args:
        model          : modèle source (non modifié — une copie est utilisée)
        eth            : seuil d'énergie PCA
        p              : ratio de pseudo-randomisation (0 = tout PCA, 1 = tout LFSR)
        K              : candidats LFSR par vecteur
        seed           : graine de base (incrémentée par couche)
        freeze_linear  : si True, gèle toutes les couches nn.Linear

    Returns:
        (compressed_model, manifest)

        manifest["layers"][layer_name] contient les seeds et métadonnées
        suffisantes pour reconstruire BNRC sans stocker ses vecteurs.
    """
    model = deepcopy(model)

    conv_layers = get_conv_layers(model)
    if not conv_layers:
        logger.warning("Aucune couche Conv2d trouvée dans le modèle.")
        return model, {"eth": eth, "p": p, "K": K, "seed": seed, "layers": {}}

    analyzer = FilterAnalyzer(energy_threshold=eth)
    analyzer.analyze(conv_layers)

    selector = PseudoRandomFilterSelector(randomization_ratio=p, K=K, seed=seed)
    selector.select(analyzer.results, cache_dir=cache_dir)

    manifest: dict = {"eth": eth, "p": p, "K": K, "seed": seed, "layers": {}}

    for layer_name, conv in conv_layers.items():
        analysis  = analyzer.results[layer_name]
        bnrc_data = selector.results[layer_name]

        n_f, c_in, kH, kW = conv.weight.shape
        Y_NRC    = (analysis["W_C"].T @ bnrc_data["BNRC"].T).astype(np.float32)
        bias_vec = (conv.bias.data.cpu().numpy()
                    if conv.bias is not None else None)

        # Préserver stride, padding, dilation de la Conv2d originale
        stride   = conv.stride[0]   if isinstance(conv.stride,   tuple) else conv.stride
        padding  = conv.padding[0]  if isinstance(conv.padding,  tuple) else conv.padding
        dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation

        nrc_block = NRCConvBlock(
            c_in=c_in, c_out=n_f, kH=kH, kW=kW,
            BNRC_keep = bnrc_data["BNRC_keep"],
            BNRC_rand = bnrc_data["BNRC_rand"],
            Y_NRC     = Y_NRC,
            bias_vec  = bias_vec,
            w_mean    = analysis["w_mean"].astype(np.float32),
            stride    = stride,
            padding   = padding,
            dilation  = dilation,
        )

        _set_module(model, layer_name, nrc_block)

        manifest["layers"][layer_name] = {
            "original_shape": list(conv.weight.shape),   # [n_f, c_in, kH, kW]
            "s":              int(analysis["s"]),
            "n_keep":         int(bnrc_data["n_keep"]),
            "n_rand":         int(bnrc_data["n_rand"]),
            "selected_seeds": [int(s) for s in bnrc_data["selected_seeds"]],
            # arrays nécessaires à la reconstruction (sans re-entraîner)
            "BNRC_keep":      bnrc_data["BNRC_keep"],    # np.ndarray (n_keep × d)
            "w_mean":         analysis["w_mean"],        # np.ndarray (d, 1)
        }

        logger.info("Couche %-20s compressée : %d params → %d params",
                    layer_name,
                    sum(p.numel() for p in conv.parameters()),
                    sum(p.numel() for p in nrc_block.parameters()))

    if freeze_linear:
        for mod in model.modules():
            if isinstance(mod, nn.Linear):
                for param in mod.parameters():
                    param.requires_grad = False

    n_orig = count_conv_params(deepcopy(model))  # après remplacement
    logger.info("Compression terminée | eth=%.2f p=%.2f | conv params=%d",
                eth, p, n_orig)

    return model, manifest


# ─── Pipeline séquentiel : pseudo-randomisation après PCA+retrain ─────────────

def apply_pseudo_after_pca(
    pca_model: nn.Module,
    manifest:  dict,
    p:         float,
    K:         int = 100,
    seed:      int = 42,
    cache_dir: str | None = None,
) -> tuple[nn.Module, dict]:
    """Remplace, après coup, une fraction p des filtres PCA d'un modèle déjà
    compressé (via compress(..., p=0)) et éventuellement ré-entraîné, par des
    filtres pseudo-aléatoires LFSR — SANS toucher aux poids déjà appris.

    Contrairement à compress(), qui repart du modèle original et effectue PCA
    + pseudo-randomisation en une seule passe, cette fonction agit comme une
    étape 2 d'un pipeline séquentiel : *pca_model* doit avoir été produit par
    compress(model, eth, p=0.0) (typiquement après fine-tuning). conv_coord
    (les coordonnées apprises) et mean_conv restent gelés à leur valeur
    entraînée ; seuls les derniers filtres de la base PCA sont remplacés par
    les candidats LFSR les plus proches (même recherche gloutonne par distance
    de Grassmann que compress()). Aucun ré-entraînement n'est effectué ici :
    l'objectif est de mesurer la dégradation d'accuracy induite par cette
    substitution seule.
    """
    model = deepcopy(pca_model)

    # BNRC_keep stocké dans le manifest à p=0 == base PCA complète (B_T), donc
    # aucune ré-analyse du modèle original n'est nécessaire.
    analyses = {
        name: {"B_T": np.asarray(layer_manifest["BNRC_keep"], dtype=np.float64)}
        for name, layer_manifest in manifest["layers"].items()
    }

    selector = PseudoRandomFilterSelector(randomization_ratio=p, K=K, seed=seed)
    selector.select(analyses, cache_dir=cache_dir)

    new_manifest: dict = {
        "eth": manifest["eth"], "p": p, "K": K, "seed": seed, "layers": {},
    }

    for name, block in model.named_modules():
        if not isinstance(block, NRCConvBlock):
            continue

        bnrc_data = selector.results[name]
        n_keep, n_rand = bnrc_data["n_keep"], bnrc_data["n_rand"]

        old_keep = block.conv_keep.weight.data
        c_in, kH, kW = old_keep.shape[1], old_keep.shape[2], old_keep.shape[3]

        # ── Filtres PCA conservés : préfixe des filtres déjà entraînés ──────
        if n_keep > 0:
            new_conv_keep = nn.Conv2d(
                c_in, n_keep, kernel_size=(kH, kW), bias=False,
                stride=block.stride, padding=block.padding, dilation=block.dilation
            ).to(old_keep.device)
            new_conv_keep.weight.data.copy_(old_keep[:n_keep])
            for param in new_conv_keep.parameters():
                param.requires_grad = False
            block.conv_keep = new_conv_keep
        else:
            block.conv_keep = None

        # ── Filtres pseudo-aléatoires : remplacent le reste de la base PCA ──
        if hasattr(block, "rand_weight"):
            delattr(block, "rand_weight")
        if n_rand > 0:
            rand_arr = bnrc_data["BNRC_rand"].reshape(n_rand, c_in, kH, kW)
            block.register_buffer(
                "rand_weight",
                torch.from_numpy(rand_arr.astype(np.float32)).to(old_keep.device),
                persistent=False,
            )
        else:
            block.rand_weight = None

        block.n_keep, block.n_rand = n_keep, n_rand

        new_manifest["layers"][name] = {
            **manifest["layers"][name],
            "n_keep":         n_keep,
            "n_rand":         n_rand,
            "selected_seeds": [int(s) for s in bnrc_data["selected_seeds"]],
            "BNRC_keep":      bnrc_data["BNRC_keep"],  # sous-ensemble, pas la base complète
        }

    n_final = count_conv_params(model)
    logger.info("Pseudo-après-PCA terminé | p=%.2f | conv params=%d", p, n_final)

    return model, new_manifest
