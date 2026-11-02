"""Sérialisation du manifest NRC.

Sauvegarde : JSON (métadonnées + seeds) + NPZ (tableaux numpy).
Chargement : reconstruit le manifest complet depuis ces deux fichiers.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_manifest(manifest: dict, path: str) -> None:
    """Sauvegarde manifest → <path>.json + <path>_arrays.npz.

    Les seeds et métadonnées vont dans le JSON (lisible humainement).
    Les tableaux numpy (BNRC_keep, w_mean) vont dans le NPZ.
    """
    base = str(path).removesuffix(".json")

    meta: dict = {k: v for k, v in manifest.items() if k != "layers"}
    meta["layers"] = {}
    arrays: dict[str, np.ndarray] = {}

    for name, layer in manifest["layers"].items():
        safe = name.replace(".", "__")
        meta["layers"][name] = {
            "original_shape": layer["original_shape"],
            "s":              layer["s"],
            "n_keep":         layer["n_keep"],
            "n_rand":         layer["n_rand"],
            "selected_seeds": layer["selected_seeds"],
        }
        arrays[f"{safe}__BNRC_keep"] = np.asarray(layer["BNRC_keep"])
        arrays[f"{safe}__w_mean"]    = np.asarray(layer["w_mean"])

    json_path = f"{base}.json"
    npz_path  = f"{base}_arrays.npz"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    np.savez(npz_path, **arrays)

    logger.info("Manifest sauvegardé → %s  +  %s", json_path, npz_path)


def load_manifest(path: str) -> dict:
    """Charge le manifest depuis <path>.json + <path>_arrays.npz."""
    base = str(path).removesuffix(".json")

    with open(f"{base}.json", encoding="utf-8") as f:
        meta = json.load(f)

    arrays = np.load(f"{base}_arrays.npz")
    for name in meta["layers"]:
        safe = name.replace(".", "__")
        meta["layers"][name]["BNRC_keep"] = arrays[f"{safe}__BNRC_keep"]
        meta["layers"][name]["w_mean"]    = arrays[f"{safe}__w_mean"]

    return meta


def save_compressed_model(
    model:    nn.Module,
    manifest: dict,
    directory: str,
    prefix:   str = "model",
) -> None:
    """Sauvegarde état du modèle compressé + manifest dans un répertoire.

    Crée :
        <directory>/<prefix>_state.pth    — state_dict PyTorch (poids entraînables)
        <directory>/<prefix>_manifest.json
        <directory>/<prefix>_manifest_arrays.npz
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)

    state_path = out / f"{prefix}_state.pth"
    torch.save(model.state_dict(), state_path)
    logger.info("State dict → %s", state_path)

    save_manifest(manifest, str(out / f"{prefix}_manifest.json"))
