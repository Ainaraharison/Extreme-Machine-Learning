#!/usr/bin/env python3
"""Test de fumée NRC — compresse rapidement chaque modèle enregistré.

Lance `run.py` en sous-processus, une fois par modèle (et éventuellement
plusieurs datasets), avec un budget d'époques et un sweep (eth × p)
volontairement réduits. L'objectif n'est pas de mesurer l'accuracy ou le
gain de compression, mais de détecter rapidement une régression qui ferait
planter le pipeline sur une architecture donnée (erreur de forme, couche
groupée non supportée, etc. — cf. §"Ajout d'un nouveau modèle" de doc.tex).

Chaque modèle tourne dans son propre sous-processus Python, ce qui évite
toute pollution d'état (mémoire CUDA, RNG) entre deux modèles testés à la
suite.

Usage :
    python test_all_models.py
    python test_all_models.py --models lenet resnet18
    python test_all_models.py --dataset cifar100
    python test_all_models.py --full                      # + retrain PCA/pseudo
    python test_all_models.py --epochs-initial 2 --eth 0.5 0.9 --p 0.0 0.5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from run import DATASET_REGISTRY, MODEL_REGISTRY

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY),
                   choices=list(MODEL_REGISTRY),
                   help="Modèles à tester (défaut : tous ceux de MODEL_REGISTRY)")
    p.add_argument("--dataset", default="cifar10", choices=list(DATASET_REGISTRY),
                   help="Dataset utilisé pour le test")
    p.add_argument("--epochs-initial", type=int, default=1,
                   help="Époques d'entraînement initial (rapide par défaut)")
    p.add_argument("--eth", type=float, nargs="+", default=[0.5],
                   help="Seuil(s) d'énergie PCA à tester")
    p.add_argument("--p", type=float, nargs="+", default=[0.0, 0.5],
                   help="Proportion(s) de pseudo-randomisation à tester "
                        "(0.0 exerce la branche PCA pure, 0.5 la branche hybride)")
    p.add_argument("--K", type=int, default=20,
                   help="Candidats LFSR par vecteur (réduit pour aller vite)")
    p.add_argument("--full", action="store_true",
                   help="Teste aussi --retrain-after-pca et "
                        "--retrain-after-pseudo (1 époque de FT chacun)")
    p.add_argument("--timeout", type=int, default=1200,
                   help="Timeout par modèle, en secondes")
    p.add_argument("--output-dir", default="results/smoke_test",
                   help="Dossier de sortie des fichiers Excel générés")
    p.add_argument("--python", default=sys.executable,
                   help="Interpréteur Python à utiliser pour les sous-processus")
    return p.parse_args()


def _tail(text: str, n: int = 25) -> str:
    lines = text.strip().splitlines()
    return "\n".join(lines[-n:])


def run_one(args, model: str) -> dict:
    cmd = [
        args.python, "run.py",
        "--model", model,
        "--dataset", args.dataset,
        "--epochs-initial", str(args.epochs_initial),
        "--eth", *[str(v) for v in args.eth],
        "--p", *[str(v) for v in args.p],
        "--K", str(args.K),
        "--no-cache",
        "--output-dir", args.output_dir,
        "--output", str(Path(args.output_dir) / f"smoke_{model}_{args.dataset}.xlsx"),
    ]
    if args.full:
        cmd += [
            "--retrain-after-pca", "--retrain-after-pseudo",
            "--epochs-retrain-pca", "1", "--epochs-retrain-pseudo", "1",
        ]

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, capture_output=True, text=True,
            timeout=args.timeout,
        )
        elapsed = time.monotonic() - start
        ok = proc.returncode == 0
        return {
            "model": model, "ok": ok, "elapsed": elapsed,
            "returncode": proc.returncode,
            "stderr_tail": "" if ok else _tail(proc.stderr),
        }
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return {
            "model": model, "ok": False, "elapsed": elapsed,
            "returncode": None, "stderr_tail": f"TIMEOUT après {args.timeout}s",
        }


def main():
    args = parse_args()
    print(f"Test de fumée NRC — {len(args.models)} modèle(s) | dataset={args.dataset}")
    print(f"  eth={args.eth} | p={args.p} | K={args.K} | "
          f"epochs_initial={args.epochs_initial} | full={args.full}\n")

    results = []
    for model in args.models:
        print(f"─── {model:<16} ", end="", flush=True)
        res = run_one(args, model)
        results.append(res)
        status = "OK" if res["ok"] else "ÉCHEC"
        print(f"→ {status} ({res['elapsed']:.1f}s)")
        if not res["ok"] and res["stderr_tail"]:
            last_line = res["stderr_tail"].splitlines()[-1]
            print(f"    dernière ligne stderr : {last_line}")

    n_ok = sum(r["ok"] for r in results)
    print("\n" + "═" * 60)
    print(f"Résumé : {n_ok}/{len(results)} modèle(s) OK")
    for r in results:
        mark = "OK  " if r["ok"] else "FAIL"
        print(f"  [{mark}] {r['model']:<16} {r['elapsed']:6.1f}s")

    if n_ok < len(results):
        print("\nDétails des échecs (fin de stderr) :")
        for r in results:
            if not r["ok"]:
                print(f"\n--- {r['model']} (code retour {r['returncode']}) ---")
                print(r["stderr_tail"])
        sys.exit(1)


if __name__ == "__main__":
    main()
