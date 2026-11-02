#!/usr/bin/env python3
# ─── CLI NRC : sweep en grille (eth × p) + export Excel ──────────────────────

import argparse
import logging
import os
import shlex
import sys
from copy import deepcopy
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from openpyxl.styles import PatternFill

from utils import setup_logging
from models.lenet          import LeNet
from models.vgg16          import VGG16Cifar
from models.vgg16_standard import VGG16Standard
from models.vgg_mini       import VGGMini
from models.resnet18       import ResNet18Cifar
from models.resnet50       import ResNet50Cifar
from nrc.compress    import compress, count_conv_params, count_conv_macs, apply_pseudo_after_pca
from nrc.train       import train_one_epoch, evaluate
from nrc.serialize   import save_compressed_model

# ─── Registre des modèles ─────────────────────────────────────────────────────
# Chaque entrée : nom → callable(args) → nn.Module
# args.num_classes est déterminé à partir de args.dataset (cf. DATASET_REGISTRY)
MODEL_REGISTRY = {
    "lenet":          lambda args: LeNet(n_filters=args.n_filters, num_classes=args.num_classes),
    "vgg16":          lambda args: VGG16Cifar(num_classes=args.num_classes),
    "vgg16_standard": lambda args: VGG16Standard(num_classes=args.num_classes),
    "resnet18":       lambda args: ResNet18Cifar(num_classes=args.num_classes),
    "resnet50":       lambda args: ResNet50Cifar(num_classes=args.num_classes),
    "vgg_mini":       lambda args: VGGMini(num_classes=args.num_classes),
}

# ─── Registre des datasets ─────────────────────────────────────────────────────
# Même format que CIFAR-10 (32×32 RGB) : seul le nombre de classes change.
DATASET_REGISTRY = {
    "cifar10":  {"cls": torchvision.datasets.CIFAR10,  "num_classes": 10},
    "cifar100": {"cls": torchvision.datasets.CIFAR100, "num_classes": 100},
}


# ─── Fichier d'arguments ─────────────────────────────────────────────────────
# Permet : python run.py --config config.txt      (ou  python run.py @config.txt)
#
# Format accepté (les deux styles peuvent être mélangés) :
#   # commentaire ; les lignes vides sont ignorées
#   --model vgg16                 → style ligne de commande
#   eth = 0.25 0.50 0.75 0.90     → style clé = valeur(s)
#   epochs_initial: 10            → « : » et « _ » acceptés aussi
#   retrain-after-pca = true      → options booléennes (true/false, oui/non, 1/0)

_TRUE_WORDS  = {"1", "true", "yes", "y", "on", "oui", "vrai"}
_FALSE_WORDS = {"0", "false", "no", "n", "off", "non", "faux"}
_QUOTES = "\"'"


def _option_specs(parser):
    """Associe chaque option (« --eth », « -c »…) à son nombre de valeurs.

    0 → drapeau booléen, "+" / "*" / n>1 → liste, None / 1 → valeur unique.
    """
    return {opt: action.nargs
            for action in parser._actions
            for opt in action.option_strings}


def _takes_list(nargs):
    return nargs in ("+", "*") or (isinstance(nargs, int) and nargs > 1)


def _split_tokens(line):
    """Découpe une ligne en tokens, en respectant les guillemets.

    Le découpage naïf est privilégié quand il n'y a pas de guillemets : cela
    évite que shlex ne mange les antislashs des chemins Windows.
    """
    if not any(q in line for q in _QUOTES):
        return line.split()
    return [t.strip(_QUOTES) for t in shlex.split(line, posix=False)]


def _strip_comment(line):
    """Retire un commentaire « # … » situé hors guillemets."""
    out, quote = [], None
    for ch in line:
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
        elif ch in _QUOTES:
            quote = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
    return "".join(out).strip()


def _line_to_args(line, specs):
    """Convertit une ligne du fichier de configuration en tokens argparse."""
    line = _strip_comment(line)
    if not line:
        return []

    # Style ligne de commande : découpage identique à celui d'un shell.
    if line.startswith("-"):
        return _split_tokens(line)

    # Style « clé = valeur » / « clé : valeur »
    for sep in ("=", ":"):
        if sep in line:
            key, value = line.split(sep, 1)
            has_sep = True
            break
    else:
        key, value, has_sep = line, "", False

    key = "--" + key.strip().lstrip("-").replace("_", "-")
    value = value.strip()
    if key not in specs:
        raise ValueError("option inconnue : %s" % key)
    nargs = specs[key]

    if nargs == 0:                                # drapeau booléen
        word = value.lower()
        if not value or word in _TRUE_WORDS:
            return [key]
        if word in _FALSE_WORDS:
            return []
        raise ValueError("%s attend une valeur booléenne (true/false), "
                         "reçu : %s" % (key, value))

    if _takes_list(nargs):                        # liste de valeurs
        values = _split_tokens(value)
        if not values:
            raise ValueError("aucune valeur fournie pour %s" % key)
        return [key] + values

    # Valeur unique : la fin de ligne est prise telle quelle, ce qui permet
    # les chemins Windows non quotés (« save_dir = E:\mes résultats »).
    if not value:
        if not has_sep:
            raise ValueError("aucune valeur fournie pour %s" % key)
        return [key, ""]                          # « output = » → valeur vide
    if len(value) > 1 and value[0] == value[-1] and value[0] in _QUOTES:
        value = value[1:-1]
    return [key, value]


def _read_args_file(path, specs):
    """Lit un fichier texte d'arguments et renvoie la liste des tokens CLI."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            lines = fh.readlines()
    except OSError as exc:
        raise SystemExit("Fichier d'arguments illisible : %s" % exc)

    tokens = []
    for num, line in enumerate(lines, 1):
        try:
            tokens.extend(_line_to_args(line, specs))
        except ValueError as exc:
            raise SystemExit("%s, ligne %d : %s" % (path, num, exc))
    return tokens


class _ConfigArgParser(argparse.ArgumentParser):
    """ArgumentParser acceptant « @fichier.txt » avec le même format étendu."""

    def convert_arg_line_to_args(self, arg_line):
        try:
            return _line_to_args(arg_line, _option_specs(self))
        except ValueError as exc:
            self.error(str(exc))


def parse_args(argv=None):
    p = _ConfigArgParser(
        description="Pipeline NRC — sweep (eth × p) + export Excel",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        fromfile_prefix_chars="@",
    )

    # ── Fichier d'arguments ──────────────────────────────────────────────────
    p.add_argument("-c", "--config", type=str, default=None,
                   metavar="FICHIER",
                   help="Fichier texte contenant les arguments "
                        "(équivalent : python run.py @FICHIER). Les options "
                        "passées en ligne de commande priment sur le fichier")

    # ── Modèle ───────────────────────────────────────────────────────────────
    p.add_argument("--model", type=str, default="lenet",
                   choices=list(MODEL_REGISTRY),
                   help="Architecture CNN à compresser")
    p.add_argument("--dataset", type=str, default="cifar10",
                   choices=list(DATASET_REGISTRY),
                   help="Jeu de données (même format 32×32 RGB que CIFAR-10, "
                        "seul le nombre de classes change)")

    # ── Paramètres principaux (listes) ───────────────────────────────────────
    p.add_argument("--eth", type=float, nargs="+", default=[0.90],
                   metavar="FLOAT",
                   help="Seuil(s) d'énergie PCA. Ex: --eth 0.25 0.50 0.75 0.90")
    p.add_argument("--p", type=float, nargs="+", default=[0.50],
                   metavar="FLOAT",
                   help="Proportion(s) de filtres pseudo-randomisés. Ex: --p 0.0 0.25 0.50")
    p.add_argument("--retrain-after-pca", action="store_true",
                   help="Ré-entraîner après PCA (p=0), puis pour chaque p du "
                        "sweep remplacer une fraction p des filtres déjà "
                        "entraînés par du pseudo-aléatoire SANS ré-entraîner "
                        "à nouveau (feuille Avec_reentrainement_PCA)")
    p.add_argument("--retrain-after-pseudo", action="store_true",
                   help="Ré-entraîner après pseudo-randomization")

    # ── Paramètres entraînement ──────────────────────────────────────────────
    p.add_argument("--n-filters", type=int, default=6,
                   help="Nombre de filtres conv1 (LeNet uniquement)")
    p.add_argument("--epochs-initial", type=int, default=4,
                   help="Époques d'entraînement du modèle initial")
    p.add_argument("--epochs-retrain-pca", type=int, default=2,
                   help="Époques de ré-entraînement après PCA "
                        "(utilisé avec --retrain-after-pca)")
    p.add_argument("--epochs-retrain-pseudo", type=int, default=2,
                   help="Époques de ré-entraînement après pseudo-randomisation "
                        "(utilisé avec --retrain-after-pseudo)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (Adam)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--freeze-fc", action="store_true",
                   help="Geler les couches Linear lors du ré-entraînement NRC")

    # ── Sauvegarde ───────────────────────────────────────────────────────────
    p.add_argument("--save-models", action="store_true",
                   help="Sauvegarder chaque modèle compressé + manifest (dans --save-dir)")
    p.add_argument("--save-dir", type=str, default="checkpoints",
                   help="Répertoire de sauvegarde des modèles compressés")

    # ── Divers ───────────────────────────────────────────────────────────────
    p.add_argument("--baseline-only", action="store_true",
                   help="Entraîner et évaluer uniquement le modèle initial")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42, help="Graine LFSR")
    p.add_argument("--K", type=int, default=100, help="Candidats par vecteur LFSR")
    p.add_argument("--output", type=str, default="",
                   help="Nom du fichier Excel (défaut : résultats_MODELE_DATASET_YYYYMMDD_HHMMSS.xlsx)")
    p.add_argument("--output-dir", type=str, default="results",
                   help="Dossier de sortie des fichiers Excel")
    p.add_argument("--log-dir", type=str, default="logs")
    p.add_argument("--cache-dir", type=str, default="cache",
                   help="Dossier de cache LFSR pour reprise (ex: --cache-dir cache)")
    p.add_argument("--no-cache", action="store_true",
                   help="Désactiver le cache (recalcule tout depuis zéro)")

    argv = list(sys.argv[1:] if argv is None else argv)

    # Pré-lecture de --config : ses tokens sont placés AVANT ceux de la ligne
    # de commande, qui gardent donc le dernier mot en cas de doublon.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--config", type=str, default=None)
    cfg, rest = pre.parse_known_args(argv)
    if cfg.config:
        argv = _read_args_file(cfg.config, _option_specs(p)) + rest

    return p.parse_args(argv)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _finetune(model, train_loader, test_loader, criterion, device,
              epochs, lr, label):
    logger = logging.getLogger(__name__)
    opt = optim.Adam(
        [param for param in model.parameters() if param.requires_grad], lr=lr)
    va = 0.0
    for ep in range(1, epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, criterion, device)
        vl, va = evaluate(model, test_loader, criterion, device)
        logger.info("[%s] Ép %02d/%d | train=%.4f/%.2f%% | val=%.4f/%.2f%%",
                    label, ep, epochs, tl, ta * 100, vl, va * 100)
    return va


def _pretrain_for(args, epochs_retrain, train_loader, test_loader, criterion,
                   device, label):
    """Entraîne un modèle pré-NRC pour (epochs_initial - epochs_retrain) époques.

    Garantit que (pré-NRC) + (ré-entraînement) = epochs_initial, pour une
    comparaison équitable avec la baseline — indépendamment pour chaque
    branche (PCA / pseudo), puisque leurs époques de retrain peuvent différer.
    """
    logger = logging.getLogger(__name__)
    epochs_pretrain = max(1, args.epochs_initial - epochs_retrain)
    logger.info("═" * 65)
    logger.info("ÉTAPE 1b — Modèle pré-NRC [%s] (%d ép. = %d - %d)",
                label, epochs_pretrain, args.epochs_initial, epochs_retrain)
    logger.info("═" * 65)
    net_pretrain = MODEL_REGISTRY[args.model](args).to(device)
    opt_pre = optim.Adam(net_pretrain.parameters(), lr=args.lr)
    va_pre = 0.0
    for ep in range(1, epochs_pretrain + 1):
        tl, ta = train_one_epoch(net_pretrain, train_loader, opt_pre, criterion, device)
        vl, va_pre = evaluate(net_pretrain, test_loader, criterion, device)
        logger.info("Ép %02d/%d | train=%.4f/%.2f%% | val=%.4f/%.2f%%",
                    ep, epochs_pretrain, tl, ta * 100, vl, va_pre * 100)
    net_pretrain = net_pretrain.cpu()
    logger.info("Pré-NRC [%s] : %.2f%% (%d ép.)", label, va_pre * 100, epochs_pretrain)
    return net_pretrain


def _s_layers(manifest: dict) -> str:
    """Résumé compact des dimensions PCA par couche. Ex: 'conv1:3, conv2:8'"""
    return ", ".join(f"{k}:{v['s']}" for k, v in manifest["layers"].items())


def _count_seeds(manifest: dict) -> tuple[int, int]:
    n = sum(len(v["selected_seeds"]) for v in manifest["layers"].values())
    return n, n * 4


# ─── Couleurs Excel ───────────────────────────────────────────────────────────

_GREEN  = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
_YELLOW = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
_RED    = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
_BLUE   = PatternFill(start_color="DDEEFF", end_color="DDEEFF", fill_type="solid")


def _apply_colors(ws, ref_acc, ref_params, ref_mem, ref_mac):
    headers = {cell.value: cell.column for cell in ws[1]}

    for row in ws.iter_rows(min_row=2):
        if row[0].value == "Référence":
            continue

        for col_name in ("Accuracy (%)", "Accuracy avant FT (%)", "Accuracy après FT (%)"):
            col_idx = headers.get(col_name)
            if col_idx is None:
                continue
            val = row[col_idx - 1].value
            if not isinstance(val, (int, float)):
                continue
            if val >= ref_acc:
                row[col_idx - 1].fill = _GREEN
            elif val >= ref_acc - 5:
                row[col_idx - 1].fill = _YELLOW
            else:
                row[col_idx - 1].fill = _RED

        for col_name, ref_val in (("Conv params", ref_params),
                                  ("Mémoire conv (KB)", ref_mem),
                                  ("MAC", ref_mac)):
            col_idx = headers.get(col_name)
            if col_idx is None:
                continue
            val = row[col_idx - 1].value
            if isinstance(val, (int, float)):
                row[col_idx - 1].fill = _GREEN if val < ref_val else _RED

        col_idx = headers.get("Gain compression")
        if col_idx:
            val = row[col_idx - 1].value
            if isinstance(val, (int, float)):
                row[col_idx - 1].fill = _GREEN if val > 1 else _RED

        for col_name in ("Seeds (nb)", "Seeds (octets)"):
            col_idx = headers.get(col_name)
            if col_idx is None:
                continue
            val = row[col_idx - 1].value
            if isinstance(val, int) and val >= 0:
                row[col_idx - 1].fill = _BLUE


# ─── Main ─────────────────────────────────────────────────────────────────────

def _cache_dir(args, eth: float, p: float, suffix: str = "") -> str | None:
    """Retourne le dossier de cache pour une combinaison (eth, p)."""
    if args.no_cache:
        return None
    tag = f"{args.model}_{args.dataset}_eth{eth:.0%}_p{p:.2f}{suffix}"
    return os.path.join(args.cache_dir, tag)


def main():
    args = parse_args()
    args.num_classes = DATASET_REGISTRY[args.dataset]["num_classes"]
    setup_logging(args.log_dir)
    logger = logging.getLogger(__name__)

    logger.info("═" * 65)
    logger.info("Modèle : %s | Dataset : %s (%d classes)",
                args.model, args.dataset, args.num_classes)
    if args.baseline_only:
        logger.info("Mode BASELINE ONLY")
    else:
        n_combos = len(args.eth) * len(args.p)
        logger.info("Pipeline NRC — sweep %d eth × %d p = %d combinaisons",
                    len(args.eth), len(args.p), n_combos)
        logger.info("  eth  = %s", args.eth)
        logger.info("  p    = %s", args.p)
        logger.info("  retrain_after_pca=%s (epochs=%d) | retrain_after_pseudo=%s (epochs=%d)",
                    args.retrain_after_pca, args.epochs_retrain_pca,
                    args.retrain_after_pseudo, args.epochs_retrain_pseudo)
    logger.info("  device=%s | epochs_initial=%d", args.device, args.epochs_initial)
    logger.info("═" * 65)

    device    = args.device
    criterion = nn.CrossEntropyLoss()

    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_cls = DATASET_REGISTRY[args.dataset]["cls"]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = dataset_cls(
        root="./data", train=True,  download=True, transform=transform)
    testset  = dataset_cls(
        root="./data", train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    test_loader  = torch.utils.data.DataLoader(
        testset,  batch_size=args.batch_size, shuffle=False, num_workers=2)

    # ── Étape 1 : entraînement du modèle initial ─────────────────────────────
    logger.info("═" * 65)
    logger.info("ÉTAPE 1 — %s (%d époques)", args.model, args.epochs_initial)
    logger.info("═" * 65)

    net = MODEL_REGISTRY[args.model](args).to(device)
    opt = optim.Adam(net.parameters(), lr=args.lr)
    for ep in range(1, args.epochs_initial + 1):
        tl, ta = train_one_epoch(net, train_loader, opt, criterion, device)
        vl, va = evaluate(net, test_loader, criterion, device)
        logger.info("Ép %02d/%d | train=%.4f/%.2f%% | val=%.4f/%.2f%%",
                    ep, args.epochs_initial, tl, ta * 100, vl, va * 100)
    baseline_acc = va
    net = net.cpu()

    ref_conv_params = count_conv_params(net)
    ref_conv_bytes  = ref_conv_params * 4
    ref_conv_macs   = count_conv_macs(net)
    logger.info("Baseline : %.2f%% | conv params réf = %d (%.1f KB) | MAC réf = %d",
                baseline_acc * 100, ref_conv_params, ref_conv_bytes / 1024, ref_conv_macs)

    # ── Modèles pré-NRC : un par branche, chacun avec son propre budget ──────
    # d'époques de retrain (epochs_initial - epochs_retrain_{pca,pseudo}),
    # pour que chaque branche totalise epochs_initial époques comme la
    # baseline, même si epochs_retrain_pca ≠ epochs_retrain_pseudo.
    net_pretrain_pca    = net
    net_pretrain_pseudo = net
    if not args.baseline_only:
        if args.retrain_after_pca:
            net_pretrain_pca = _pretrain_for(
                args, args.epochs_retrain_pca, train_loader, test_loader,
                criterion, device, "PCA")
        if args.retrain_after_pseudo:
            net_pretrain_pseudo = _pretrain_for(
                args, args.epochs_retrain_pseudo, train_loader, test_loader,
                criterion, device, "Pseudo")

    # ── Arrêt anticipé si --baseline-only ────────────────────────────────────
    if args.baseline_only:
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = args.output or os.path.join(
            args.output_dir, f"baseline_{args.model}_{args.dataset}_{timestamp}.xlsx")
        df = pd.DataFrame([{
            "Modèle":            args.model,
            "Dataset":           args.dataset,
            "Époques":           args.epochs_initial,
            "lr":                args.lr,
            "Accuracy (%)":      round(baseline_acc * 100, 2),
            "Conv params":       ref_conv_params,
            "Mémoire conv (KB)": round(ref_conv_bytes / 1024, 2),
            "MAC":               ref_conv_macs,
        }])
        df.to_excel(output_path, index=False, sheet_name="Baseline")
        logger.info("Export Excel → %s", output_path)
        return

    # ── Collecte des résultats ───────────────────────────────────────────────
    rows_no_ft     = []
    rows_ft_pca    = []
    rows_ft_pseudo = []

    ref_acc_pct = round(baseline_acc * 100, 2)
    ref_mem_kb  = round(ref_conv_bytes / 1024, 2)

    total = len(args.eth) * len(args.p)
    done  = 0

    for eth in args.eth:

        # ── Ré-entraînement après PCA seule (p=0) ────────────────────────────
        if args.retrain_after_pca:
            logger.info("[PCA-FT] eth=%.2f — compression p=0 + fine-tuning", eth)
            nrc_pca, manifest_pca = compress(
                deepcopy(net_pretrain_pca), eth=eth, p=0.0,
                K=args.K, seed=args.seed,
                freeze_linear=args.freeze_fc,
                cache_dir=_cache_dir(args, eth, 0.0, "_pca_ft"),
            )
            nrc_pca = nrc_pca.to(device)
            _, va_pca_before = evaluate(nrc_pca, test_loader, criterion, device)
            va_pca_ft = _finetune(nrc_pca, train_loader, test_loader, criterion,
                                  device, args.epochs_retrain_pca, args.lr,
                                  f"PCA-FT eth={eth:.2f}")

            logger.info("[PCA-FT] eth=%.2f | sans FT=%.2f%% | avec FT=%.2f%% | Δ=%+.2f%%",
                        eth, va_pca_before * 100, va_pca_ft * 100,
                        (va_pca_ft - baseline_acc) * 100)

            if args.save_models:
                save_compressed_model(
                    nrc_pca, manifest_pca, args.save_dir,
                    prefix=f"{args.model}_eth{eth:.0%}_p0_pca_ft",
                )

            # ── Pour chaque p du sweep : remplace après coup une fraction p
            # des filtres PCA déjà entraînés par du pseudo-aléatoire, SANS
            # ré-entraînement supplémentaire (conv_coord reste gelé à sa
            # valeur apprise ci-dessus). p=0.0 reproduit le PCA+FT pur.
            for p_val in args.p:
                seq_net, seq_manifest = apply_pseudo_after_pca(
                    nrc_pca, manifest_pca, p=p_val,
                    K=args.K, seed=args.seed,
                    cache_dir=_cache_dir(args, eth, p_val, "_pca_ft_then_pseudo"),
                )
                seq_net = seq_net.to(device)
                _, va_seq = evaluate(seq_net, test_loader, criterion, device)

                cp_seq  = count_conv_params(seq_net)
                mac_seq = count_conv_macs(seq_net)
                ns_seq, sb_seq = _count_seeds(seq_manifest)
                rows_ft_pca.append({
                    "Modèle":            args.model,
                    "Dataset":           args.dataset,
                    "eTh":               f"{eth:.0%}",
                    "p":                 p_val,
                    "Accuracy (%)":      round(va_seq * 100, 2),
                    "Conv params":       cp_seq,
                    "Mémoire conv (KB)": round(cp_seq * 4 / 1024, 2),
                    "Gain compression":  round(ref_conv_params / cp_seq, 4),
                    "MAC":               mac_seq,
                    "Seeds (nb)":        ns_seq,
                    "Seeds (octets)":    sb_seq,
                    "s_layers":          _s_layers(seq_manifest),
                    "Ré-entraînement":   "Oui (après PCA, p=0)",
                })
                logger.info(
                    "[PCA-FT→Pseudo] eth=%.2f p=%.2f | acc=%.2f%% (FT p=0=%.2f%%) | Δ=%+.2f%%",
                    eth, p_val, va_seq * 100, va_pca_ft * 100,
                    (va_seq - va_pca_ft) * 100)

                if args.save_models:
                    save_compressed_model(
                        seq_net, seq_manifest, args.save_dir,
                        prefix=f"{args.model}_eth{eth:.0%}_p{p_val:.2f}_pca_ft_then_pseudo",
                    )
                seq_net = seq_net.cpu()

            nrc_pca = nrc_pca.cpu()

        # ── Boucle sur les valeurs de p ──────────────────────────────────────
        for p_val in args.p:
            done += 1
            logger.info("─" * 65)
            logger.info("[%d/%d] eth=%.2f | p=%.2f", done, total, eth, p_val)
            logger.info("─" * 65)

            nrc_net, manifest = compress(
                deepcopy(net), eth=eth, p=p_val,
                K=args.K, seed=args.seed,
                freeze_linear=args.freeze_fc,
                cache_dir=_cache_dir(args, eth, p_val),
            )
            nrc_net = nrc_net.to(device)

            _, va_before = evaluate(nrc_net, test_loader, criterion, device)
            cp   = count_conv_params(nrc_net)
            mac  = count_conv_macs(nrc_net)
            gain = ref_conv_params / cp
            ns, sb = _count_seeds(manifest)

            rows_no_ft.append({
                "Modèle":              args.model,
                "Dataset":             args.dataset,
                "eTh":                 f"{eth:.0%}",
                "p":                   p_val,
                "Accuracy (%)":        round(va_before * 100, 2),
                "Conv params":         cp,
                "Mémoire conv (KB)":   round(cp * 4 / 1024, 2),
                "Gain compression": round(gain, 4),
                "MAC":                 mac,
                "Seeds (nb)":          ns,
                "Seeds (octets)":      sb,
                "s_layers":            _s_layers(manifest),
                "Ré-entraînement":     "Non",
            })

            if args.retrain_after_pseudo:
                nrc_net_ft, manifest_ft = compress(
                    deepcopy(net_pretrain_pseudo), eth=eth, p=p_val,
                    K=args.K, seed=args.seed,
                    freeze_linear=args.freeze_fc,
                    cache_dir=_cache_dir(args, eth, p_val, "_ft"),
                )
                nrc_net_ft = nrc_net_ft.to(device)
                _, va_ft_before = evaluate(nrc_net_ft, test_loader, criterion, device)
                va_ft = _finetune(nrc_net_ft, train_loader, test_loader, criterion,
                                  device, args.epochs_retrain_pseudo, args.lr,
                                  f"FT eth={eth:.2f} p={p_val:.2f}")
                cp_ft  = count_conv_params(nrc_net_ft)
                mac_ft = count_conv_macs(nrc_net_ft)
                ns_ft, sb_ft = _count_seeds(manifest_ft)
                rows_ft_pseudo.append({
                    "Modèle":               args.model,
                    "Dataset":              args.dataset,
                    "eTh":                  f"{eth:.0%}",
                    "p":                    p_val,
                    # "Accuracy avant FT (%)": round(va_ft_before * 100, 2),
                    "Accuracy après FT (%)": round(va_ft * 100, 2),
                    "Conv params":           cp_ft,
                    "Mémoire conv (KB)":     round(cp_ft * 4 / 1024, 2),
                    "Gain compression":      round(ref_conv_params / cp_ft, 4),
                    "MAC":                   mac_ft,
                    "Seeds (nb)":            ns_ft,
                    "Seeds (octets)":        sb_ft,
                    "s_layers":              _s_layers(manifest_ft),
                    "Ré-entraînement":       "Oui (après pseudo-rand)",
                })
                logger.info("eth=%.2f p=%.2f | sans FT=%.2f%% | avec FT=%.2f%% | Δ=%+.2f%%",
                            eth, p_val, va_ft_before * 100, va_ft * 100,
                            (va_ft - baseline_acc) * 100)

                if args.save_models:
                    save_compressed_model(
                        nrc_net_ft, manifest_ft, args.save_dir,
                        prefix=f"{args.model}_eth{eth:.0%}_p{p_val:.2f}_ft",
                    )
                nrc_net_ft = nrc_net_ft.cpu()
            else:
                logger.info("eth=%.2f p=%.2f | acc=%.2f%% | gain=%+.1f%% | conv=%d params",
                            eth, p_val, va_before * 100, gain, cp)

                if args.save_models:
                    save_compressed_model(
                        nrc_net, manifest, args.save_dir,
                        prefix=f"{args.model}_eth{eth:.0%}_p{p_val:.2f}",
                    )

            nrc_net = nrc_net.cpu()

    # ── Lignes référence ─────────────────────────────────────────────────────
    ref_no_ft = {
        "Modèle": args.model, "Dataset": args.dataset, "eTh": "Référence", "p": "—",
        "Accuracy (%)":        ref_acc_pct,
        "Conv params":         ref_conv_params,
        "Mémoire conv (KB)":   ref_mem_kb,
        "Gain compression": None,
        "MAC":                 ref_conv_macs,
        "Seeds (nb)": "—", "Seeds (octets)": "—",
        "s_layers": "—", "Ré-entraînement": "—",
    }
    ref_ft = {
        "Modèle": args.model, "Dataset": args.dataset, "eTh": "Référence", "p": "—",
        # "Accuracy avant FT (%)": ref_acc_pct,
        "Accuracy après FT (%)": ref_acc_pct,
        "Conv params":           ref_conv_params,
        "Mémoire conv (KB)":     ref_mem_kb,
        "Gain compression":      None,
        "MAC":                   ref_conv_macs,
        "Seeds (nb)": "—", "Seeds (octets)": "—",
        "s_layers": "—", "Ré-entraînement": "—",
    }

    # ── Export Excel ─────────────────────────────────────────────────────────
    timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        args.output_dir, f"résultats_{args.model}_{args.dataset}_{timestamp}.xlsx")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_no_ft = pd.DataFrame(rows_no_ft + [ref_no_ft])
        df_no_ft.to_excel(writer, index=False, sheet_name="Sans_reentrainement")
        _apply_colors(writer.sheets["Sans_reentrainement"],
                      ref_acc_pct, ref_conv_params, ref_mem_kb, ref_conv_macs)

        if args.retrain_after_pca and rows_ft_pca:
            df_ft_pca = pd.DataFrame(rows_ft_pca + [ref_no_ft])
            df_ft_pca.to_excel(writer, index=False, sheet_name="Avec_reentrainement_PCA")
            _apply_colors(writer.sheets["Avec_reentrainement_PCA"],
                          ref_acc_pct, ref_conv_params, ref_mem_kb, ref_conv_macs)

        if args.retrain_after_pseudo and rows_ft_pseudo:
            df_ft_pseudo = pd.DataFrame(rows_ft_pseudo + [ref_ft])
            df_ft_pseudo.to_excel(writer, index=False, sheet_name="Avec_reentrainement_Pseudo")
            _apply_colors(writer.sheets["Avec_reentrainement_Pseudo"],
                          ref_acc_pct, ref_conv_params, ref_mem_kb, ref_conv_macs)

    logger.info("═" * 65)
    logger.info("Export Excel → %s", output_path)
    logger.info("═" * 65)


if __name__ == "__main__":
    main()
