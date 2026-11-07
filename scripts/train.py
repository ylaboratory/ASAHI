"""Train ASAHI from a YAML config: ``python scripts/train.py --config example_train``."""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from asahi.data.genome import assert_cpg_convention, load_chrom_sequence
from asahi.data_engine import build_store, build_train_loaders
from asahi.metrics import cpgwise_mean
from asahi.model import ASAHI
from asahi.model_engine import fit

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("asahi.train")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def load_config(name):
    """``name`` is a path to a YAML file or the stem of one in ``configs/``."""
    path = Path(name)
    if not (path.suffix == ".yaml" and path.exists()):
        path = ROOT / "configs" / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def resolve(path):
    """Config paths are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_training(config, device=None, max_epochs=None, ckpt_path=None):
    """Train one model and return its checkpoint path."""
    data, model_cfg, train = config["data"], config["model"], config["train"]
    device = device or train["device"]
    max_epochs = max_epochs if max_epochs is not None else train["max_epochs"]
    if ckpt_path is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ckpt_path = ROOT / "checkpoints" / f"{config['name']}_{data['chrom']}_{stamp}.pth"

    with open(resolve(data["split_file"])) as f:
        split = json.load(f)
    if set(split["train"]) & set(split["val"]):
        raise ValueError("a library is in both train and val")

    set_seed(train["seed"])
    store = build_store(resolve(data["data_root"]), data["dataset"], data["chrom"], data["num_cpgs"])

    set_seed(train["seed"])
    train_loader, val_loader = build_train_loaders(store, split["train"], split["val"], data, train)
    logger.info("train: %d libraries, %d batches per epoch | val: %d libraries, %d batches",
                len(split["train"]), len(train_loader), len(split["val"]), len(val_loader))

    model = ASAHI(**model_cfg, lr=train["lr"], lr_decay_factor=train["lr_decay_factor"],
                  warmup_steps=train["warmup_steps"])
    seq = load_chrom_sequence(resolve(data["genome_path"]), data["chrom"])
    assert_cpg_convention(seq, store.loci)
    model.set_genome(seq)
    model.set_cohort_mean(cpgwise_mean(store.fm, store.cols_for(split["train"])))
    logger.info("model: %d parameters", sum(p.numel() for p in model.parameters()))

    return fit(model, train_loader, val_loader, device=device, ckpt_path=ckpt_path,
               max_epochs=max_epochs, patience=train["patience"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="example_train")
    p.add_argument("--device", default=None, help="e.g. cuda:0 or cpu (default: from the config)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--ckpt", default=None, help="Where to save the model.")
    args = p.parse_args()
    run_training(load_config(args.config), device=args.device, max_epochs=args.epochs,
                 ckpt_path=args.ckpt)


if __name__ == "__main__":
    main()
