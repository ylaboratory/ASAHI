"""Predict the full-depth methylome of low-depth libraries with a trained ASAHI model.

    python scripts/inference.py --checkpoint checkpoints/asahi_chr21.pth --chrom chr21 \\
        --anchor-dataset encode_pt_frac0.1_seed11 --support-dataset encode_pt \\
        --out results/encode_pt_frac0.1_seed11_chr21.csv

Anchors are read from ``--anchor-dataset``; support rows come from the full-depth
``--support-dataset``. A library is never its own support, and with ``--donor-table``
neither is another library from the same donor.
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from asahi.data.genome import assert_cpg_convention, load_chrom_sequence
from asahi.data_engine import build_inference_loader, build_store
from asahi.inference_engine import (MethylomeAccumulator, predict, resolve_support_panel,
                                    write_predictions)
from asahi.model import load_checkpoint

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s",
                    datefmt="%d-%b-%y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger("asahi.infer")

DEFAULT_GENOME = ROOT / "data" / "metadata" / "hg38_encoded_chr21.npz"
DEFAULT_DONORS = ROOT / "data" / "metadata" / "encode_libraries.tsv"


def run_inference(checkpoint, chrom, anchor_dataset, support_dataset, out,
                  data_root=ROOT / "data", genome=DEFAULT_GENOME, anchors=None,
                  support_samples=None, support_panel_size=64, slice_height=8, repeats=5,
                  num_cpgs=1024, batch_size=16, num_workers=4, device="cuda:0", seed=11,
                  donor_table=DEFAULT_DONORS, cutoff_patches=None):
    """Write predictions to ``out`` and return its path."""
    device = torch.device(device)
    anchor_store = build_store(data_root, anchor_dataset, chrom, num_cpgs)
    support_store = build_store(data_root, support_dataset, chrom, num_cpgs)
    anchors = list(anchors) if anchors is not None else list(anchor_store.samples)
    panel = resolve_support_panel(support_samples if support_samples is not None
                                  else support_store.samples, support_panel_size, seed=seed)
    donor_of = None
    if donor_table is not None:
        t = pd.read_csv(donor_table, sep="\t")
        donor_of = dict(zip(t["accession"], t["donor"]))

    seq = load_chrom_sequence(genome, chrom)
    assert_cpg_convention(seq, anchor_store.loci)
    model = load_checkpoint(checkpoint, map_location=device).set_genome(seq).to(device)

    ds, loader = build_inference_loader(
        anchor_store, anchors, support_store, panel, slice_height, repeats, batch_size,
        num_workers, seed=seed, donor_of=donor_of, cutoff_patches=cutoff_patches)
    logger.info("%d anchors x %d patches x %d repeats = %d slices; support panel of %d",
                len(anchors), ds.n_patches, repeats, len(ds), len(panel))

    n_cpgs = min(anchor_store.n_cpgs, ds.n_patches * num_cpgs)
    acc = MethylomeAccumulator(anchors, num_cpgs, ds.n_patches, repeats, n_cpgs)
    predict(model, loader, device, acc, log_every=max(1, len(loader) // 10))
    return write_predictions(acc.finalize(), anchor_store.loci[:n_cpgs], out, columns=anchors)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--anchor-dataset", required=True, help="Low-depth dataset to predict.")
    p.add_argument("--support-dataset", required=True, help="Full-depth dataset for support rows.")
    p.add_argument("--out", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--genome", default=str(DEFAULT_GENOME),
                   help="Encoded genome .npz holding this chromosome (scripts/encode_genome.py).")
    p.add_argument("--anchors", nargs="+", default=None, help="Default: every library in the dataset.")
    p.add_argument("--support-samples-file", default=None,
                   help="One library per line: the candidates for the support panel. "
                        "Default: every library in the support dataset.")
    p.add_argument("--support-panel-size", type=int, default=64)
    p.add_argument("--repeats", type=int, default=5, help="Support draws per (anchor, patch), averaged.")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--donor-table", default=str(DEFAULT_DONORS),
                   help="TSV with accession and donor columns; pass '' to disable.")
    p.add_argument("--cutoff-patches", type=int, default=None, help="Only the first N patches (quick test).")
    a = p.parse_args()
    support = None
    if a.support_samples_file:
        support = Path(a.support_samples_file).read_text().split()
    run_inference(a.checkpoint, a.chrom, a.anchor_dataset, a.support_dataset, a.out,
                  data_root=a.data_root, genome=a.genome, anchors=a.anchors,
                  support_samples=support, support_panel_size=a.support_panel_size,
                  repeats=a.repeats, batch_size=a.batch_size, num_workers=a.num_workers,
                  device=a.device, donor_table=a.donor_table or None,
                  cutoff_patches=a.cutoff_patches)


if __name__ == "__main__":
    main()
