"""Collect slice predictions into per-library methylomes and write them to disk."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


class MethylomeAccumulator:
    """Average each anchor's row-0 predictions over repeats, into one array per library.

    The loader must keep the dataset's order (anchor -> patch -> repeat). Every item is
    checked against the position its index implies.
    """

    def __init__(self, samples, num_cpgs, num_patches, num_repeats, n_cpgs):
        self.samples = list(samples)
        self.P, self.r, self.L, self.n_cpgs = int(num_patches), int(num_repeats), int(num_cpgs), int(n_cpgs)
        self.buf = np.zeros((len(self.samples), self.P * self.L), dtype=np.float32)
        self.cursor = 0

    def add(self, batch, pred):
        row0 = pred[:, 0, :].detach().float().cpu().numpy()
        anchors = batch["samps"][0]
        pids = batch["patch_id"].numpy()
        for j in range(row0.shape[0]):
            k = (self.cursor + j) // self.r
            pid, s = k % self.P, k // self.P
            if anchors[j] != self.samples[s] or int(pids[j]) != pid:
                raise RuntimeError("the inference loader's item order was changed")
            lo = pid * self.L
            self.buf[s, lo:lo + self.L] += row0[j]
        self.cursor += row0.shape[0]

    def finalize(self):
        expected = len(self.samples) * self.P * self.r
        if self.cursor != expected:
            raise RuntimeError(f"saw {self.cursor} items, expected {expected}")
        return {s: self.buf[i, :self.n_cpgs] / float(self.r) for i, s in enumerate(self.samples)}


@torch.no_grad()
def predict(model, loader, device, collector, log_every=None):
    """Predict every position of every slice. The input's own gaps are its only mask."""
    model.eval()
    for i, batch in enumerate(loader):
        pred = model(batch["sparse"].to(device, non_blocking=True),
                     batch["cpg_mask"].to(device, non_blocking=True),
                     batch["pos"].to(device, non_blocking=True),
                     batch["cov_sparse"].to(device, non_blocking=True))
        collector.add(batch, pred)
        if log_every and i % log_every == 0:
            logger.info("batch %d/%d", i, len(loader))
    return collector


def resolve_support_panel(candidates, panel_size=None, seed=11):
    """A fixed, seeded random panel of support libraries; order follows ``candidates``."""
    candidates = list(candidates)
    if panel_size is None or panel_size >= len(candidates):
        return candidates
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(candidates), size=int(panel_size), replace=False))
    return [candidates[i] for i in idx]


def write_predictions(preds, loci, path, columns=None):
    """CSV: index ``start``, one column per library, values clipped to [0, 1]."""
    columns = list(columns) if columns is not None else list(preds)
    mat = np.clip(np.column_stack([preds[c] for c in columns]), 0.0, 1.0)
    df = pd.DataFrame(mat, columns=columns)
    df.index = pd.Index(np.asarray(loci, dtype=np.int64), name="start")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True, float_format="%.4f")
    logger.info("wrote %s (%d x %d)", path, mat.shape[0], mat.shape[1])
    return path
