"""Serve fixed-size CpG patches from one chromosome's methylation matrices.

A matrix has one row per canonical CpG and one column per library. A patch is
simply ``num_cpgs`` consecutive rows, so patch ``pid`` of a library is
``matrix[pid * num_cpgs : (pid + 1) * num_cpgs, column]``.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _read_matrix(path):
    """A tab-separated matrix with a ``start`` index: ``(values, samples, loci)``."""
    df = pd.read_csv(path, sep="\t", index_col="start")
    values = np.ascontiguousarray(df.to_numpy(dtype=np.float32))
    # Coordinates stay integral: float32 rounds values above ~16.7M and would
    # corrupt the base-pair distances used by the relative positional encoding.
    return values, list(map(str, df.columns)), df.index.to_numpy().astype(np.int64)


class ChromMethylomeStore:
    """Betas (``fm``) and read depth (``cov``) of one chromosome, ``(n_cpgs, n_samples)``.

    Missing cells are NaN in both. Short final patches are padded to ``num_cpgs``:
    betas with 0 (flagged missing by ``cpg_mask``), depth with NaN.
    """

    def __init__(self, fm_path, cov_path, num_cpgs, chrom=None):
        self.num_cpgs = int(num_cpgs)
        self.chrom = chrom or Path(fm_path).name.split(".")[0]
        logger.info("reading %s", os.path.relpath(fm_path))
        self.fm, self.samples, self.loci = _read_matrix(fm_path)
        logger.info("reading %s", os.path.relpath(cov_path))
        cov, cov_samples, _ = _read_matrix(cov_path)
        if cov_samples != self.samples:
            cov = np.ascontiguousarray(cov[:, [cov_samples.index(s) for s in self.samples]])
        cov[np.isnan(self.fm)] = np.nan
        self.cov = cov
        self.sample_to_col = {s: i for i, s in enumerate(self.samples)}
        self.n_cpgs, self.n_samples = self.fm.shape
        self.num_patches = int(np.ceil(self.n_cpgs / self.num_cpgs))
        logger.info("%s: %d CpGs x %d libraries -> %d patches of %d CpGs", self.chrom,
                    self.n_cpgs, self.n_samples, self.num_patches, self.num_cpgs)

    def cols_for(self, sample_names):
        return np.fromiter((self.sample_to_col[s] for s in sample_names),
                           dtype=np.int64, count=len(sample_names))

    def patch_bounds(self, pid):
        start = pid * self.num_cpgs
        return start, min(start + self.num_cpgs, self.n_cpgs)

    def get_slice(self, col_idxs, pid):
        """``(betas, cpg_mask)``, each ``(len(col_idxs), num_cpgs)``; True = missing."""
        start, end = self.patch_bounds(pid)
        raw = self.fm[start:end][:, col_idxs].T.copy()
        cpg_mask = np.isnan(raw)
        if end - start < self.num_cpgs:
            gt = np.zeros((len(col_idxs), self.num_cpgs), dtype=np.float32)
            gt[:, :end - start] = raw
            mask = np.ones((len(col_idxs), self.num_cpgs), dtype=bool)
            mask[:, :end - start] = cpg_mask
            return gt, mask
        return raw, cpg_mask

    def get_cov_slice(self, col_idxs, pid):
        """Read depth for patch ``pid``, ``(len(col_idxs), num_cpgs)``."""
        start, end = self.patch_bounds(pid)
        raw = self.cov[start:end][:, col_idxs].T.copy()
        if end - start < self.num_cpgs:
            out = np.full((len(col_idxs), self.num_cpgs), np.nan, dtype=np.float32)
            out[:, :end - start] = raw
            return out
        return raw

    def padded_patch_loci(self, pid):
        """Coordinates of patch ``pid``; a short patch repeats its last locus. int64."""
        start, end = self.patch_bounds(pid)
        loci = self.loci[start:end]
        if len(loci) < self.num_cpgs:
            loci = np.pad(loci, (0, self.num_cpgs - len(loci)), mode="edge")
        return loci.astype(np.int64)


def patch_depth_matrix(store, samples):
    """Mean read depth per ``(patch, sample)``, unobserved cells counted as 0.

    Columns follow ``samples``. Used to rank libraries by depth within each window.
    """
    out = np.zeros((store.num_patches, store.n_samples), dtype=np.float32)
    for pid in range(store.num_patches):
        start, end = store.patch_bounds(pid)
        out[pid] = np.nan_to_num(store.cov[start:end], nan=0.0).mean(axis=0)
    return np.ascontiguousarray(out[:, store.cols_for(samples)])
