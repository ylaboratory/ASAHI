"""Inference slices.

Row 0 is the library being predicted (the anchor), read at its real low depth; only
its predictions are kept. The other rows are support libraries, drawn uniformly with
replacement from a fixed panel and read from a full-depth matrix. Nothing is masked
or thinned, so the model sees everything observed in the anchor.
"""

import itertools

import numpy as np
from torch.utils.data import Dataset

from asahi.data.transforms import apply_transforms


class InferenceSliceDataset(Dataset):
    """Item order is anchor -> patch -> repeat, which ``MethylomeAccumulator`` relies on.

    Each item seeds its own RNG from ``(seed, anchor index, patch, repeat)``, so output
    does not depend on the number of workers.

    ``donor_of`` optionally maps library name to donor; a library sharing a donor with
    the anchor is then never drawn. A library is never its own support.
    """

    def __init__(self, anchor_store, anchor_samples, support_store, support_samples,
                 slice_height=8, num_repeats=5, cutoff_patches=None, transform=None,
                 seed=11, donor_of=None):
        if not np.array_equal(anchor_store.loci, support_store.loci):
            raise ValueError("anchor and support matrices are not on the same CpG rows")
        self.anchor_store, self.support_store = anchor_store, support_store
        self.samples = list(anchor_samples)
        self.h, self.r = int(slice_height), int(num_repeats)
        self.num_cpgs = anchor_store.num_cpgs
        self.transform = transform
        self.seed = int(seed)

        donor_of = donor_of or {}
        self._cands = {}
        for a in self.samples:
            donor = donor_of.get(a)
            cands = [s for s in support_samples
                     if s != a and (donor is None or donor_of.get(s) != donor)]
            if not cands:
                raise ValueError(f"no support libraries left for {a}")
            self._cands[a] = np.asarray(cands, dtype=object)

        self.n_patches = anchor_store.num_patches
        if cutoff_patches:
            self.n_patches = min(int(cutoff_patches), self.n_patches)
        self.pool = [pair for pair in itertools.product(self.samples, range(self.n_patches))
                     for _ in range(self.r)]

    def __len__(self):
        return len(self.pool)

    def __getitem__(self, idx):
        anchor, pid = self.pool[idx]
        j = idx // self.r
        rng = np.random.default_rng(np.random.SeedSequence(
            [self.seed, j // self.n_patches, pid, idx % self.r]))
        support = list(rng.choice(self._cands[anchor], size=self.h - 1, replace=True))

        a_cols = self.anchor_store.cols_for([anchor])
        s_cols = self.support_store.cols_for(support)
        a_gt, a_mask = self.anchor_store.get_slice(a_cols, pid)
        s_gt, s_mask = self.support_store.get_slice(s_cols, pid)
        item = {
            "samps": [anchor] + support,
            "patch_id": pid,
            "sparse": np.concatenate([a_gt, s_gt], axis=0),
            "cpg_mask": np.concatenate([a_mask, s_mask], axis=0),
            "pos": self.anchor_store.padded_patch_loci(pid),
            "cov_sparse": np.concatenate([self.anchor_store.get_cov_slice(a_cols, pid),
                                          self.support_store.get_cov_slice(s_cols, pid)], axis=0),
        }
        return apply_transforms(item, self.transform)
