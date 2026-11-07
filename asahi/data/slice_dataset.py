"""Training slices: the same genomic patch stacked across ``slice_height`` libraries.

Every row of a training slice is supervised, so no row is special. Each epoch,
every patch's libraries are partitioned into disjoint groups, covering each
(library, patch) pair once. Groups are built to span the window's read-depth range.
"""

import numpy as np
from torch.utils.data import Dataset, Sampler

from asahi.data.transforms import apply_transforms


class SliceDataset(Dataset):
    """Indexed by ``(pid, rows)``, as yielded by :class:`DepthStratifiedSliceSampler`."""

    def __init__(self, store, samples, transform):
        self.store = store
        self.samples = list(samples)
        self.transform = transform
        self.n_patches = store.num_patches
        self._cols = store.cols_for(self.samples)

    def __getitem__(self, index):
        pid, rows = index
        rows = np.asarray(rows, dtype=np.int64)
        cols = self._cols[rows]
        gt, cpg_mask = self.store.get_slice(cols, pid)
        item = {
            "samps": [self.samples[i] for i in rows],
            "patch_id": pid,
            "true": gt,
            "cpg_mask": cpg_mask,
            "pos": self.store.padded_patch_loci(pid),
            "cov": self.store.get_cov_slice(cols, pid),
        }
        return apply_transforms(item, self.transform)


class DepthStratifiedSliceSampler(Sampler):
    """Yield ``(pid, rows)`` groups that each draw from every depth stratum of the window.

    Per patch: rank libraries by their depth in that window, cut the ranking into
    ``n_depth_strata`` equal strata, shuffle within each, then give every group
    ``slice_height // n_depth_strata`` libraries from each stratum. At 8 and 4, each
    slice holds 2 libraries from each depth quartile of its window.

    ``remainder='drop'`` discards each stratum's leftovers, a fresh random set per
    patch and epoch (training). ``'wrap'`` pads each stratum from its own front so no
    library is left out, and ``fixed=True`` reuses one composition every epoch
    (validation), so the selection loss is not moved by composition noise.
    """

    def __init__(self, depth, slice_height, n_depth_strata, remainder="drop",
                 shuffle=True, seed=0, fixed=False):
        depth = np.asarray(depth, dtype=np.float32)
        self.n = depth.shape[1]
        self.p = depth.shape[0]
        if slice_height % n_depth_strata:
            raise ValueError("slice_height must be divisible by n_depth_strata")
        if self.n < slice_height:
            raise ValueError(f"need at least {slice_height} libraries, got {self.n}")
        self.depth = depth
        self.h = int(slice_height)
        self.s = int(n_depth_strata)
        self.per = self.h // self.s
        self.shuffle, self.seed, self.fixed = bool(shuffle), int(seed), bool(fixed)
        self._epoch = 0
        sizes = [len(x) for x in np.array_split(np.arange(self.n), self.s)]
        self.g = min(sizes) // self.per if remainder == "drop" else -(-max(sizes) // self.per)
        if self.g < 1:
            raise ValueError(f"strata of {sizes} cannot supply {self.per} per group")

    def __len__(self):
        return self.g * self.p

    def _groups_for_patch(self, pid, rng):
        order = np.argsort(self.depth[pid], kind="stable")     # shallowest -> deepest
        cols = []
        for st in np.array_split(order, self.s):
            st = st.copy()
            rng.shuffle(st)
            need = self.g * self.per
            st = st[:need] if len(st) >= need else np.concatenate([st, st[: need - len(st)]])
            cols.append(st.reshape(self.g, self.per))
        # Shuffle within each group so row position does not encode the stratum.
        return rng.permuted(np.concatenate(cols, axis=1), axis=1)

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed if self.fixed else np.random.SeedSequence([self.seed, self._epoch]))
        self._epoch += 1
        out = np.empty((len(self), self.h + 1), dtype=np.int64)
        row = 0
        for pid in range(self.p):
            out[row:row + self.g, 0] = pid
            out[row:row + self.g, 1:] = self._groups_for_patch(pid, rng)
            row += self.g
        if self.shuffle:
            rng.shuffle(out, axis=0)
        for rec in out:
            yield int(rec[0]), tuple(rec[1:].tolist())
