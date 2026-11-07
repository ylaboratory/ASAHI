"""Turn a config into matrix paths, stores and data loaders.

Matrices live at ``<data_root>/<dataset>/chrom_centric/true/``
``{FractionalMethylation/<chrom>.fm, ReadDepth/<chrom>.cov}``, optionally gzipped.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from asahi.data.chrom_store import ChromMethylomeStore, patch_depth_matrix
from asahi.data.inference_dataset import InferenceSliceDataset
from asahi.data.slice_dataset import DepthStratifiedSliceSampler, SliceDataset
from asahi.data.transforms import AddMissingness, ReadThinning, ReplaceNaN, ToTensor


def _existing(path):
    """``path`` if present, else ``path.gz``."""
    gz = Path(f"{path}.gz")
    if Path(path).exists():
        return Path(path)
    if gz.exists():
        return gz
    raise FileNotFoundError(f"neither {path} nor {gz} exists")


def matrix_paths(data_root, dataset, chrom):
    """``(fm_path, cov_path)`` of one dataset and chromosome."""
    root = Path(data_root) / dataset / "chrom_centric" / "true"
    return (_existing(root / "FractionalMethylation" / f"{chrom}.fm"),
            _existing(root / "ReadDepth" / f"{chrom}.cov"))


def build_store(data_root, dataset, chrom, num_cpgs):
    fm_path, cov_path = matrix_paths(data_root, dataset, chrom)
    return ChromMethylomeStore(fm_path, cov_path, num_cpgs, chrom=chrom)


def _worker_init_fn(worker_id):
    """Give each worker its own numpy stream; torch does not reseed numpy."""
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))


def _loader(dataset, batch_size, num_workers, sampler=None, pin_memory=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                      num_workers=num_workers, pin_memory=pin_memory,
                      persistent_workers=num_workers > 0, drop_last=False,
                      worker_init_fn=_worker_init_fn if num_workers > 0 else None)


def build_train_loaders(store, train_samples, val_samples, data_cfg, train_cfg):
    """``(train_loader, val_loader)``. Validation is thinned too, so it has a denoising loss."""
    tfs = [AddMissingness(data_cfg["missing_rate"]),
           ReadThinning(data_cfg["thin_ratio"], data_cfg["missing_rate"],
                        p_low=data_cfg["thin_p_low"], p_high=data_cfg["thin_p_high"],
                        min_cov=data_cfg["thin_min_cov"]),
           ReplaceNaN(data_cfg["replace_val"]),
           ToTensor()]
    h, s, seed = data_cfg["slice_height"], data_cfg["n_depth_strata"], train_cfg["seed"]
    train_sampler = DepthStratifiedSliceSampler(
        patch_depth_matrix(store, train_samples), h, s,
        remainder="drop", shuffle=True, seed=seed, fixed=False)
    val_sampler = DepthStratifiedSliceSampler(
        patch_depth_matrix(store, val_samples), h, s,
        remainder="wrap", shuffle=False, seed=seed, fixed=True)
    kw = dict(batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    return (_loader(SliceDataset(store, train_samples, tfs), sampler=train_sampler, **kw),
            _loader(SliceDataset(store, val_samples, tfs), sampler=val_sampler, **kw))


def build_inference_loader(anchor_store, anchor_samples, support_store, support_samples,
                           slice_height, num_repeats, batch_size, num_workers,
                           replace_val=0.5, seed=11, donor_of=None, cutoff_patches=None):
    """A loader in the fixed item order that the prediction accumulator expects."""
    ds = InferenceSliceDataset(
        anchor_store, anchor_samples, support_store, support_samples,
        slice_height=slice_height, num_repeats=num_repeats, cutoff_patches=cutoff_patches,
        transform=[ReplaceNaN(replace_val), ToTensor()], seed=seed, donor_of=donor_of)
    return ds, _loader(ds, batch_size, num_workers)
