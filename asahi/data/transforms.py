"""Slice transforms. Masks follow the convention True = missing/selected.

Training order: ``AddMissingness -> ReadThinning -> ReplaceNaN -> ToTensor``.
Inference order: ``ReplaceNaN -> ToTensor``; nothing is masked or thinned there.
"""

import numpy as np
import torch


class AddMissingness:
    """Hide each cell independently with probability ``missing_rate``.

    Adds ``sparse`` (model input), ``eval_mask`` (observed and hidden: the
    imputation loss) and ``cov_sparse`` (read depth, hidden where the beta is).
    """

    def __init__(self, missing_rate):
        if not 0.0 < missing_rate < 1.0:
            raise ValueError("missing_rate must be in (0, 1)")
        self.missing_rate = missing_rate

    def __call__(self, item):
        true, cpg_mask = item["true"], item["cpg_mask"]
        sim_mask = np.random.random(size=true.shape) < self.missing_rate
        sparse = true.copy()
        sparse[sim_mask] = np.nan
        item["sparse"] = sparse
        item["eval_mask"] = (~cpg_mask) & sim_mask
        cov_sparse = item["cov"].copy()
        cov_sparse[sim_mask] = np.nan
        item["cov_sparse"] = cov_sparse
        return item


class ReadThinning:
    """Binomially thin the reads of some visible cells: the denoising self-supervision.

    A selected cell with depth ``d`` and ``k`` methylated reads is shown to the
    model as ``k'/d'``, where each read is kept with probability ``p``::

        p ~ Uniform(p_low, p_high),  k' ~ Bin(k, p),  u' ~ Bin(d - k, p),  d' = k' + u'

    and it is supervised against the full-depth ``k/d``. If no read survives the
    cell becomes missing, but is still supervised.

    ``thin_ratio`` is the nominal share of observed cells selected, disjoint from
    the hidden cells. Cells with ``d < min_cov`` are left clean, so the realised
    share is ``thin_ratio * P(d >= min_cov)``.

    Adds ``thin_mask`` (visible to the model, unlike ``eval_mask``) and
    ``thin_drop_mask`` (the subset thinned to zero reads). ``cov_sparse`` carries the
    depth the model is shown, ``d'``. The target's depth ``cov`` is removed, since it
    must never reach the model.
    """

    def __init__(self, thin_ratio, missing_rate, p_low=0.1, p_high=0.9, min_cov=5):
        if not 0.0 < thin_ratio < 1.0 or missing_rate + thin_ratio > 1.0 + 1e-9:
            raise ValueError("need 0 < thin_ratio < 1 and missing_rate + thin_ratio <= 1")
        if not 0.0 < p_low < p_high < 1.0:
            raise ValueError("need 0 < p_low < p_high < 1")
        self.p_low, self.p_high, self.min_cov = float(p_low), float(p_high), int(min_cov)
        # Selection probability among visible cells that gives thin_ratio overall.
        self.q = float(thin_ratio) / (1.0 - float(missing_rate))

    def __call__(self, item):
        true, cov, sparse = item["true"], item["cov"], item["sparse"]
        cand = (~item["cpg_mask"]) & (~item["eval_mask"]) & (
            np.random.random(size=true.shape) < self.q)
        m = cand & np.isfinite(cov) & (cov >= self.min_cov)

        thin_drop = np.zeros_like(m)
        if m.any():
            d = cov[m].astype(np.int64)
            kf = true[m].astype(np.float64) * d
            # Betas must be exact k/d ratios for the read counts to be recoverable.
            if float(np.abs(kf - np.rint(kf)).max()) >= 1e-2:
                raise AssertionError("beta * depth is not integral; betas must be exact k/d")
            k = np.clip(np.rint(kf).astype(np.int64), 0, d)

            p = np.random.uniform(self.p_low, self.p_high, size=d.shape)
            kp = np.random.binomial(k, p)
            up = np.random.binomial(d - k, p)
            dp = kp + up

            nz = dp > 0
            vals = np.full(d.shape, np.nan, dtype=sparse.dtype)
            vals[nz] = kp[nz] / dp[nz]
            sparse[m] = vals
            thin_drop[m] = ~nz
            item["cov_sparse"][m] = np.where(nz, dp, np.nan).astype(item["cov_sparse"].dtype)

        item["thin_mask"] = m
        item["thin_drop_mask"] = thin_drop
        item.pop("cov")
        return item


class ReplaceNaN:
    """Betas: NaN -> ``replace_val``. Depth: NaN -> 0. Masks are untouched."""

    def __init__(self, replace_val=0.5):
        self.replace_val = replace_val

    def __call__(self, item):
        for key in ("true", "sparse"):
            if key in item:
                item[key] = np.nan_to_num(item[key], nan=self.replace_val)
        if "cov_sparse" in item:
            item["cov_sparse"] = np.nan_to_num(item["cov_sparse"], nan=0)
        return item


class ToTensor:
    """numpy -> torch. Coordinates stay int64."""

    def __call__(self, item):
        for key in ("true", "sparse", "cov_sparse"):
            if key in item:
                item[key] = torch.from_numpy(np.ascontiguousarray(item[key])).to(torch.float32)
        item["pos"] = torch.from_numpy(np.ascontiguousarray(item["pos"])).to(torch.int64)
        return item


def apply_transforms(item, transforms):
    for t in transforms:
        item = t(item)
    return item
