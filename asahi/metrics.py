"""Score predicted methylomes against the full-depth ground truth."""

import warnings

import numpy as np
import pandas as pd


def cpgwise_mean(fm, cols):
    """Per-CpG mean beta over the given columns; NaN where none observed the CpG."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(fm[:, np.asarray(cols, dtype=np.int64)], axis=1)


def score_library(gt_beta, gt_cov, input_beta, pred, gt_min_cov=10):
    """MSE rows for one library, on cells with ground-truth depth >= ``gt_min_cov``.

    ``denoised`` cells were observed in the low-depth input; ``imputed`` cells were not.
    Within a subset, the model and Obs (the input left as is, denoised only) use the same cells.
    """
    scored = np.isfinite(gt_beta) & np.isfinite(gt_cov) & (gt_cov >= gt_min_cov) & np.isfinite(pred)
    seen = np.isfinite(input_beta)
    pred = np.clip(pred, 0.0, 1.0)
    rows = []
    for subset, m in (("denoised", scored & seen), ("imputed", scored & ~seen)):
        if not m.any():
            continue
        rows.append({"subset": subset, "method": "ASAHI", "n": int(m.sum()),
                     "mse": float(np.mean((pred[m] - gt_beta[m]) ** 2.0))})
        if subset == "denoised":
            rows.append({"subset": subset, "method": "Obs", "n": int(m.sum()),
                         "mse": float(np.mean((input_beta[m] - gt_beta[m]) ** 2.0))})
    return rows


def summarize(rows):
    """Mean MSE over libraries, per subset and method."""
    df = pd.DataFrame(rows)
    return df.groupby(["subset", "method"]).agg(
        libraries=("mse", "size"), mean_cells=("n", "mean"), mean_mse=("mse", "mean")).reset_index()
