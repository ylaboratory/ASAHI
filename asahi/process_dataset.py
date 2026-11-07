"""Build the per-chromosome matrices the model reads from per-sample Bismark files.

Input is a directory with one subdirectory per sample, each holding a
``*.cpgMerged.CpG_report.merged_CpG_evidence.cov[.gz]``. Output is two tab-separated
files with one row per canonical hg38 CpG and one column per sample: ``<chrom>.fm``
(beta in [0, 1]) and ``<chrom>.cov`` (read depth), both empty where unobserved.
"""

import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from asahi.read_utils import get_processed_df

COV_GLOB = "*.cpgMerged.CpG_report.merged_CpG_evidence.cov*"


def load_cpg_index(index_path, chrom):
    """The canonical CpGs of a chromosome as an empty frame indexed by 0-based ``start``."""
    df = pd.read_csv(index_path)
    df = df[df["seqname"] == chrom].set_index("start")
    if df.empty:
        raise ValueError(f"{chrom} is not in {index_path}")
    return df


def get_model_ready_data(samp_file, chrom, cpg_df):
    """Put one sample on the canonical CpGs: ``(name, betas, depths, % of CpGs observed)``.

    The sample's name is the file name up to its first underscore.
    """
    samp_name = Path(samp_file).name.split("_")[0]
    samp_df = get_processed_df(samp_file, chrom)
    std_df = samp_df.merge(cpg_df, how="right", on="start")
    return (samp_name, std_df["beta"].to_numpy(), std_df["read_depth"].to_numpy(),
            float(samp_df.shape[0] / cpg_df.shape[0] * 100))


def make_cc_gt(data_dir, chrom, index_path, fm_save_path, cov_save_path, num_workers=8):
    """Build and save ``<chrom>.fm`` and ``<chrom>.cov``. Returns the two DataFrames."""
    data_dir = Path(data_dir)
    cpg_df = load_cpg_index(index_path, chrom)
    files = []
    for samp_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        files.extend(str(p) for p in samp_dir.glob(COV_GLOB))
    if not files:
        raise ValueError(f"no merged CpG files found under {data_dir}")
    print(f"{chrom}: {len(files)} samples, {cpg_df.shape[0]} canonical CpGs")

    with multiprocessing.Pool(processes=min(num_workers, len(files))) as pool:
        with tqdm(total=len(files)) as pbar:
            pending = [pool.apply_async(get_model_ready_data, args=(f, chrom, cpg_df),
                                        callback=lambda *_: pbar.update()) for f in files]
            results = [r.get() for r in pending]

    fm = pd.DataFrame({r[0]: r[1] for r in results}).set_index(cpg_df.index)
    cov = pd.DataFrame({r[0]: r[2] for r in results}).set_index(cpg_df.index)
    obs = [r[3] for r in results]
    print(f"CpGs observed: mean {np.mean(obs):.3f}%, s.d. {np.std(obs):.3f}%")

    for df, path in ((fm, Path(fm_save_path)), (cov, Path(cov_save_path))):
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, sep="\t", header=True, index=True)
        print(f"saved {path}")
    return fm, cov
