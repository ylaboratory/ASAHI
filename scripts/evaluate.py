"""Score a prediction CSV: MSE against full depth, per library, at GT depth >= 10.

    python scripts/evaluate.py --preds results/encode_pt_frac0.1_seed11_chr21.csv \\
        --chrom chr21 --input-dataset encode_pt_frac0.1_seed11 --gt-dataset encode_pt
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from asahi.data_engine import build_store
from asahi.metrics import score_library, summarize


def evaluate(preds, chrom, input_dataset, gt_dataset, data_root=ROOT / "data", gt_min_cov=10):
    """``(per_library, summary)`` DataFrames."""
    pred_df = pd.read_csv(preds, index_col="start")
    gt = build_store(data_root, gt_dataset, chrom, 1024)
    inp = build_store(data_root, input_dataset, chrom, 1024)
    n = pred_df.shape[0]
    if not (np.array_equal(pred_df.index.to_numpy(), gt.loci[:n])
            and np.array_equal(gt.loci, inp.loci)):
        raise ValueError("predictions and matrices are not on the same CpG rows")
    rows = []
    for lib in pred_df.columns:
        g, i = gt.sample_to_col[lib], inp.sample_to_col[lib]
        for row in score_library(gt.fm[:n, g].astype(np.float64), gt.cov[:n, g],
                                 inp.fm[:n, i].astype(np.float64),
                                 pred_df[lib].to_numpy(dtype=np.float64), gt_min_cov):
            rows.append({"library": lib, **row})
    return pd.DataFrame(rows), summarize(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preds", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--input-dataset", required=True)
    p.add_argument("--gt-dataset", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--gt-min-cov", type=int, default=10)
    a = p.parse_args()
    _, summary = evaluate(a.preds, a.chrom, a.input_dataset, a.gt_dataset, a.data_root, a.gt_min_cov)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
