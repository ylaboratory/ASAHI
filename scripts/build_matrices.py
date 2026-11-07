"""Build ``<chrom>.fm`` and ``<chrom>.cov`` from per-sample Bismark files.

    python scripts/build_matrices.py --cov-dir data/encode_subset --dataset encode_pt_rebuilt \\
        --chrom chr21

Writes to ``data/<dataset>/chrom_centric/true/{FractionalMethylation,ReadDepth}/``.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from asahi.process_dataset import make_cc_gt


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cov-dir", required=True, help="One subdirectory per sample.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--data-root", default=str(ROOT / "data"))
    p.add_argument("--cpg-index", default=str(ROOT / "data" / "metadata" / "hg38_cpg_py0idx.csv.gz"))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    root = Path(a.data_root) / a.dataset / "chrom_centric" / "true"
    make_cc_gt(a.cov_dir, a.chrom, a.cpg_index, root / "FractionalMethylation" / f"{a.chrom}.fm",
               root / "ReadDepth" / f"{a.chrom}.cov", num_workers=a.workers)


if __name__ == "__main__":
    main()
