"""The encoded reference genome used by the DNA branch of the model.

One chromosome of hg38 as a flat ``uint8`` array, one byte per base, in the
encoding written by ``scripts/encode_genome.py``::

    A=0  T=1  C=2  G=3  N=4    (then IUPAC: M R W S Y K V H D B X = 5..15)

Note the A, T, C, G order. Loci are 0-based and point at the C of a CpG.
"""

from pathlib import Path

import numpy as np

MAX_CODE = 15


def load_chrom_sequence(genome_path, chrom):
    """``(n_bases,) uint8`` for one chromosome of an encoded-genome ``.npz``."""
    genome_path = Path(genome_path)
    with np.load(genome_path) as z:
        if chrom not in z:
            raise KeyError(f"{chrom} is not in {genome_path.name} (has {sorted(z.files)}). "
                           "Build it with scripts/encode_genome.py.")
        seq = z[chrom]
    seq = np.ascontiguousarray(seq).astype(np.uint8, copy=False)
    if seq.ndim != 1 or (seq.size and int(seq.max()) > MAX_CODE):
        raise ValueError(f"{chrom} in {genome_path.name} is not an encoded sequence")
    return seq


def assert_cpg_convention(seq, loci, n_check=20000, seed=0):
    """Fail unless ``loci`` are 0-based positions of a CpG's C on this genome."""
    loci = np.asarray(loci, dtype=np.int64)
    rng = np.random.default_rng(seed)
    take = loci if len(loci) <= n_check else rng.choice(loci, n_check, replace=False)
    take = take[(take >= 0) & (take + 1 < seq.shape[0])]
    frac = float(np.mean((seq[take] == 2) & (seq[take + 1] == 3)))
    if frac < 0.99:
        raise AssertionError(f"only {frac:.1%} of loci sit on a CpG 'C'; the matrices and the "
                             "genome disagree on chromosome, build or coordinate convention.")
    return frac
