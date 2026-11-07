"""Encode chromosomes of a genome FASTA for the model's DNA branch.

    python scripts/encode_genome.py hg38.fa data/metadata/hg38_encoded.npz --chroms chr1 chr11 chr21

Writes one integer array per chromosome. The base codes are A=0 T=1 C=2 G=3 N=4, then IUPAC.
"""

import argparse

import numpy as np

CODES = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4, 'M': 5, 'R': 6, 'W': 7, 'S': 8, 'Y': 9,
         'K': 10, 'V': 11, 'H': 12, 'D': 13, 'B': 14, 'X': 15}
LUT = np.full(256, -1, dtype=np.int8)
for base, code in CODES.items():
    LUT[ord(base)] = code


def encode(lines):
    arr = LUT[np.frombuffer("".join(lines).upper().encode("ascii"), dtype=np.uint8)]
    if (arr < 0).any():
        raise ValueError("unexpected base in FASTA")
    return arr


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fasta")
    p.add_argument("out")
    p.add_argument("--chroms", nargs="+", required=True, help="Names as in the FASTA headers.")
    p.add_argument("--prepend-chr", action="store_true",
                   help="Headers read '21', not 'chr21'; store the arrays as 'chr21'.")
    a = p.parse_args()
    want, out, name, lines = set(a.chroms), {}, None, []
    with open(a.fasta) as f:
        for line in f:
            if line.startswith(">"):
                if name in want:
                    out[name] = encode(lines)
                name, lines = line[1:].split()[0], []
            elif name in want:
                lines.append(line.strip())
    if name in want:
        out[name] = encode(lines)
    missing = want - set(out)
    if missing:
        raise KeyError(f"not in {a.fasta}: {sorted(missing)}")
    if a.prepend_chr:
        out = {f"chr{k}": v for k, v in out.items()}
    np.savez_compressed(a.out, **out)
    print(f"wrote {a.out}: " + ", ".join(f"{k} ({v.size} bases)" for k, v in out.items()))


if __name__ == "__main__":
    main()
