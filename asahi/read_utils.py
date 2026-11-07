"""Read per-sample Bismark output.

Input is the ``*.cpgMerged.CpG_report.merged_CpG_evidence.cov`` file made by
``bismark_methylation_extractor --bedGraph`` followed by ``coverage2cytosine --merge_CpG
--zero_based`` (see ``assets/bioinfopipe_pairedWGBS.sh``). Columns, without a header:
sequence, start, end, beta (%), methylated reads, unmethylated reads.
"""

import pandas as pd


def read_sample(filepath):
    """Rows on autosomes, with sequence names written ``chr1`` .. ``chr22``."""
    target_chr = [str(i) for i in range(1, 23)]
    df = pd.read_csv(filepath, header=None, sep="\t",
                     names=["seqname", "start", "end", "beta", "num_methylated", "num_unmethylated"],
                     dtype={"seqname": "str"})
    df = df[df["seqname"].isin(target_chr)].reset_index(drop=True)
    df["seqname"] = "chr" + df["seqname"]
    return df


def get_processed_df(samp_file, chrom):
    """One chromosome of a sample, indexed by ``start``, with beta in [0, 1] and ``read_depth``."""
    df = read_sample(samp_file)
    chr_df = df[df["seqname"] == chrom].copy()
    chr_df["beta"] = chr_df["beta"] / 100
    chr_df["read_depth"] = chr_df[["num_methylated", "num_unmethylated"]].sum(axis=1)
    return chr_df.set_index("start")
