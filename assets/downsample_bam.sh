#!/usr/bin/env bash
# Make low-depth versions of a sample from its deduplicated Bismark BAM.
#
# Run assets/bioinfopipe_pairedWGBS.sh with --keep-intermediates first, so the BAM still
# exists. Each fraction is subsampled with samtools view, which keeps or drops a read pair
# together, then passes through the same methylation calling as the full-depth data.
#
#   ./downsample_bam.sh --bam sample.deduplicated.bam --genome /path/to/hg38 \
#       --out-root ./runs --sample-id ENCFF000XXX --layout paired
set -euo pipefail

SEED=11
FRACS="0.1 0.2 0.4 0.6"
CORES=12
OUT_ROOT="$(pwd)"
LAYOUT=paired
BAM=""; GENOME=""; SAMPLE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bam) BAM="$2"; shift 2;;
    --genome) GENOME="$2"; shift 2;;
    --sample-id) SAMPLE_ID="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --layout) LAYOUT="$2"; shift 2;;     # paired or single
    --fracs) FRACS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --cores) CORES="$2"; shift 2;;
    *) echo "[ERROR] Unknown argument: $1"; exit 1;;
  esac
done
if [[ -z "${BAM}" || -z "${GENOME}" || -z "${SAMPLE_ID}" ]]; then
  echo "[ERROR] --bam, --genome and --sample-id are required."; exit 1
fi

if [[ "${LAYOUT}" == "paired" ]]; then
  EXTRACT_FLAGS=(--paired-end --no_overlap --ignore_r2 2)
else
  EXTRACT_FLAGS=(--single-end)
fi

for FRAC in ${FRACS}; do
  # samtools reads -s <seed>.<fraction>: 11.1 is seed 11, keep 10%.
  OUT_DIR="${OUT_ROOT}/frac${FRAC}_seed${SEED}/${SAMPLE_ID}"
  mkdir -p "${OUT_DIR}"
  STEM="$(basename "${BAM}" .bam).downsampled_frac${FRAC}_seed${SEED}"
  samtools view -b -@ "${CORES}" -s "${SEED}.${FRAC#0.}" "${BAM}" -o "${OUT_DIR}/${STEM}.bam"

  bismark_methylation_extractor "${EXTRACT_FLAGS[@]}" \
    --gzip --multicore "${CORES}" --bedGraph --zero_based \
    --output "${OUT_DIR}" "${OUT_DIR}/${STEM}.bam"

  coverage2cytosine --merge_CpG --zero_based \
    --genome_folder "${GENOME}" --dir "${OUT_DIR}" \
    -o "${STEM}.bismark.cpgMerged" "${OUT_DIR}/${STEM}.bismark.cov.gz"
done
