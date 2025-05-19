#!/usr/bin/env bash
set -euo pipefail

cd /Users/liangxuntan/Code/fyp2025/data/

# --- input check -------------------------------------------------------------
taxid="${1:-}"
if [[ -z "$taxid" ]]; then
  echo "Usage: $0 <taxid>"
  exit 1
fi

tmpdir="tempfiles_${taxid}"

# --- download reference genome ----------------------------------------------
datasets download genome taxon "$taxid" \
        --reference \
        --filename "${taxid}_genomes.zip"

# --- unzip into a throw-away folder -----------------------------------------
unzip -q "${taxid}_genomes.zip" -d "$tmpdir"

# --- find the first .fna file ------------------------------------------------
fna_path=$(find "$tmpdir"/ncbi_dataset/data/GCF* -maxdepth 1 -name "*.fna" | head -n 1)

if [[ -z "$fna_path" ]]; then
  echo "No .fna file found for taxid $taxid."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
fi

# --- rename & move to destination -------------------------------------------
dest_dir="/Users/liangxuntan/Code/fyp2025/data/labelled_genomes"
mkdir -p "$dest_dir"

gcf_id=$(basename "$fna_path")           # e.g. GCF_000001405.40_GRCh38.p14_genomic.fna
mv "$fna_path" "${dest_dir}/${taxid}_${gcf_id}"

# --- cleanup -----------------------------------------------------------------
rm -rf "$tmpdir" "${taxid}_genomes.zip"

echo "Saved: ${dest_dir}/${taxid}_${gcf_id}"