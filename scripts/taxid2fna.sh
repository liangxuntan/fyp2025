#!/usr/bin/env bash
set -euo pipefail

cd /Users/liangxuntan/Code/fyp2025/data/ || {
  echo "Error: Failed to change to data directory."
  exit 1
}

# --- input check -------------------------------------------------------------
taxid="${1:-}"
if [[ -z "$taxid" ]]; then
  echo "Usage: $0 <taxid>"
  exit 1
fi

# --- tool checks -------------------------------------------------------------
if ! command -v datasets >/dev/null 2>&1; then
  echo "Error: 'datasets' CLI not found."
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "Error: 'unzip' not found."
  exit 1
fi

# --- create temp dir ---------------------------------------------------------
tmpdir=$(mktemp -d "tempfiles_${taxid}_XXXX") || {
  echo "Error: Failed to create temporary directory."
  exit 1
}

# --- download genome ---------------------------------------------------------
if ! datasets download genome taxon "$taxid" --reference --filename "${taxid}_genomes.zip"; then
  echo "Error: Failed to download genome for taxid $taxid."
  rm -rf "$tmpdir"
  exit 1
fi

# --- unzip genome ------------------------------------------------------------
if ! unzip -q "${taxid}_genomes.zip" -d "$tmpdir"; then
  echo "Error: Failed to unzip downloaded genome."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
fi

# --- locate genome directory -------------------------------------------------
data_dir=$(find "$tmpdir"/ncbi_dataset/data -type d -name "GCF*" | head -n 1)

if [[ -z "$data_dir" ]]; then
  echo "Error: No genome directory found for taxid $taxid."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
fi

# --- find .fna file ----------------------------------------------------------
fna_path=$(find "$data_dir" -maxdepth 1 -name "*.fna" | head -n 1)

if [[ -z "$fna_path" ]]; then
  echo "Error: No .fna file found for taxid $taxid."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
fi

# --- move and rename ---------------------------------------------------------
dest_dir="/Users/liangxuntan/Code/fyp2025/data/labelled_genomes"
mkdir -p "$dest_dir" || {
  echo "Error: Failed to create destination directory."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
}

gcf_id=$(basename "$fna_path")
if ! mv "$fna_path" "${dest_dir}/${taxid}_${gcf_id}"; then
  echo "Error: Failed to move .fna file."
  rm -rf "$tmpdir" "${taxid}_genomes.zip"
  exit 1
fi

# --- cleanup -----------------------------------------------------------------
rm -rf "$tmpdir" "${taxid}_genomes.zip" || {
  echo "Warning: Failed to clean up temporary files."
}
