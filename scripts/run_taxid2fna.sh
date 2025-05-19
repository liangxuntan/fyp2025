#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Run taxid2fna.sh for one or more NCBI TaxIDs.
# Prints a "Processing TaxID …" line to the console, but redirects all
# script output (stdout + stderr) into one combined log file.
# ---------------------------------------------------------------------------

if [[ "$#" -eq 0 ]]; then
  echo "Usage: $0 <taxid1> [taxid2 taxid3 ...]" >&2
  exit 1
fi

script_dir="/Users/liangxuntan/Code/fyp2025/scripts"
runlog_dir="/Users/liangxuntan/Code/fyp2025/data/logs"
mkdir -p "$runlog_dir"

# Timestamped log file
log_file="${runlog_dir}/taxid2fna_$(date +%Y%m%d_%H%M%S).log"

cd "$script_dir"
source ~/anaconda3/bin/activate ncbi_datasets

for taxid in "$@"; do
  echo "→ Processing TaxID ${taxid}"          # shown on console
  ./taxid2fna.sh "${taxid}" >>"$log_file" 2>&1   # only in log
done

conda deactivate 2>/dev/null || true
echo "✓ Finished. Full log: $log_file"