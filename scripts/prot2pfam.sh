#!/bin/bash
# This script converts a FASTA file to a PFAM file using hmmscan.
set -euo pipefail

cd /Users/liangxuntan/Code/fyp2025/data/hmm_scoringmodel/

# --- input check -------------------------------------------------------------
protfile="${1:-}"  # path/to/file.fna
if [[ -z "$protfile" ]]; then
  echo "Usage: $0 <protfile>"
  exit 1
fi

basename=$(basename "$protfile")    # filename only
idnum="${basename%%_*}"            # part before first underscore
logfile="hmmscan_${idnum}.log"

# --- Run hmmscan -------------------------------------------------------------
start=$(date +%s)

hmmscan --cpu 8 -E 1e-20 --domE 1e-100 --pfamtblout "${idnum}_table.txt" Pfam-A.hmm "$protfile" > "$logfile" 2>&1

end=$(date +%s)
echo "hmmscan completed in $((end - start)) seconds."

# --- Move results ------------------------------------------------------------
dest_dir="/Users/liangxuntan/Code/fyp2025/data/hmmer_output"
mkdir -p "$dest_dir"
mv "${idnum}_table.txt" "$dest_dir/"
echo "Output saved to ${dest_dir}/${idnum}_table.txt"